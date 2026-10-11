"""Qwen3 causal-LM wrapper for deterministic three-way NLI prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.data.labels import MNLI_LABELS


class QwenNLIStudent:
    """Score the three NLI label strings by conditional log-likelihood.

    This avoids unconstrained text generation: every example receives exactly
    one of entailment, neutral, or contradiction. An optional PEFT adapter can
    be loaded for later checkpoints, but training is intentionally outside
    Milestone 1.
    """

    def __init__(
        self,
        name_or_path: str = "Qwen/Qwen3-1.7B",
        adapter_path: str | None = None,
        device_map: str | None = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if max_length < 32:
            raise ValueError("max_length must be at least 32")
        self._torch = torch
        self.max_length = max_length
        self.progress_label = "NLI inference"
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_value: Any
        if dtype == "auto":
            dtype_value = "auto"
        else:
            try:
                dtype_value = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(f"Unsupported torch dtype: {dtype!r}") from exc

        model_kwargs: dict[str, Any] = {
            "dtype": dtype_value,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path, **model_kwargs
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=True
            )
        self.model.eval()

    @property
    def input_device(self) -> Any:
        """Device holding the input embeddings (also works with device maps)."""
        return self.model.get_input_embeddings().weight.device

    def format_prompt(self, premise: str, hypothesis: str) -> str:
        """Format the shared inference/training prompt for one NLI pair."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify natural-language inference. Choose exactly one "
                    "label: entailment, neutral, or contradiction."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Premise: {premise}\n"
                    f"Hypothesis: {hypothesis}\n"
                    "Label:"
                ),
            },
        ]
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        return (
            "Classify the relationship as entailment, neutral, or contradiction.\n"
            f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
        )

    def encode_training_example(
        self,
        premise: str,
        hypothesis: str,
        label: str,
        max_length: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """Encode prompt and target, masking prompt tokens in the loss labels."""
        if label not in MNLI_LABELS:
            raise ValueError(f"Unsupported MNLI training label: {label!r}")
        limit = self.max_length if max_length is None else max_length
        prompt_ids = self.tokenizer.encode(
            self.format_prompt(premise, hypothesis), add_special_tokens=False
        )
        target_ids = self.tokenizer.encode(" " + label, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            target_ids.append(self.tokenizer.eos_token_id)
        available = limit - len(target_ids)
        if available < 1:
            raise ValueError("max_length is too short for the training target")
        prompt_ids = self._truncate_prompt(prompt_ids, available)
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return input_ids, labels

    def _truncate_prompt(self, token_ids: list[int], available: int) -> list[int]:
        if len(token_ids) <= available:
            return token_ids
        # Preserve both the instruction prefix and the final hypothesis/answer cue.
        prefix_size = available // 3
        return token_ids[:prefix_size] + token_ids[-(available - prefix_size) :]

    def _score_candidates(self, prompts: Sequence[str]) -> list[list[float]]:
        torch = self._torch
        sequences: list[list[int]] = []
        candidate_starts: list[int] = []

        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            for label in MNLI_LABELS:
                candidate_ids = self.tokenizer.encode(
                    " " + label, add_special_tokens=False
                )
                if not candidate_ids:
                    raise RuntimeError(f"Tokenizer produced no tokens for {label!r}")
                available = self.max_length - len(candidate_ids)
                if available < 1:
                    raise ValueError("max_length is too short for label candidates")
                shortened = self._truncate_prompt(prompt_ids, available)
                candidate_starts.append(len(shortened))
                sequences.append(shortened + candidate_ids)

        pad_id = self.tokenizer.pad_token_id
        max_sequence_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_sequence_length),
            pad_id,
            dtype=torch.long,
            device=self.input_device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            sequence_tensor = torch.tensor(
                sequence, dtype=torch.long, device=self.input_device
            )
            input_ids[row, : len(sequence)] = sequence_tensor
            attention_mask[row, : len(sequence)] = 1

        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        flat_scores: list[float] = []
        for row, (sequence, start) in enumerate(
            zip(sequences, candidate_starts, strict=True)
        ):
            if start == 0:
                raise RuntimeError("Prompt must contain at least one token")
            token_scores = []
            for position in range(start, len(sequence)):
                token_scores.append(
                    torch.log_softmax(
                        logits[row, position - 1].float(), dim=-1
                    )[sequence[position]]
                )
            flat_scores.append(torch.stack(token_scores).mean().item())

        width = len(MNLI_LABELS)
        return [
            flat_scores[offset : offset + width]
            for offset in range(0, len(flat_scores), width)
        ]

    def predict(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int = 4,
    ) -> list[str]:
        """Return one canonical MNLI label for each premise/hypothesis pair."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        from tqdm.auto import tqdm

        num_examples = len(premises)
        num_batches = (num_examples + batch_size - 1) // batch_size
        label = getattr(self, "progress_label", "NLI inference")
        print(
            f"[PROGRESS] {label}: {num_examples:,} examples, "
            f"{num_examples * len(MNLI_LABELS):,} candidate sequences, "
            f"{num_batches:,} batches (example batch_size={batch_size}, "
            f"candidate microbatch<={batch_size * len(MNLI_LABELS)})"
        )
        predictions: list[str] = []
        batch_starts = tqdm(
            range(0, num_examples, batch_size),
            total=num_batches,
            desc=label,
            unit="batch",
            dynamic_ncols=True,
            mininterval=2.0,
        )
        for start in batch_starts:
            prompts = [
                self.format_prompt(premise, hypothesis)
                for premise, hypothesis in zip(
                    premises[start : start + batch_size],
                    hypotheses[start : start + batch_size],
                    strict=True,
                )
            ]
            scores = self._score_candidates(prompts)
            predictions.extend(
                MNLI_LABELS[max(range(len(row)), key=row.__getitem__)]
                for row in scores
            )
            batch_starts.set_postfix(
                examples=f"{min(start + batch_size, num_examples):,}/{num_examples:,}",
                refresh=False,
            )
        return predictions

    def predict_proba(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int = 4,
    ) -> list[list[float]]:
        """Return normalized probabilities in ``MNLI_LABELS`` order.

        The class distribution is the softmax over normalized conditional
        label log-likelihoods used by :meth:`predict`.
        """
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        from tqdm.auto import tqdm

        num_examples = len(premises)
        num_batches = (num_examples + batch_size - 1) // batch_size
        label = getattr(self, "progress_label", "NLI probability scoring")
        print(
            f"[PROGRESS] {label}: {num_examples:,} examples, "
            f"{num_examples * len(MNLI_LABELS):,} candidate sequences, "
            f"{num_batches:,} batches (example batch_size={batch_size}, "
            f"candidate microbatch<={batch_size * len(MNLI_LABELS)})"
        )
        probabilities: list[list[float]] = []
        torch = self._torch
        batch_starts = tqdm(
            range(0, num_examples, batch_size),
            total=num_batches,
            desc=label,
            unit="batch",
            dynamic_ncols=True,
            mininterval=2.0,
        )
        for start in batch_starts:
            prompts = [
                self.format_prompt(premise, hypothesis)
                for premise, hypothesis in zip(
                    premises[start : start + batch_size],
                    hypotheses[start : start + batch_size],
                    strict=True,
                )
            ]
            for row in self._score_candidates(prompts):
                normalized = torch.softmax(
                    torch.tensor(row, dtype=torch.float64), dim=-1
                )
                probabilities.append(normalized.tolist())
            batch_starts.set_postfix(
                examples=f"{min(start + batch_size, num_examples):,}/{num_examples:,}",
                refresh=False,
            )
        return probabilities
