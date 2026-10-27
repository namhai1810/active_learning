"""One shared LoRA fine-tuning recipe for acquisition baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.data.mnli import MNLIExample
from src.data.labels import MNLI_LABELS
from src.models.qwen_nli import QwenNLIStudent
from src.synthetic.types import SYNTHETIC_PAIR_MEMBERS, SyntheticNLIExample


@dataclass(frozen=True)
class LoraTrainingConfig:
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    gradient_checkpointing: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LoraTrainingConfig":
        target_modules = value.get("target_modules")
        if not isinstance(target_modules, list) or not target_modules:
            raise ValueError("training.target_modules must be a non-empty list")
        config = cls(
            num_epochs=int(value.get("num_epochs", 1)),
            batch_size=int(value.get("batch_size", 2)),
            gradient_accumulation_steps=int(
                value.get("gradient_accumulation_steps", 8)
            ),
            learning_rate=float(value.get("learning_rate", 2e-4)),
            weight_decay=float(value.get("weight_decay", 0.0)),
            max_grad_norm=float(value.get("max_grad_norm", 1.0)),
            max_length=int(value.get("max_length", 512)),
            lora_r=int(value.get("lora_r", 8)),
            lora_alpha=int(value.get("lora_alpha", 16)),
            lora_dropout=float(value.get("lora_dropout", 0.05)),
            target_modules=tuple(str(item) for item in target_modules),
            gradient_checkpointing=bool(
                value.get("gradient_checkpointing", True)
            ),
        )
        positive_fields = {
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "max_grad_norm": config.max_grad_norm,
            "max_length": config.max_length,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        }
        invalid = [name for name, item in positive_fields.items() if item <= 0]
        if invalid:
            raise ValueError(f"Training values must be positive: {invalid}")
        if config.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= config.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        return config


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Validation-loss early stopping for a single stateful training call."""

    max_epochs: int
    patience: int
    min_delta: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EarlyStoppingConfig":
        early = value.get("early_stopping")
        if not isinstance(early, Mapping) or early.get("enabled") is not True:
            raise ValueError("training.early_stopping.enabled must be true")
        config = cls(
            max_epochs=int(value.get("max_epochs", value.get("num_epochs", 1))),
            patience=int(early.get("patience", 2)),
            min_delta=float(early.get("min_delta", 0.0)),
        )
        if config.max_epochs <= 0 or config.patience <= 0:
            raise ValueError("Early-stopping max_epochs and patience must be positive")
        if config.min_delta < 0:
            raise ValueError("Early-stopping min_delta must be non-negative")
        return config


@dataclass
class EarlyStoppingTracker:
    """Small deterministic state machine, kept separate for unit testing."""

    config: EarlyStoppingConfig
    best_loss: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def observe(self, epoch: int, validation_loss: float) -> tuple[bool, bool]:
        import math

        if epoch <= 0 or not math.isfinite(validation_loss):
            raise ValueError("Epoch and validation loss must be positive and finite")
        improved = validation_loss < self.best_loss - self.config.min_delta
        if improved:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.config.patience


@dataclass(frozen=True)
class PairwiseLossConfig:
    token_cross_entropy_weight: float = 1.0
    contrastive_weight: float = 1.0
    consistency_weight: float = 1.0
    contrastive_margin: float = 0.25

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PairwiseLossConfig":
        config = cls(
            token_cross_entropy_weight=float(
                value.get("token_cross_entropy_weight", 1.0)
            ),
            contrastive_weight=float(value.get("contrastive_weight", 1.0)),
            consistency_weight=float(value.get("consistency_weight", 1.0)),
            contrastive_margin=float(value.get("contrastive_margin", 0.25)),
        )
        if config.token_cross_entropy_weight <= 0:
            raise ValueError("token_cross_entropy_weight must be positive")
        if config.contrastive_weight <= 0 or config.consistency_weight <= 0:
            raise ValueError("Pairwise loss weights must be positive")
        if not 0.0 < config.contrastive_margin < 0.6931471805599453:
            raise ValueError("contrastive_margin must be in (0, log(2))")
        return config


@dataclass(frozen=True)
class PairedTrainingResult:
    total_loss: float
    token_cross_entropy_loss: float
    contrastive_loss: float
    consistency_loss: float
    counterfactual_pairs: int
    invariant_pairs: int
    reasoning_contrast_pairs: int
    optimizer_state_loaded: bool = False
    optimizer_step_start: int = 0
    optimizer_step_end: int = 0
    optimizer_step_last_attempted: int = 0
    epochs_completed: int = 0
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    stopped_early: bool = False
    validation_seconds: float = 0.0
    epoch_history: tuple[dict[str, Any], ...] = ()


def _optimizer_step(optimizer: Any) -> int:
    """Return the common AdamW step, rejecting internally inconsistent state."""
    steps: set[int] = set()
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        steps.add(int(value))
    if len(steps) > 1:
        raise RuntimeError(f"Optimizer parameters disagree on AdamW step: {steps}")
    return next(iter(steps), 0)


def _cpu_tree(value: Any) -> Any:
    """Detach tensor leaves to CPU so optimizer checkpoints are portable."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        # clone() is required for CPU-resident AdamW step tensors: detach().cpu()
        # can otherwise alias live optimizer state and mutate the best snapshot.
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _load_optimizer_checkpoint(
    optimizer: Any,
    path: str | Path,
    parameter_names: list[str],
) -> int:
    """Load a named AdamW checkpoint and return its completed step count."""
    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Optimizer checkpoint is missing: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format_version") != "named_adamw_v1":
        raise ValueError("Unsupported optimizer checkpoint format")
    saved_names = payload.get("parameter_names")
    if saved_names != parameter_names:
        raise ValueError(
            "Optimizer checkpoint parameter order differs from the loaded adapter"
        )
    state_dict = payload.get("optimizer_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Optimizer checkpoint has no optimizer_state_dict")
    optimizer.load_state_dict(state_dict)
    loaded_step = _optimizer_step(optimizer)
    declared_step = int(payload.get("optimizer_step", -1))
    if loaded_step != declared_step:
        raise ValueError(
            "Optimizer checkpoint step metadata does not match its tensor state"
        )
    return loaded_step


def _save_optimizer_checkpoint(
    optimizer: Any,
    path: str | Path,
    parameter_names: list[str],
) -> int:
    """Atomically save named AdamW state and return its completed step count."""
    import os
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_step = _optimizer_step(optimizer)
    payload = {
        "format_version": "named_adamw_v1",
        "parameter_names": parameter_names,
        "optimizer_step": completed_step,
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
    }
    temporary = output_path.with_name(output_path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return completed_step


def pairwise_distribution_losses(
    probabilities: Any,
    pair_types: list[str],
    gold_labels: list[str],
    contrastive_margin: float,
) -> tuple[Any, Any]:
    """Return contrastive and consistency losses for adjacent pair members."""
    import torch

    if probabilities.ndim != 2 or probabilities.shape[1] != len(MNLI_LABELS):
        raise ValueError("Pairwise probabilities must have shape [examples, 3]")
    if probabilities.shape[0] != 2 * len(pair_types):
        raise ValueError("Pairwise probabilities must contain two rows per pair")
    if len(gold_labels) != probabilities.shape[0]:
        raise ValueError("Pairwise gold-label count is inconsistent")
    if any(pair_type not in SYNTHETIC_PAIR_MEMBERS for pair_type in pair_types):
        raise ValueError("Pairwise loss received an unknown pair type")
    left = probabilities[0::2]
    right = probabilities[1::2]
    midpoint = 0.5 * (left + right)
    left_log = left.clamp_min(1e-12).log()
    right_log = right.clamp_min(1e-12).log()
    log_midpoint = midpoint.clamp_min(1e-12).log()
    js_divergence = 0.5 * (
        (left * (left_log - log_midpoint)).sum(dim=-1)
        + (right * (right_log - log_midpoint)).sum(dim=-1)
    )
    contrastive_indices = []
    consistency_indices = []
    for pair_index, pair_type in enumerate(pair_types):
        labels_differ = (
            gold_labels[2 * pair_index] != gold_labels[2 * pair_index + 1]
        )
        if pair_type == "counterfactual" or (
            pair_type == "reasoning_contrast" and labels_differ
        ):
            contrastive_indices.append(pair_index)
        else:
            consistency_indices.append(pair_index)
    zero = probabilities.sum() * 0.0
    contrastive_loss = zero
    consistency_loss = zero
    if contrastive_indices:
        selected = js_divergence[
            torch.tensor(
                contrastive_indices,
                dtype=torch.long,
                device=js_divergence.device,
            )
        ]
        contrastive_loss = torch.relu(contrastive_margin - selected).mean()
    if consistency_indices:
        consistency_loss = js_divergence[
            torch.tensor(
                consistency_indices,
                dtype=torch.long,
                device=js_divergence.device,
            )
        ].mean()
    return contrastive_loss, consistency_loss


def _train_lora(
    student: QwenNLIStudent,
    examples: list[MNLIExample],
    config: LoraTrainingConfig,
    adapter_output_dir: str | Path,
    seed: int,
    *,
    pair_types: list[str] | None = None,
    pairwise_config: PairwiseLossConfig | None = None,
    optimizer_state_input: str | Path | None = None,
    optimizer_state_output: str | Path | None = None,
    early_stopping: EarlyStoppingConfig | None = None,
    validation_callback: Callable[[], Mapping[str, Any]] | None = None,
) -> PairedTrainingResult:
    """Fine-tune only LoRA parameters and save the resulting adapter."""
    if not examples:
        raise ValueError("LoRA training requires at least one acquired example")
    if any(example.label is None for example in examples):
        raise ValueError("All acquired MNLI examples must have labels")
    paired = pairwise_config is not None
    if paired:
        if pair_types is None or len(pair_types) * 2 != len(examples):
            raise ValueError("Paired training requires one type per complete pair")
        if config.batch_size < 2 or config.batch_size % 2:
            raise ValueError("Paired training batch_size must be positive and even")
    elif pair_types is not None:
        raise ValueError("pair_types require a pairwise loss configuration")
    if (early_stopping is None) != (validation_callback is None):
        raise ValueError(
            "Early stopping requires both a configuration and validation callback"
        )

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    if not isinstance(student.model, PeftModel):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
        )
        student.model = get_peft_model(student.model, peft_config)

    # Keep optimizer states and updates in FP32. With BF16 LoRA parameters,
    # AdamW's small second moments can round to zero and produce NaNs after the
    # first update for otherwise valid gradients.
    for parameter in student.model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()

    named_trainable_parameters = [
        (name, parameter)
        for name, parameter in student.model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameters = [parameter for _, parameter in named_trainable_parameters]
    trainable_parameter_names = [name for name, _ in named_trainable_parameters]
    if not trainable_parameters:
        raise RuntimeError("The model has no trainable LoRA parameters")

    encoded = []
    for example in examples:
        input_ids, labels = student.encode_training_example(
            example.premise,
            example.hypothesis,
            str(example.label),
            max_length=config.max_length,
        )
        try:
            label_start = next(
                index for index, token_label in enumerate(labels) if token_label != -100
            )
        except StopIteration as exc:
            raise RuntimeError(f"Training example {example.id!r} has no label tokens") from exc
        encoded.append(
            {
                "id": example.id,
                "input_ids": input_ids,
                "labels": labels,
                "label": str(example.label),
                "label_start": label_start,
            }
        )

    if paired:
        encoded_units = []
        assert pair_types is not None
        for pair_index, index in enumerate(range(0, len(encoded), 2)):
            unit = encoded[index : index + 2]
            for row in unit:
                row["pair_type"] = pair_types[pair_index]
            encoded_units.append(unit)
        unit_batch_size = config.batch_size // 2
    else:
        encoded_units = [[row] for row in encoded]
        unit_batch_size = config.batch_size

    def collate(
        units: list[list[dict[str, Any]]]
    ) -> dict[str, Any]:
        rows = [row for unit in units for row in unit]
        width = max(len(row["input_ids"]) for row in rows)
        input_batch = torch.full(
            (len(rows), width),
            student.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        label_batch = torch.full(
            (len(rows), width), -100, dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for row_index, row in enumerate(rows):
            input_ids = row["input_ids"]
            labels = row["labels"]
            length = len(input_ids)
            input_batch[row_index, :length] = torch.tensor(input_ids)
            label_batch[row_index, :length] = torch.tensor(labels)
            attention_mask[row_index, :length] = 1
        return {
            "example_ids": [str(row["id"]) for row in rows],
            "gold_labels": [str(row["label"]) for row in rows],
            "pair_types": (
                [str(unit[0]["pair_type"]) for unit in units] if paired else None
            ),
            "label_starts": torch.tensor(
                [int(row["label_start"]) for row in rows], dtype=torch.long
            ),
            "input_ids": input_batch,
            "labels": label_batch,
            "attention_mask": attention_mask,
        }

    generator = torch.Generator()
    generator.manual_seed(seed)
    data_loader = DataLoader(
        encoded_units,
        batch_size=unit_batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_state_loaded = optimizer_state_input is not None
    optimizer_step_start = (
        _load_optimizer_checkpoint(
            optimizer,
            optimizer_state_input,
            trainable_parameter_names,
        )
        if optimizer_state_input is not None
        else 0
    )
    label_first_token_ids: list[int] | None = None
    if paired:
        label_first_token_ids = []
        for label in MNLI_LABELS:
            token_ids = student.tokenizer.encode(
                " " + label, add_special_tokens=False
            )
            if not token_ids:
                raise RuntimeError(f"Tokenizer produced no tokens for label {label!r}")
            label_first_token_ids.append(int(token_ids[0]))
        if len(set(label_first_token_ids)) != len(MNLI_LABELS):
            raise RuntimeError(
                "Pairwise label distributions require distinct first tokens for labels"
            )

    original_use_cache = getattr(student.model.config, "use_cache", None)
    if config.gradient_checkpointing:
        student.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        student.model.enable_input_require_grads()
        student.model.config.use_cache = False

    student.model.train()
    optimizer.zero_grad(set_to_none=True)
    gradient_accumulators = {
        id(parameter): torch.zeros_like(
            parameter, dtype=torch.float32, device="cpu"
        )
        for parameter in trainable_parameters
    }
    accumulated_batches = 0
    total_loss = 0.0
    loss_count = 0
    token_loss_total = 0.0
    contrastive_loss_total = 0.0
    contrastive_loss_count = 0
    consistency_loss_total = 0.0
    consistency_loss_count = 0
    epoch_limit = (
        early_stopping.max_epochs if early_stopping is not None else config.num_epochs
    )
    tracker = EarlyStoppingTracker(early_stopping) if early_stopping else None
    best_parameters: list[Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    epoch_history: list[dict[str, Any]] = []
    validation_seconds = 0.0
    stopped_early = False
    for epoch_index in range(epoch_limit):
        epoch_loss_start = total_loss
        epoch_token_loss_start = token_loss_total
        epoch_loss_count_start = loss_count
        num_batches = len(data_loader)
        batches = tqdm(
            data_loader,
            total=num_batches,
            desc=f"LoRA training {epoch_index + 1}/{epoch_limit}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=2.0,
        )
        for batch_index, batch in enumerate(batches):
            example_ids = batch.pop("example_ids")
            gold_labels = batch.pop("gold_labels")
            batch_pair_types = batch.pop("pair_types")
            batch = {
                key: tensor.to(student.input_device)
                for key, tensor in batch.items()
            }
            labels = batch.pop("labels")
            label_starts = batch.pop("label_starts")
            logits = student.model(**batch, use_cache=False).logits
            shifted_logits = logits[:, :-1, :].contiguous().float()
            shifted_labels = labels[:, 1:].contiguous()
            active = shifted_labels != -100
            if not active.any():
                raise RuntimeError("Training batch contains no target tokens")
            token_loss = torch.nn.functional.cross_entropy(
                shifted_logits[active], shifted_labels[active]
            )
            contrastive_loss = token_loss * 0.0
            consistency_loss = token_loss * 0.0
            if paired:
                assert pairwise_config is not None
                assert label_first_token_ids is not None
                assert batch_pair_types is not None
                row_indices = torch.arange(
                    logits.shape[0], device=logits.device
                )
                first_label_logits = logits[
                    row_indices, label_starts.to(logits.device) - 1
                ].float()
                class_token_ids = torch.tensor(
                    label_first_token_ids, dtype=torch.long, device=logits.device
                )
                class_logits = first_label_logits.index_select(1, class_token_ids)
                probabilities = torch.softmax(class_logits, dim=-1)
                contrastive_loss, consistency_loss = pairwise_distribution_losses(
                    probabilities,
                    batch_pair_types,
                    gold_labels,
                    pairwise_config.contrastive_margin,
                )
                contrastive_pair_count = sum(
                    pair_type == "counterfactual"
                    or (
                        pair_type == "reasoning_contrast"
                        and gold_labels[2 * pair_index]
                        != gold_labels[2 * pair_index + 1]
                    )
                    for pair_index, pair_type in enumerate(batch_pair_types)
                )
                if contrastive_pair_count:
                    contrastive_loss_total += contrastive_loss.detach().float().item()
                    contrastive_loss_count += 1
                if contrastive_pair_count < len(batch_pair_types):
                    consistency_loss_total += consistency_loss.detach().float().item()
                    consistency_loss_count += 1
                loss = (
                    pairwise_config.token_cross_entropy_weight * token_loss
                    + pairwise_config.contrastive_weight * contrastive_loss
                    + pairwise_config.consistency_weight * consistency_loss
                )
            else:
                loss = token_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss "
                    f"at epoch={epoch_index}, batch={batch_index}, "
                    f"example_ids={example_ids}: {loss.item()}"
                )
            total_loss += loss.detach().float().item()
            token_loss_total += token_loss.detach().float().item()
            loss_count += 1
            batches.set_postfix(
                loss=f"{loss.detach().float().item():.4f}",
                ce=f"{token_loss.detach().float().item():.4f}",
                refresh=False,
            )
            gradients = torch.autograd.grad(
                loss,
                trainable_parameters,
                allow_unused=True,
            )
            for parameter, gradient in zip(
                trainable_parameters, gradients, strict=True
            ):
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        raise RuntimeError(
                            "Non-finite LoRA gradient "
                            f"at epoch={epoch_index}, batch={batch_index}, "
                            f"example_ids={example_ids}"
                        )
                    gradient_accumulators[id(parameter)].add_(
                        gradient.detach().float().cpu()
                    )
            accumulated_batches += 1
            # Keeping .grad attached across forwards is numerically unstable
            # with this torch/PEFT stack, so accumulation lives in FP32 buffers.
            student.model.zero_grad(set_to_none=True)

            should_step = (
                accumulated_batches == config.gradient_accumulation_steps
                or batch_index + 1 == num_batches
            )
            if should_step:
                for parameter in trainable_parameters:
                    parameter.grad = (
                        gradient_accumulators[id(parameter)] / accumulated_batches
                    ).to(device=parameter.device, dtype=parameter.dtype)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.max_grad_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(
                        "Non-finite LoRA gradient norm "
                        f"at epoch={epoch_index}, batch={batch_index}, "
                        f"example_ids={example_ids}"
                    )
                optimizer.step()
                if not all(
                    torch.isfinite(parameter).all()
                    for parameter in trainable_parameters
                ):
                    raise RuntimeError(
                        "Non-finite LoRA parameter after optimizer step "
                        f"at epoch={epoch_index}, batch={batch_index}, "
                        f"example_ids={example_ids}"
                    )
                student.model.zero_grad(set_to_none=True)
                for accumulator in gradient_accumulators.values():
                    accumulator.zero_()
                accumulated_batches = 0

        epoch_count = loss_count - epoch_loss_count_start
        epoch_record: dict[str, Any] = {
            "epoch": epoch_index + 1,
            "training_loss": (total_loss - epoch_loss_start) / epoch_count,
            "training_token_cross_entropy_loss": (
                token_loss_total - epoch_token_loss_start
            )
            / epoch_count,
            "optimizer_step_end": _optimizer_step(optimizer),
        }
        if tracker is not None:
            import time

            student.model.eval()
            validation_started = time.perf_counter()
            validation = dict(validation_callback())
            validation_seconds += time.perf_counter() - validation_started
            validation_loss = float(validation.get("loss", float("nan")))
            improved, should_stop = tracker.observe(epoch_index + 1, validation_loss)
            epoch_record["validation"] = validation
            epoch_record["improved"] = improved
            epoch_record["bad_epochs"] = tracker.bad_epochs
            if improved:
                best_parameters = [
                    parameter.detach().cpu().clone()
                    for parameter in trainable_parameters
                ]
                best_optimizer_state = _cpu_tree(optimizer.state_dict())
            epoch_history.append(epoch_record)
            if should_stop:
                stopped_early = True
                break
            student.model.train()
        else:
            epoch_history.append(epoch_record)

    optimizer_step_last_attempted = _optimizer_step(optimizer)
    if tracker is not None:
        if best_parameters is None or best_optimizer_state is None:
            raise RuntimeError("Early stopping completed without a best checkpoint")
        for parameter, best_value in zip(
            trainable_parameters, best_parameters, strict=True
        ):
            parameter.data.copy_(
                best_value.to(device=parameter.device, dtype=parameter.dtype)
            )
        optimizer.load_state_dict(best_optimizer_state)

    if original_use_cache is not None:
        student.model.config.use_cache = original_use_cache
    student.model.eval()
    output_path = Path(adapter_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    student.model.save_pretrained(output_path)
    optimizer_step_end = _optimizer_step(optimizer)
    if optimizer_state_output is not None:
        saved_step = _save_optimizer_checkpoint(
            optimizer,
            optimizer_state_output,
            trainable_parameter_names,
        )
        if saved_step != optimizer_step_end:
            raise RuntimeError("Saved optimizer step differs from in-memory state")
    best_epoch = tracker.best_epoch if tracker is not None else None
    selected_history = (
        epoch_history[best_epoch - 1] if best_epoch is not None else None
    )
    return PairedTrainingResult(
        total_loss=(
            float(selected_history["training_loss"])
            if selected_history is not None
            else total_loss / loss_count
        ),
        token_cross_entropy_loss=(
            float(selected_history["training_token_cross_entropy_loss"])
            if selected_history is not None
            else token_loss_total / loss_count
        ),
        contrastive_loss=(
            contrastive_loss_total / contrastive_loss_count
            if contrastive_loss_count
            else 0.0
        ),
        consistency_loss=(
            consistency_loss_total / consistency_loss_count
            if consistency_loss_count
            else 0.0
        ),
        counterfactual_pairs=(
            pair_types.count("counterfactual") if pair_types is not None else 0
        ),
        invariant_pairs=(
            pair_types.count("invariant") if pair_types is not None else 0
        ),
        reasoning_contrast_pairs=(
            pair_types.count("reasoning_contrast") if pair_types is not None else 0
        ),
        optimizer_state_loaded=optimizer_state_loaded,
        optimizer_step_start=optimizer_step_start,
        optimizer_step_end=optimizer_step_end,
        optimizer_step_last_attempted=optimizer_step_last_attempted,
        epochs_completed=len(epoch_history),
        best_epoch=best_epoch,
        best_validation_loss=(tracker.best_loss if tracker is not None else None),
        stopped_early=stopped_early,
        validation_seconds=validation_seconds,
        epoch_history=tuple(epoch_history),
    )


def train_lora(
    student: QwenNLIStudent,
    examples: list[MNLIExample],
    config: LoraTrainingConfig,
    adapter_output_dir: str | Path,
    seed: int,
) -> float:
    """Fine-tune with the shared per-example token cross-entropy objective."""
    return _train_lora(
        student, examples, config, adapter_output_dir, seed
    ).total_loss


def train_lora_stateful(
    student: QwenNLIStudent,
    examples: list[MNLIExample],
    config: LoraTrainingConfig,
    adapter_output_dir: str | Path,
    seed: int,
    *,
    optimizer_state_input: str | Path | None,
    optimizer_state_output: str | Path,
) -> PairedTrainingResult:
    """Train CE LoRA while carrying AdamW state across closed-loop rounds."""
    return _train_lora(
        student,
        examples,
        config,
        adapter_output_dir,
        seed,
        optimizer_state_input=optimizer_state_input,
        optimizer_state_output=optimizer_state_output,
    )


def train_lora_stateful_early_stopping(
    student: QwenNLIStudent,
    examples: list[MNLIExample],
    config: LoraTrainingConfig,
    early_stopping: EarlyStoppingConfig,
    validation_callback: Callable[[], Mapping[str, Any]],
    adapter_output_dir: str | Path,
    seed: int,
    *,
    optimizer_state_input: str | Path | None,
    optimizer_state_output: str | Path,
) -> PairedTrainingResult:
    """Train statefully and commit the best validation-loss epoch only."""
    return _train_lora(
        student,
        examples,
        config,
        adapter_output_dir,
        seed,
        optimizer_state_input=optimizer_state_input,
        optimizer_state_output=optimizer_state_output,
        early_stopping=early_stopping,
        validation_callback=validation_callback,
    )


def train_paired_lora(
    student: QwenNLIStudent,
    examples: list[SyntheticNLIExample],
    config: LoraTrainingConfig,
    pairwise_config: PairwiseLossConfig,
    adapter_output_dir: str | Path,
    seed: int,
) -> PairedTrainingResult:
    """Fine-tune on complete synthetic pairs with CE and pairwise objectives."""
    if not examples:
        raise ValueError("Paired LoRA training requires synthetic examples")
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    seen_ids: set[str] = set()
    for example in examples:
        if example.id in seen_ids:
            raise ValueError(f"Duplicate paired training ID: {example.id}")
        seen_ids.add(example.id)
        grouped.setdefault(example.pair_id, []).append(example)

    ordered: list[SyntheticNLIExample] = []
    pair_types: list[str] = []
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Paired training pair {pair_id!r} is incomplete")
        pair_type = members[0].pair_type
        if pair_type not in SYNTHETIC_PAIR_MEMBERS or any(
            member.pair_type != pair_type for member in members
        ):
            raise ValueError(f"Paired training pair {pair_id!r} has invalid type")
        expected_roles = SYNTHETIC_PAIR_MEMBERS[pair_type]
        if {member.pair_member for member in members} != set(expected_roles):
            raise ValueError(f"Paired training pair {pair_id!r} has invalid roles")
        members.sort(key=lambda member: expected_roles.index(member.pair_member))
        labels = {member.label for member in members}
        if pair_type == "counterfactual" and len(labels) != 2:
            raise ValueError("Counterfactual training pairs must change labels")
        if pair_type == "invariant" and len(labels) != 1:
            raise ValueError("Invariant training pairs must preserve labels")
        ordered.extend(members)
        pair_types.append(pair_type)

    return _train_lora(
        student,
        [item.to_mnli() for item in ordered],
        config,
        adapter_output_dir,
        seed,
        pair_types=pair_types,
        pairwise_config=pairwise_config,
    )
