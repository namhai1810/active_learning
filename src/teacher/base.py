"""Strict JSON teacher interface with a real local Hugging Face provider.

The research pipeline depends only on :class:`TeacherLLM`.  It never silently
falls back to HANS metadata or to fabricated responses when teacher generation
fails.  The concrete provider loads and invokes the configured model only when
the pipeline is explicitly executed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class TeacherCall:
    task: str
    prompt_version: str
    seed: int
    system_prompt: str
    payload: dict[str, Any]

    @property
    def request_id(self) -> str:
        serialized = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "request_id": self.request_id}


class TeacherLLM(Protocol):
    """Provider-neutral contract for one deterministic JSON response."""

    def generate_json(self, call: TeacherCall) -> dict[str, Any]: ...


def _parse_json_object(text: str) -> dict[str, Any]:
    """Require a pure JSON object; markdown or explanatory text is rejected."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Teacher response is not a valid standalone JSON value") from exc
    if not isinstance(value, dict):
        raise ValueError("Teacher response must be a JSON object")
    return value


class HuggingFaceJSONTeacher:
    """Run a configured causal LM locally and parse its response as strict JSON."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        name_or_path = config.get("name_or_path")
        if not isinstance(name_or_path, str) or not name_or_path.strip():
            raise ValueError(
                "teacher.name_or_path must identify an explicitly chosen teacher model"
            )
        self.name_or_path = name_or_path
        self.device_map = config.get("device_map", "auto")
        self.dtype = str(config.get("dtype", "bfloat16"))
        self.trust_remote_code = bool(config.get("trust_remote_code", False))
        self.local_files_only = bool(config.get("local_files_only", False))
        self.max_input_tokens = int(config.get("max_input_tokens", 8192))
        self.max_new_tokens = int(config.get("max_new_tokens", 2048))
        if self.max_input_tokens <= 0 or self.max_new_tokens <= 0:
            raise ValueError("Teacher token limits must be positive")
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._usage: dict[str, dict[str, Any]] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.name_or_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype: Any = "auto" if self.dtype == "auto" else getattr(torch, self.dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported teacher dtype: {self.dtype!r}")
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        self._tokenizer = tokenizer
        self._model = AutoModelForCausalLM.from_pretrained(
            self.name_or_path, **kwargs
        )
        self._model.eval()

    def generate_json(self, call: TeacherCall) -> dict[str, Any]:
        self._ensure_loaded()
        import torch
        from transformers import set_seed

        assert self._model is not None and self._tokenizer is not None
        set_seed(call.seed)
        user_content = json.dumps(call.payload, ensure_ascii=False, sort_keys=True)
        messages = [
            {"role": "system", "content": call.system_prompt},
            {"role": "user", "content": user_content},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = f"SYSTEM:\n{call.system_prompt}\nUSER:\n{user_content}\nASSISTANT:\n"
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )
        if encoded["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError(
                "Teacher prompt exceeds teacher.max_input_tokens; reduce the "
                "configured discovery/verification/generation batch size"
            )
        device = self._model.get_input_embeddings().weight.device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(
            generated[0, input_width:], skip_special_tokens=True
        )
        self._usage[call.request_id] = {
            "input_tokens": int(input_width),
            "output_tokens": int(generated.shape[1] - input_width),
            "wall_clock_seconds": time.perf_counter() - started,
            "provider": "huggingface",
            "model": self.name_or_path,
        }
        return _parse_json_object(response)

    def usage_for(self, request_id: str) -> dict[str, Any] | None:
        """Return measured token/runtime usage for a completed request."""
        value = self._usage.get(request_id)
        return None if value is None else dict(value)


def teacher_exchange_log(
    teacher: TeacherLLM, call: TeacherCall, response: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a provider-neutral request/response log with measured usage."""
    getter = getattr(teacher, "usage_for", None)
    usage = getter(call.request_id) if callable(getter) else None
    return {
        "teacher_call": call.to_dict(),
        "teacher_response": dict(response),
        "usage": usage,
    }


def build_teacher(config: Mapping[str, Any]) -> TeacherLLM:
    """Build the explicitly configured provider; unknown providers fail fast."""
    provider = str(config.get("provider", "huggingface")).strip().lower()
    if provider == "huggingface":
        return HuggingFaceJSONTeacher(config)
    raise ValueError(
        f"Unsupported teacher provider {provider!r}; no oracle fallback is available"
    )
