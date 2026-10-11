"""One shared LoRA fine-tuning recipe for acquisition baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.data.mnli import MNLIExample
from src.models.qwen_nli import QwenNLIStudent


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


def train_lora(
    student: QwenNLIStudent,
    examples: list[MNLIExample],
    config: LoraTrainingConfig,
    adapter_output_dir: str | Path,
    seed: int,
) -> float:
    """Fine-tune only LoRA parameters and save the resulting adapter."""
    if not examples:
        raise ValueError("LoRA training requires at least one acquired example")
    if any(example.label is None for example in examples):
        raise ValueError("All acquired MNLI examples must have labels")

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

    trainable_parameters = [
        parameter for parameter in student.model.parameters() if parameter.requires_grad
    ]
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
        encoded.append((example.id, input_ids, labels))

    def collate(
        rows: list[tuple[str, list[int], list[int]]]
    ) -> dict[str, Any]:
        width = max(len(input_ids) for _, input_ids, _ in rows)
        input_batch = torch.full(
            (len(rows), width),
            student.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        label_batch = torch.full(
            (len(rows), width), -100, dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for row_index, (_, input_ids, labels) in enumerate(rows):
            length = len(input_ids)
            input_batch[row_index, :length] = torch.tensor(input_ids)
            label_batch[row_index, :length] = torch.tensor(labels)
            attention_mask[row_index, :length] = 1
        return {
            "example_ids": [example_id for example_id, _, _ in rows],
            "input_ids": input_batch,
            "labels": label_batch,
            "attention_mask": attention_mask,
        }

    generator = torch.Generator()
    generator.manual_seed(seed)
    data_loader = DataLoader(
        encoded,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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
    for epoch_index in range(config.num_epochs):
        num_batches = len(data_loader)
        batches = tqdm(
            data_loader,
            total=num_batches,
            desc=f"LoRA training {epoch_index + 1}/{config.num_epochs}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=2.0,
        )
        for batch_index, batch in enumerate(batches):
            example_ids = batch.pop("example_ids")
            batch = {
                key: tensor.to(student.input_device)
                for key, tensor in batch.items()
            }
            labels = batch.pop("labels")
            logits = student.model(**batch, use_cache=False).logits
            shifted_logits = logits[:, :-1, :].contiguous().float()
            shifted_labels = labels[:, 1:].contiguous()
            active = shifted_labels != -100
            if not active.any():
                raise RuntimeError("Training batch contains no target tokens")
            loss = torch.nn.functional.cross_entropy(
                shifted_logits[active], shifted_labels[active]
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss "
                    f"at epoch={epoch_index}, batch={batch_index}, "
                    f"example_ids={example_ids}: {loss.item()}"
                )
            total_loss += loss.detach().float().item()
            loss_count += 1
            batches.set_postfix(loss=f"{loss.detach().float().item():.4f}", refresh=False)
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

    if original_use_cache is not None:
        student.model.config.use_cache = original_use_cache
    student.model.eval()
    output_path = Path(adapter_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    student.model.save_pretrained(output_path)
    return total_loss / loss_count
