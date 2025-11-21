# 5. LoRA Fine-Tuning Configuration

This module implements parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) for legal domain adaptation.

## Components

- `lora_config.py`: LoRA hyperparameters and configuration
- `train_lora.py`: Full fine-tuning code template
- `legal_tuning_strategy.py`: Legal-domain parameter-efficient tuning strategy

## LoRA Configuration

- **r (rank)**: 16
- **alpha**: 32
- **dropout**: 0.05
- **target_modules**: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj

## Usage

```bash
# Train LoRA model
python train_lora.py \
    --model indiclegal-llama \
    --dataset data/splits/train.json \
    --output models/lora_legal \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

