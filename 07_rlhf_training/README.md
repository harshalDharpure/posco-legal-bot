# 7. RLHF Training (PPO + DPO)

This module implements Reinforcement Learning from Human Feedback using PPO and DPO for legal correctness.

## Components

- `reward_model.py`: Reward model design with legal factuality, citation accuracy, and language fluency
- `ppo_training.py`: PPO loop for legal correctness
- `dpo_training.py`: DPO preference pair structure and training
- `safety_layer.py`: Safety layer design to avoid harmful legal advice

## Reward Components

- Legal factuality reward (40%)
- Citation accuracy reward (30%)
- Language fluency reward (20%)
- Safety reward (10%)

## Usage

```bash
# Train reward model
python reward_model.py --dataset data/preferences.json

# PPO training
python ppo_training.py --model models/lora_legal --reward-model models/reward_model

# DPO training
python dpo_training.py --dataset data/preference_pairs.json
```

