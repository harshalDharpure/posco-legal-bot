# 8. Multi-Bot Model Architecture

This module implements 4 coordinated LLM modules for specialized legal tasks.

## Components

- `legal_q_bot.py`: Generates legal answers
- `citation_bot.py`: Adds IPC/CrPC/Constitution sections
- `translation_bot.py`: Handles Hindi ↔ English translation
- `validator_bot.py`: RLHF-trained hallucination detector
- `multi_bot_coordinator.py`: Coordinates all bots
- `architecture_diagram.txt`: Multi-agent architecture diagram

## Bot Architecture

1. **Legal-Q-Bot**: Primary answer generation
2. **Citation-Bot**: Citation extraction and addition
3. **Translation-Bot**: Multilingual support
4. **Validator-Bot**: Quality and safety validation

## Usage

```bash
# Run multi-bot system
python multi_bot_coordinator.py \
    --query "What is IPC Section 302?" \
    --language en
```

