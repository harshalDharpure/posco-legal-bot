# GPU Commands Guide - Multilingual Legal Conversational Bot

Complete guide for running the project on GPU (NVIDIA CUDA).

## Table of Contents
1. [GPU Setup & Verification](#gpu-setup--verification)
2. [GPU Installation Commands](#gpu-installation-commands)
3. [GPU-Optimized Training Commands](#gpu-optimized-training-commands)
4. [GPU Memory Management](#gpu-memory-management)
5. [Multi-GPU Setup](#multi-gpu-setup)
6. [Performance Optimization](#performance-optimization)

---

## GPU Setup & Verification

### Check GPU Availability

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

**Expected Output:**
```
CUDA Available: True
CUDA Version: 11.8
GPU Count: 1
GPU Name: NVIDIA GeForce RTX 3090
```

### Check GPU Memory

```bash
# Check GPU memory
nvidia-smi

# Python script to check GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
```

### Set GPU Device

```bash
# Set CUDA device (if multiple GPUs)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
# export CUDA_VISIBLE_DEVICES=1  # Use second GPU
# export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
```

---

## GPU Installation Commands

### Install PyTorch with CUDA Support

```bash
# For CUDA 11.8 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Install GPU-Accelerated Libraries

```bash
# Install FAISS GPU version (instead of CPU)
pip uninstall faiss-cpu
pip install faiss-gpu

# Install CUDA-enabled transformers
pip install transformers[torch] accelerate

# Install bitsandbytes for quantization
pip install bitsandbytes

# Verify installations
python -c "import faiss; print('FAISS GPU:', hasattr(faiss, 'StandardGpuResources'))"
python -c "import bitsandbytes; print('BitsAndBytes:', bitsandbytes.__version__)"
```

### Complete GPU Setup Script

```bash
# Create virtual environment
python -m venv venv_gpu
source venv_gpu/bin/activate  # Linux/Mac
# venv_gpu\Scripts\activate  # Windows

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies with GPU support
pip install -r requirements.txt

# Replace FAISS CPU with GPU
pip uninstall faiss-cpu -y
pip install faiss-gpu

# Verify GPU setup
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

---

## GPU-Optimized Training Commands

### 1. Base Model Selection (GPU)

```bash
cd 01_base_model

# Load model on GPU with quantization
python model_selection.py --model indiclegal-llama --no-quantization

# Check GPU memory usage
watch -n 1 nvidia-smi
```

### 2. OCR Pipeline (CPU - No GPU needed)

```bash
cd 02_ocr_pipeline

# OCR runs on CPU (Tesseract)
python ocr_pipeline.py --input data/legal_docs/ --output data/ocr_output/
```

### 3. Dataset Creation (CPU - No GPU needed)

```bash
cd 03_dataset_creation

# Dataset creation is CPU-based
python dataset_builder.py --input data/clauses/ --output data/dataset/legal_qa.json
python train_test_split.py --input data/dataset/legal_qa.json --output data/splits/
```

### 4. RAG Pipeline (GPU-Accelerated)

```bash
cd 04_rag_pipeline

# Build FAISS index with GPU support
python faiss_index.py \
    --input data/cleaned/ \
    --output data/faiss_index/ \
    --index-type flat \
    --embedding-model paraphrase-multilingual

# GPU-accelerated embedding generation
# Embeddings will use GPU if available
```

**GPU-Optimized FAISS Index Building:**
```python
# Modify faiss_index.py to use GPU
import faiss

# Use GPU resources
res = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
```

### 5. LoRA Fine-Tuning (GPU - Most Important)

```bash
cd 05_lora_finetuning

# GPU-optimized LoRA training
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train_augmented.json \
    --eval-dataset ../data/splits/val.json \
    --output models/lora_legal \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4

# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

**GPU Memory Optimization for LoRA:**
```bash
# Use 4-bit quantization to reduce GPU memory
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train.json \
    --output models/lora_legal \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --use-4bit

# For 8GB GPU
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train.json \
    --output models/lora_legal \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --use-4bit \
    --max-length 512
```

### 6. RAG + LLM Fusion (GPU)

```bash
cd 06_rag_llm_fusion

# GPU-accelerated inference
python rag_llm_fusion.py \
    --query "What is IPC Section 302?" \
    --model ../models/lora_legal \
    --index-dir ../data/faiss_index/ \
    --top-k 5 \
    --device cuda
```

### 7. RLHF Training (GPU)

```bash
cd 07_rlhf_training

# PPO Training on GPU
python ppo_training.py \
    --model ../models/lora_legal \
    --reward-model models/reward_model \
    --dataset data/preferences.json \
    --output models/ppo_legal \
    --steps 1000 \
    --batch-size 4 \
    --device cuda

# DPO Training on GPU
python dpo_training.py \
    --model ../models/lora_legal \
    --dataset data/preference_pairs.json \
    --output models/dpo_legal \
    --epochs 3 \
    --device cuda
```

### 8. Multi-Bot System (GPU)

```bash
cd 08_multi_bot_architecture

# Run multi-bot system on GPU
python multi_bot_coordinator.py \
    --query "What is IPC Section 302?" \
    --model ../models/lora_legal \
    --language en \
    --device cuda
```

### 9. Evaluation (GPU for embeddings)

```bash
cd 09_evaluation

# Evaluation uses GPU for embedding generation
python evaluation_pipeline.py \
    --test-dataset ../data/splits/test.json \
    --output results/evaluation.json \
    --device cuda
```

---

## GPU Memory Management

### Check GPU Memory Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Python script to check memory
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
    print(f'GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB')
    print(f'GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.2f} GB')
"
```

### Clear GPU Cache

```bash
# Python command to clear GPU cache
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Or in your code
import torch
torch.cuda.empty_cache()
```

### Memory-Efficient Training

```bash
# Use gradient checkpointing
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train.json \
    --output models/lora_legal \
    --batch-size 2 \
    --gradient-checkpointing \
    --use-4bit

# Use mixed precision training
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train.json \
    --output models/lora_legal \
    --fp16 \
    --batch-size 4
```

### GPU Memory by Component

| Component | GPU Memory Usage | Optimization |
|-----------|-----------------|--------------|
| Base Model (7B) | ~14 GB (FP16) | Use 4-bit quantization (~4 GB) |
| LoRA Training | +2-4 GB | Reduce batch size |
| RAG Embeddings | ~2 GB | Batch processing |
| Inference | ~8 GB (FP16) | Use quantization |

---

## Multi-GPU Setup

### Data Parallel Training

```bash
# Use multiple GPUs for training
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train.json \
    --output models/lora_legal \
    --multi-gpu \
    --gpus 0,1,2,3
```

### Model Parallel (for very large models)

```python
# In train_lora.py, use model parallel
from torch.nn.parallel import DataParallel

model = DataParallel(model, device_ids=[0, 1, 2, 3])
```

### Distributed Training

```bash
# Use torch.distributed for multi-GPU
torchrun --nproc_per_node=4 train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train.json \
    --output models/lora_legal
```

---

## Performance Optimization

### Enable TensorFloat-32 (TF32)

```bash
# Enable TF32 for faster training (Ampere GPUs)
export NVIDIA_TF32_OVERRIDE=1

# Or in Python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Optimize CUDA Operations

```bash
# Set CUDA optimization flags
export CUDA_LAUNCH_BLOCKING=0  # Async execution
export TORCH_CUDNN_V8_API_ENABLED=1  # Use cuDNN v8
```

### Batch Size Optimization

```bash
# Find optimal batch size (start small, increase)
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train.json \
    --output models/lora_legal \
    --batch-size 1  # Start with 1

# If no OOM error, try:
--batch-size 2
--batch-size 4
--batch-size 8
```

### Mixed Precision Training

```bash
# Use FP16 for faster training
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train.json \
    --output models/lora_legal \
    --fp16 \
    --batch-size 4

# Use BF16 (better for training)
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train.json \
    --output models/lora_legal \
    --bf16 \
    --batch-size 4
```

---

## Complete GPU Pipeline Commands

### Full Pipeline with GPU

```bash
# 1. Setup GPU environment
python -m venv venv_gpu
source venv_gpu/bin/activate  # Linux/Mac
# venv_gpu\Scripts\activate  # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall faiss-cpu -y && pip install faiss-gpu

# 2. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# 3. Process documents (CPU)
cd 02_ocr_pipeline
python ocr_pipeline.py --input ../data/legal_docs/ --output ../data/ocr_output/
python ocr_cleaning.py --input ../data/ocr_output/ --output ../data/cleaned/
python clause_extraction.py --input ../data/cleaned/ --output ../data/clauses/

# 4. Create dataset (CPU)
cd ../03_dataset_creation
python dataset_builder.py --input ../data/clauses/ --output ../data/dataset/legal_qa.json
python train_test_split.py --input ../data/dataset/legal_qa.json --output ../data/splits/
python data_augmentation.py --input ../data/splits/train.json --output ../data/splits/train_augmented.json

# 5. Build RAG index (GPU-accelerated embeddings)
cd ../04_rag_pipeline
python faiss_index.py --input ../data/cleaned/ --output ../data/faiss_index/

# 6. Train LoRA model (GPU - Main training)
cd ../05_lora_finetuning
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train_augmented.json \
    --eval-dataset ../data/splits/val.json \
    --output ../models/lora_legal \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --fp16

# Monitor GPU during training
watch -n 1 nvidia-smi

# 7. Test RAG + LLM (GPU inference)
cd ../06_rag_llm_fusion
python rag_llm_fusion.py \
    --query "What is IPC Section 302?" \
    --model ../models/lora_legal \
    --index-dir ../data/faiss_index/ \
    --top-k 5

# 8. Run multi-bot system (GPU)
cd ../08_multi_bot_architecture
python multi_bot_coordinator.py \
    --query "What is IPC Section 302?" \
    --model ../models/lora_legal

# 9. Evaluate (GPU for embeddings)
cd ../09_evaluation
python evaluation_pipeline.py \
    --test-dataset ../data/splits/test.json \
    --output ../results/evaluation.json
```

---

## GPU Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Solution 1: Reduce batch size
python train_lora.py --batch-size 1 --gradient-accumulation-steps 16

# Solution 2: Use 4-bit quantization
python train_lora.py --use-4bit --batch-size 2

# Solution 3: Reduce sequence length
python train_lora.py --max-length 512 --batch-size 4

# Solution 4: Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Issue: CUDA Not Available

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Slow GPU Performance

```bash
# Enable TF32 (Ampere GPUs)
export NVIDIA_TF32_OVERRIDE=1

# Use mixed precision
python train_lora.py --fp16

# Optimize batch size
python train_lora.py --batch-size 8  # Increase if memory allows
```

### Issue: FAISS GPU Not Working

```bash
# Uninstall CPU version
pip uninstall faiss-cpu

# Install GPU version
pip install faiss-gpu

# Verify
python -c "import faiss; print(hasattr(faiss, 'StandardGpuResources'))"
```

---

## GPU Performance Benchmarks

### Expected Training Times (RTX 3090, 24GB)

| Component | Time | GPU Memory |
|-----------|------|------------|
| RAG Index Building (1000 docs) | ~5 min | ~2 GB |
| LoRA Training (1000 samples, 3 epochs) | ~2-3 hours | ~12 GB |
| RLHF PPO (1000 steps) | ~4-6 hours | ~14 GB |
| Inference (single query) | ~2-5 seconds | ~8 GB |

### Optimization Tips

1. **Use 4-bit quantization**: Reduces memory by 75%
2. **Gradient accumulation**: Simulates larger batch size
3. **Mixed precision (FP16)**: 2x faster training
4. **FAISS GPU**: 10-100x faster retrieval
5. **Batch inference**: Process multiple queries together

---

## Quick GPU Commands Reference

```bash
# Check GPU
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Train with GPU
python train_lora.py --device cuda --batch-size 4 --fp16

# Inference with GPU
python rag_llm_fusion.py --device cuda

# Check GPU memory
python -c "import torch; print(f'{torch.cuda.memory_allocated(0)/1e9:.2f} GB')"
```

---

**Note**: All commands assume you have NVIDIA GPU with CUDA support. For CPU-only execution, remove `--device cuda` flags and use CPU-compatible versions of libraries.

