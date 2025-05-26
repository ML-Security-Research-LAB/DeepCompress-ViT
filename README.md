# DeepCompress-ViT (CVPR 2025)
Code for DeepCompress-ViT: Rethinking Model Compression to Enhance Efficiency of Vision Transformers at the Edge
 

## Requirements
- Python 3.8.19  
- PyTorch 2.3.1
- torchvision 0.18.1
- timm 1.0.7

Install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Training

```bash
# Train a compressed DeiT-Small backbone (rank 277)
python main.py \
  --model_name deit_small_patch16_224 \
  --batch_size 256 \
  --epochs 275 \
  --eval_interval 20000 \
  --mixed_precision \
  --device cuda:0 \
  --base_dir small_rank_277 \
  --distillation_weight 3000 \
  --initial_iters 1000 \
  --finetune_other_params \
  --rank 277 \
  --distilled_model

# Train a compressed DeiT-Base backbone (rank 502)
python main.py \
  --model_name deit_base_patch16_224 \
  --batch_size 256 \
  --epochs 275 \
  --eval_interval 20000 \
  --mixed_precision \
  --device cuda:0 \
  --base_dir base_rank_502 \
  --distillation_weight 3000 \
  --initial_iters 1000 \
  --finetune_other_params \
  --rank 502 \
  --distilled_model
```

### 2. Download Pre-trained Weights

Download our pre-compressed checkpoints from Google Drive and extract into:

```
saved_models/
├── small_rank_277/
│   └── deit_small_patch16_224.pth
└── base_rank_502/
    └── deit_base_patch16_224.pth
```

### 3. Inference

```bash
# Inference with small model
python inference.py \
  --model_name deit_small_patch16_224 \
  --batch_size 256 \
  --device cuda:0 \
  --rank 277 \
  --mixed_precision \
  --state_path saved_models/small_rank_277/deit_small_patch16_224.pth

# Inference with base model
python inference.py \
  --model_name deit_base_patch16_224 \
  --batch_size 256 \
  --device cuda:0 \
  --rank 502 \
  --mixed_precision \
  --state_path saved_models/base_rank_502/deit_base_patch16_224.pth
```
