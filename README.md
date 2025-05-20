# B3

This repo contains the code and data for B3. More details coming soon.

Preprint Out: https://arxiv.org/pdf/2505.11293

## 🧠 Trained Models

| Model Scale | Hugging Face Model Hub |
|-------------|-------------------------|
| 2B          | [raghavlite/B3_Qwen2_2B](https://huggingface.co/raghavlite/B3_Qwen2_2B) |
| 8B          | [raghavlite/B3_Qwen2_7B](https://huggingface.co/raghavlite/B3_Qwen2_7B) |


## Inference & Evaluation

Download the image file zip from huggingface
```bash
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
```

1. To evaluate our model on an MMEB dataset (e.g., MSCOCO_i2t), run:
```bash 
python  eval_mmeb.py  --model_name raghavlite/B3_Qwen2_7B --encode_output_path  ./MMEB-evaloutputs/B2_Qwen2_7B/  --pooling  eos  --normalize  True  --lora  --lora_r  8  --bf16  --dataset_name  TIGER-Lab/MMEB-eval  --subset_name  MSCOCO_i2t  --dataset_split  test  --per_device_eval_batch_size  4  --image_dir  eval_images/  --tgt_prefix_mod
```

## Acknowledgement
- We have adapted code from [VLM2Vec]([https://github.com/TIGER-AI-Lab/VLM2Vec]).
