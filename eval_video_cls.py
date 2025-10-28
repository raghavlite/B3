import json
import numpy as np
import os
import pickle
import sys
import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoConfig

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, flatten_and_deduplicate_eval_dataset
from src.eval_utils.metrics import Metrics
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor
from src.utils import batch_to_device, print_rank


def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    # Loading model
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_args.model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    # Config the dataset used for evaluation
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    # Compute embeddings
    for idx, (dataset_name, dataset_config) in enumerate(dataset_configs.items()):
        score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")

        if os.path.exists(score_path) or (os.path.exists(query_embed_path) and os.path.exists(cand_embed_path)):
            continue

        eval_qry_dataset = AutoPairDataset.instantiate(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            **dataset_config
        )
        eval_cand_dataset = flatten_and_deduplicate_eval_dataset(eval_qry_dataset)
        eval_cand_dataset = eval_cand_dataset.select_columns(["cand_text", "cand_image", "dataset_infos"])

        eval_qry_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "qry")
        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_qry_collator,
            shuffle=False,
            drop_last=False,
            num_workers=0 # training_args.dataloader_num_workers,
        )
        query_embeddings, dataset_infos = [], []
        with torch.no_grad():
            for qry_inputs, dataset_info in tqdm(eval_qry_loader, desc=f"Encoding - {dataset_name}"):
                qry_inputs = batch_to_device(qry_inputs, training_args.device)
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = model(qry=qry_inputs)
                    query_embeddings.append(output["qry_reps"].cpu().detach().float().numpy())
                dataset_infos.extend(dataset_info)
        query_embeddings = np.concatenate(query_embeddings)

        eval_cand_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "cand")
        eval_cand_loader = DataLoader(
            eval_cand_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_cand_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        cand_embeddings = []
        cand_embed_dict = {}
        all_labels = []
        with torch.no_grad():
            for cand_inputs, dataset_info in tqdm(eval_cand_loader, desc=f"Encoding - {dataset_name}"):
                cand_inputs = batch_to_device(cand_inputs, training_args.device)
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = model(qry=cand_inputs)
                    cand_embeddings.append(output["qry_reps"].cpu().detach().float().numpy())
                    for info in dataset_info:
                        all_labels.append(info["cand_name"])
        cand_embeddings = np.concatenate(cand_embeddings)
        for embed, video_id in zip(cand_embeddings, all_labels):
            cand_embed_dict[video_id] = embed

        with open(query_embed_path, 'wb') as f:
            pickle.dump(query_embeddings, f)
        with open(cand_embed_path, 'wb') as f:
            pickle.dump(cand_embed_dict, f)
        with open(dataset_info_path, 'w') as f:
            for dataset_info in dataset_infos:
                f.write(json.dumps(dataset_info) + '\n')


    # Compute scores
    for idx, (dataset_name, dataset_config) in enumerate(dataset_configs.items()):
        score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score or embeddings, skipping {dataset_name}")
                formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
                print(formatted)
                continue
            except Exception as e:
                print(f"Failed to load score for {dataset_name}, skipping {dataset_name}")

        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")
        with open(query_embed_path, 'rb') as f:
            qry_embed = pickle.load(f)
        with open(cand_embed_path, 'rb') as f:
            cand_embed_dict = pickle.load(f)
        dataset_infos = []
        with open(dataset_info_path, 'r') as f:
            for l in f:
                dataset_infos.append(json.loads(l.strip()))

        # ssv2-mc has different candidates per query
        preds = []
        for qry, info in zip(qry_embed, dataset_infos):
            cand_embed = np.stack([cand_embed_dict[key] for key in info["cand_names"]])
            cosine_sim = np.dot(qry, cand_embed.T)
            sorted_indices = np.argsort(-cosine_sim)
            preds.append({
                "prediction": [info["cand_names"][i] for i in sorted_indices],
                "label": info["label_name"],
            })

        score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
        pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")
        metrics = Metrics(dataset_config["metrics"])
        score_dict = metrics.evaluate(preds)
        formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
        score_dict["num_pred"] = len(preds)
        score_dict["num_data"] = len(dataset_infos)
        print(f"Score of {dataset_name}:")
        print(formatted)
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            json.dump(score_dict, f, indent=4)
        with open(pred_path, "w") as f:
            for pred in preds:
                f.write(json.dumps(pred) + '\n')


if __name__ == "__main__":
    main()

