import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoConfig

from src.data.eval_dataset.base_eval_dataset import AutoPairDataset
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from src.model.model import MMEBModel
from src.utils import print_rank
from src.model.processor import get_backbone_name, load_processor
from src.eval_utils.metrics import Metrics
from src.eval_utils.index import HNSWIndex


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def task_to_cand_type(task_name: str) -> str:
    try:
        target = task_name.split('->')[1]
        if target.startswith('S'):
            return 'state'
        elif target.startswith('W') and len(target) > 2 and target[2].isdigit():
            return 'interval'
        else:
            return 'trajectory'
    except Exception as e:
        raise ValueError(f"Error parsing task name '{task_name}': {e}")

def generate_configs(
        datasets, 
        limit="limit_10", 
        base_path="/home/ubuntu/GUI-MMEB",
        metrics=["precision", "recall", "ndcg"], 
        k_values=[1, 5, 10]
    ):
    dataset_configs = defaultdict(dict)
    candidate_configs = defaultdict(dict)
    
    for dataset in datasets:
        # Generate dataset configs for both in-domain and out-of-domain
        for split in ["ood", "ind"]:
            config_key = f"{dataset}_{limit}_{split}"
            dataset_configs[config_key] = {
                "dataset_parser": "gui",
                "dataset_name": base_path,
                "subset_name": "test",
                "image_dir": f"{base_path}/images",
                "dataset_split": config_key,
                "num_sample_per_subset": None,
                "metrics": metrics,
                "k_values": k_values,
                "candidate_name": f"{dataset}_{limit}",
            }
        
        # Generate candidate configs
        candidate_key = f"{dataset}_{limit}"
        
        for cand_type, suffix in [
            ("state", "states"), 
            ("trajectory", "trajs"), 
            ("interval", "invls")
        ]:
            candidate_configs[candidate_key][cand_type] = {
                "dataset_parser": "gui",
                "dataset_name": base_path,
                "subset_name": "pool",
                "image_dir": f"{base_path}/images",
                "dataset_split": f"{dataset}_{limit}_{suffix}",
                "num_sample_per_subset": None,
            }
    
    return dataset_configs, candidate_configs

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
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    datasets = ["autowebglm", "mind2web", "weblinx", "guiact", "webarena"]
    dataset_configs, candidate_configs = generate_configs(datasets)

    # Compute embeddings
    for idx, (dataset_name, dataset_config) in enumerate(dataset_configs.items()):
        score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")

        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print_rank(f"Found previous eval score or embeddings, skipping {dataset_name}")
                print_rank(score_dict)
            except Exception as e:
                pass
            continue

        if os.path.exists(query_embed_path) and os.path.exists(dataset_info_path):
            with open(query_embed_path, 'rb') as f:
                qry_embed = pickle.load(f)
            with open(dataset_info_path, 'r') as f:
                dataset_infos = [json.loads(l.strip()) for l in f]
            if len(qry_embed) > 0 and len(qry_embed) == len(dataset_infos):
                print_rank(f"Found previous query embeddings, skipping {dataset_name}")
                continue

        eval_qry_dataset = AutoPairDataset.instantiate(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            **dataset_config
        )
        eval_qry_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "qry")
        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_qry_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        query_embeddings, dataset_infos = [], []
        with torch.no_grad():
            for qry_inputs, dataset_info in tqdm(eval_qry_loader, desc=f"Encoding query - {dataset_name}"):
                qry_inputs = batch_to_device(qry_inputs, training_args.device)
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = model(qry=qry_inputs)
                    query_embeddings.append(output["qry_reps"].cpu().detach().float().numpy())
                dataset_infos.extend(dataset_info)
        query_embeddings = np.concatenate(query_embeddings)

        with open(query_embed_path, 'wb') as f:
            pickle.dump(query_embeddings, f)
        with open(dataset_info_path, 'w') as f:
            for dataset_info in dataset_infos:
                f.write(json.dumps(dataset_info) + '\n')

    for candidate_name, candidate_config in candidate_configs.items():
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{candidate_name}_tgt")

        for candidate_type, local_config in candidate_config.items():
            if os.path.exists(cand_embed_path):
                with open(cand_embed_path, 'rb') as f:
                    cand_embed_dict = pickle.load(f)
                assert isinstance(cand_embed_dict, dict)
                if candidate_type in cand_embed_dict and len(cand_embed_dict[candidate_type]) > 0:
                    print_rank(f"Found previous candidate embeddings, skipping {candidate_name} ({candidate_type})")
                    continue
            else:
                cand_embed_dict = defaultdict(dict)
            
            eval_cand_dataset= AutoPairDataset.instantiate(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                **local_config
            )
            
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
            all_candidate_id = []
            with torch.no_grad():
                for cand_inputs, dataset_info in tqdm(eval_cand_loader, desc=f"Encoding candidates - {candidate_name} ({candidate_type})"):
                    cand_inputs = batch_to_device(cand_inputs, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(tgt=cand_inputs)
                        cand_embeddings.append(output["tgt_reps"].cpu().detach().float().numpy())
                    for info in dataset_info:
                        all_candidate_id.append(info["cand_id"])
            cand_embeddings = np.concatenate(cand_embeddings)
            for embed, cand_id in zip(cand_embeddings, all_candidate_id):
                cand_embed_dict[candidate_type][cand_id] = embed
            
            with open(cand_embed_path, 'wb') as f:
                pickle.dump(cand_embed_dict, f)

    # Compute scores
    for idx, (dataset_name, dataset_config) in enumerate(dataset_configs.items()):
        candidate_name = dataset_config.get("candidate_name")
        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{candidate_name}_tgt")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")
        with open(query_embed_path, 'rb') as f:
            qry_embed = pickle.load(f)
        with open(cand_embed_path, 'rb') as f:
            cand_embed_dict = pickle.load(f)
        dataset_infos = []
        with open(dataset_info_path, 'r') as f:
            for l in f:
                dataset_infos.append(json.loads(l.strip()))

        top_k = max(dataset_config["k_values"])
        cand_index = HNSWIndex()

        for cand_type in cand_embed_dict:
            cand_keys = list(cand_embed_dict[cand_type].keys())
            cand_vectors = np.array([cand_embed_dict[cand_type][k] for k in cand_keys], dtype=np.float32)
            cand_index.create_index(cand_type, cand_vectors, cand_keys)

        # Group queries by retrieval task
        retrieval_order = defaultdict(list)
        for data_idx, info in enumerate(dataset_infos):
            retrieval_order[info["retrieval_type"]].append(data_idx)

        overall_pred = [None] * len(dataset_infos)
        for retrieval_task, qry_idx in retrieval_order.items():
            cand_type = task_to_cand_type(retrieval_task)
            # Get queries using indices rather than storing them separately
            batch_queries = np.stack([qry_embed[idx] for idx in qry_idx])
            print_rank(f"Searching {len(batch_queries)} queries for task {retrieval_task} ({cand_type})")

            similarities, predictions = cand_index.search(cand_type, batch_queries, top_k)

            for calc_idx, batch_idx in enumerate(qry_idx):
                overall_pred[batch_idx] = {
                    "prediction": predictions[calc_idx],
                    "label": dataset_infos[batch_idx]["cand_id"],
                    "similarity": similarities[calc_idx],
                    "retrieval_task": retrieval_task,
                }
        
        metrics = Metrics(dataset_config["metrics"])
        overall_score_dict = metrics.evaluate(overall_pred)

        task_preds = defaultdict(list)
        for pred in overall_pred:
            task_preds[pred["retrieval_task"]].append(pred)
        
        task_score_dict = {}
        for task, preds in task_preds.items():
            task_score_dict[task] = metrics.evaluate(preds)

        saved_score_dict = {
            "overall": overall_score_dict,
            "task": task_score_dict,
        }

        score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
        pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")
        with open(pred_path, "w") as f:
            json.dump(overall_pred, f, indent=4)
        with open(score_path, "w") as f:
            json.dump(saved_score_dict, f, indent=4)


if __name__ == "__main__":
    main()
