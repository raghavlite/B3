import yaml
import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration

from src.data.eval_dataset.base_eval_dataset import AutoPairDataset
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from torch.utils.data import DataLoader
from datasets import Dataset
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os

from src.utils import print_rank
from src.model.processor import get_backbone_name, load_processor
from src.eval_utils.metrics import Metrics


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch


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

    # Load E5V only, can be generalized to other models
    processor = processor = LlavaNextProcessor.from_pretrained(
            model_args.model_name, # should be 'royokong/e5-v'
            trust_remote_code=True
        )
    model = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name, torch_dtype=torch.float16).cuda()

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

        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print_rank(f"Found previous eval score, skipping {dataset_name}")
                print_rank(score_dict)
            except Exception as e:
                pass

        if not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path):
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
                for qry_inputs, dataset_info in tqdm(eval_qry_loader, desc=f"Encoding - {dataset_name}"):
                    qry_inputs = batch_to_device(qry_inputs, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(**qry_inputs)
                        query_embeddings.append(output["qry_reps"].cpu().detach().float().numpy())
                    dataset_infos.extend(dataset_info)
            query_embeddings = np.concatenate(query_embeddings)
            with open(query_embed_path, 'wb') as f:
                pickle.dump(query_embeddings, f)
            with open(dataset_info_path, 'w') as f:
                for dataset_info in dataset_infos:
                    f.write(json.dumps(dataset_info) + '\n')

        if not os.path.exists(cand_embed_path):
            all_class_labels = dataset_infos[0]["candidates"]

            eval_cand_dataset = Dataset.from_list(
                [{"cand_text": [label],
                  "cand_image": [None],
                  "dataset_infos": {"label": label}} for label in all_class_labels]
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
            cand_embed_dict = {}
            all_labels = []
            with torch.no_grad():
                for cand_inputs, dataset_info in tqdm(eval_cand_loader, desc=f"Encoding - {dataset_name}"):
                    cand_inputs = batch_to_device(cand_inputs, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(**cand_inputs)
                        cand_embeddings.append(output["qry_reps"].cpu().detach().float().numpy())
                        for info in dataset_info:
                            all_labels.append(info["label"])
            cand_embeddings = np.concatenate(cand_embeddings)
            for embed, label in zip(cand_embeddings, all_labels):
                cand_embed_dict[label] = embed
            with open(cand_embed_path, 'wb') as f:
                pickle.dump(cand_embed_dict, f)


    # Compute scores
    for idx, (dataset_name, dataset_config) in enumerate(dataset_configs.items()):
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

        cand_keys = list(cand_embed_dict.keys())
        cand_embed = np.stack([cand_embed_dict[key] for key in cand_keys])
        cosine_sim = np.dot(qry_embed, cand_embed.T)
        sorted_indices = np.argsort(-cosine_sim, axis=1)
        preds = []
        for indices, info in zip(sorted_indices, dataset_infos):
            preds.append({
                "prediction": [cand_keys[i] for i in indices],
                "label": info["class_label"],
            })

        metrics = Metrics(dataset_config["metrics"])
        score_dict = metrics.evaluate(preds)
        score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
        pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")
        print(f'Successfully computed scores for {dataset_name}: {score_dict}!')
        with open(score_path, "w") as f:
            json.dump(score_dict, f, indent=4)
        # TODO, save dataset_info as well
        with open(pred_path, "w") as f:
            for pred in preds:
                f.write(json.dumps(pred) + '\n')
        print(f"\033[91m{dataset_name} accuracy: {score_dict['precision@1']:.6f}\033[0m")


if __name__ == "__main__":
    main()
