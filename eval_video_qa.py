import json
import sys
import yaml

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoConfig

from src.data.dataset.base_pair_dataset import AutoPairDataset
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from evaluation.mmeb_baselines.eval_utils import get_pred
from src.model.model import MMEBModel
from src.utils import print_rank
from src.model.processor import get_backbone_name, load_processor


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

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print(f'model_backbone: {model_args.model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    with open(data_args.dataset_config, 'r') as yaml_file:
        eval_config = yaml.safe_load(yaml_file)

    # Step 1: compute embeddings
    for idx, (task_name, task_config) in enumerate(eval_config.items()):
        score_path = os.path.join(data_args.encode_output_path, f"{task_name}_score.json")
        query_embed_path = os.path.join(data_args.encode_output_path, f"{task_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{task_name}_tgt")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{task_name}_info.jsonl")

        # check accuracy of random choice
        '''
        eval_dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, **task_config)
        avg_accuracy = []
        for row in tqdm(eval_dataset, desc=f"{task_name}"):
            # print(row['cand_text'])
            avg_accuracy.append(1.0 / len(row['cand_text']))
            pass
        print(task_name, np.mean(avg_accuracy))
        continue
        '''

        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {task_name}")
                print(score_dict)
                continue
            except Exception as e:
                pass

        # encode queries
        if not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path):
            eval_dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, **task_config)
            eval_qry_collator = MultimodalEvalDataCollator(processor, model_args, data_args, encode_side="qry")
            eval_qry_loader = DataLoader(
                eval_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=eval_qry_collator,
                shuffle=False,
                drop_last=False,
                num_workers=training_args.dataloader_num_workers,
            )
            query_embeddings, dataset_infos = [], []
            with torch.no_grad():
                for qry_inputs, dataset_info in tqdm(eval_qry_loader, desc=f"Encoding queries - {task_name}"):
                    qry_inputs = batch_to_device(qry_inputs, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(qry=qry_inputs)
                        query_embeddings.append(output["qry_reps"].cpu().detach().float().numpy())
                    dataset_infos.extend(dataset_info)
            query_embeddings = np.concatenate(query_embeddings)
            with open(query_embed_path, 'wb') as f:
                pickle.dump(query_embeddings, f)
            with open(dataset_info_path, 'w') as f:
                print(f"Writing {len(dataset_infos)} dataset info to {dataset_info_path}")
                for dataset_info in dataset_infos:
                    f.write(json.dumps(dataset_info) + '\n')

        # encode candidates
        if not os.path.exists(cand_embed_path):
            eval_dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, **task_config)
            eval_cand_collator = MultimodalEvalDataCollator(processor, model_args, data_args, encode_side="cand")
            eval_cand_loader = DataLoader(
                eval_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=eval_cand_collator,
                shuffle=False,
                drop_last=False,
                num_workers=training_args.dataloader_num_workers,
            )
            cand_embeddings = []
            with torch.no_grad():
                for cand_inputs, _ in tqdm(eval_cand_loader, desc=f"Encoding candidates - {task_name}"):
                    cand_inputs = batch_to_device(cand_inputs, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(tgt=cand_inputs)
                        cand_embeddings.append(output["tgt_reps"].cpu().detach().float().numpy())
            cand_embeddings = np.concatenate(cand_embeddings)
            with open(cand_embed_path, 'wb') as f:
                pickle.dump(cand_embeddings, f)

    # Step 2: compute scores
    for idx, (task_name, task_config) in enumerate(eval_config.items()):
        query_embed_path = os.path.join(data_args.encode_output_path, f"{task_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{task_name}_tgt")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{task_name}_info.jsonl")
        with open(query_embed_path, 'rb') as f:
            qry_embed = pickle.load(f)
        with open(cand_embed_path, 'rb') as f:
            cand_embed = pickle.load(f)
        dataset_infos = []
        with open(dataset_info_path, 'r') as f:
            for l in f:
                dataset_infos.append(json.loads(l.strip()))

        n_correct = 0
        cand_offset_idx = 0
        for i, dataset_info in enumerate(dataset_infos):
            qry_t = qry_embed[i]
            num_cand = len(dataset_info['candidates'])
            answer_idx = dataset_info['answer_idx']
            cand_t = cand_embed[cand_offset_idx: cand_offset_idx + num_cand]
            scores, pred = get_pred(qry_t, cand_t, normalization=model_args.normalize)
            if pred == answer_idx:
                n_correct += 1
            cand_offset_idx += num_cand
            dataset_info['pred_idx'] = int(pred)
            dataset_info['is_correct'] = int(pred == answer_idx)
        score_path = os.path.join(data_args.encode_output_path, f"{task_name}_score.json")
        pred_path = os.path.join(data_args.encode_output_path, f"{task_name}_pred.jsonl")
        print(f"Outputting final score to: {score_path}")
        print(f"\033[91m{task_name} accuracy: {n_correct / len(dataset_infos):.6f}\033[0m")
        score_dict = {"acc": n_correct / len(dataset_infos), "num_correct": n_correct, "num_pred": len(dataset_infos), "num_data": len(dataset_infos)}
        with open(score_path, "w") as f:
            json.dump(score_dict, f, indent=4)
        with open(pred_path, "w") as f:
            for dataset_info in dataset_infos:
                f.write(json.dumps(dataset_info) + '\n')


if __name__ == "__main__":
    main()
