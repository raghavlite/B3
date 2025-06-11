import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
import time

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoConfig

from src.model.model import MMEBModel
from src.data.dataset.mmeb_dataset import EvalDataset
from src.data.collator.eval_collator import EvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.mmeb_baselines.eval_utils import get_pred
from src.utils import print_rank
from src.model.processor import get_backbone_name, load_processor, COLPALI
from torch.nn.utils.rnn import pad_sequence


def delete_pycache(root='.'):
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            if dirname == '__pycache__':
                full_path = os.path.join(dirpath, dirname)
                print(f"Deleting: {full_path}")
                try:
                    shutil.rmtree(full_path)
                except:
                    print(">>>>>", "Module not exists", full_path, flush=True)
                    pass
delete_pycache()


POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
                "ImageNet-1K": POS_MOD_CLASS_LABEL,"HatefulMemes":POS_MOD_CLASS_LABEL,"SUN397":POS_MOD_CLASS_LABEL,"N24News":POS_MOD_CLASS_LABEL,"VOC2007":POS_MOD_CLASS_LABEL, "Place365":POS_MOD_CLASS_LABEL,"ImageNet-A":POS_MOD_CLASS_LABEL,"ImageNet-R":POS_MOD_CLASS_LABEL,"ObjectNet":POS_MOD_CLASS_LABEL,"Country211":POS_MOD_CLASS_LABEL,
                
                "OK-VQA":POS_MOD_ANSWER, "A-OKVQA":POS_MOD_ANSWER, "DocVQA":POS_MOD_ANSWER, "InfographicsVQA":POS_MOD_ANSWER, "ChartQA":POS_MOD_ANSWER, "Visual7W":POS_MOD_ANSWER,"ScienceQA":POS_MOD_ANSWER, "GQA":POS_MOD_ANSWER, "TextVQA":POS_MOD_ANSWER, "VizWiz":POS_MOD_ANSWER,
                
                "MSCOCO_i2t":POS_MOD_IMAGE_CAPTION, "VisualNews_i2t":POS_MOD_IMAGE_CAPTION,
                }


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
    print_rank(f'model_backbone: {model_args.model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    
    subset=data_args.subset_name[0]

    eval_qry_dataset = EvalDataset(
        data_args=data_args,
        model_args=model_args,
        subset=subset,
        text_field="qry_text",
        img_path_field="qry_img_path",
    )
    eval_tgt_dataset = EvalDataset(
        data_args=data_args,
        model_args=model_args,
        subset=subset,
        text_field="tgt_text",
        img_path_field="tgt_img_path",
        mod_instruction=POS_MOD_DICT.get(subset, None) if data_args.tgt_prefix_mod else None
    )

    eval_qry_loader = DataLoader(
        eval_qry_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=eval_collator,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    eval_tgt_loader = DataLoader(
        eval_tgt_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=eval_collator,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    encoded_tensor = []
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        for batch in tqdm(eval_qry_loader, desc=f"Encode query - {subset}"):
            batch = batch_to_device(batch, training_args.device)
            # import ipdb; ipdb.set_trace()
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                output = model(qry=batch)
            encoded_tensor.append(output["qry_reps"].cpu().detach().float())
    encoded_tensor = np.concatenate(encoded_tensor)
    qry_tensor, qry_index = encoded_tensor, eval_qry_dataset.paired_data
        

    encoded_tensor = []

    with torch.no_grad():
        for batch in tqdm(eval_tgt_loader, desc=f"Encode target - {subset}"):
            batch = batch_to_device(batch, training_args.device)
            output = model(tgt=batch)
            encoded_tensor.append(output["tgt_reps"].cpu().detach().float())
    encoded_tensor = np.concatenate(encoded_tensor)
    tgt_tensor, tgt_index = encoded_tensor, eval_tgt_dataset.paired_data        

    print(f"Loading eval dataset")
    eval_data = load_dataset(
        data_args.dataset_name,
        subset,
        split=data_args.dataset_split,
    )
    # build a map for dedup
    qry_key2emb, tgt_key2emb = OrderedDict(), OrderedDict()
    for qry_t, tt in zip(qry_tensor, qry_index):
        text, img_path = tt["text"], tt["img_path"]
        qry_key2emb[(text, img_path)] = qry_t
    for tgt_t, tt in zip(tgt_tensor, tgt_index):
        text, img_path = tt["text"], tt["img_path"]
        tgt_key2emb[(text, img_path)] = tgt_t

    print(f'len(tgt_key2emb) = {len(tgt_key2emb)}')

    n_correct = 0
    all_pred = []
    for row in tqdm(eval_data, desc=f"calculate score for {subset}"):
        qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]  # (dim,)
        tgt_t, all_candidates = [], []
        for tt in zip(row["tgt_text"], row["tgt_img_path"]):
            tgt_t.append(tgt_key2emb[tt])
            all_candidates.append(tt)
        qry_t = torch.from_numpy(np.array(qry_t))
        tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
        scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
        if pred == 0:
            n_correct += 1
        all_pred.append(all_candidates[pred])

    print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)}\033[0m")
    score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data),
                    "num_pred": len(all_pred), "num_data": len(eval_data)}
    print(score_dict)


if __name__ == "__main__":
    main()
