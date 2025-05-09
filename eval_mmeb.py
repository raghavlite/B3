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

@contextmanager
def time_block(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[Timer] {name}: {elapsed:.4f}s")


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
    for idx, subset in enumerate(data_args.subset_name):
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
                continue
            except Exception as e:
                pass

        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            continue

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
        if not os.path.exists(encode_qry_path):
            with torch.no_grad():
                for batch in tqdm(eval_qry_loader, desc=f"Encode query - {subset}"):
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(qry=batch)
                    encoded_tensor.append(output["qry_reps"].cpu().detach().float())
            encoded_tensor = np.concatenate(encoded_tensor)
            with open(encode_qry_path, 'wb') as f:
                pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)
            

        encoded_tensor = []
        if not os.path.exists(encode_tgt_path):
            with torch.no_grad():
                for batch in tqdm(eval_tgt_loader, desc=f"Encode target - {subset}"):
                    batch = batch_to_device(batch, training_args.device)
                    output = model(tgt=batch)
                    encoded_tensor.append(output["tgt_reps"].cpu().detach().float())
            encoded_tensor = np.concatenate(encoded_tensor)
            with open(encode_tgt_path, 'wb') as f:
                pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    for subset in tqdm(data_args.subset_name, desc="Iterate datasets to calculate scores"):
        print(f"\033[91m{subset}: Calculating score now!\033[0m")
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
                continue
            except Exception as e:
                pass

        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        print(f"Loading cached query/target tensors")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)
        

        print(f"Loading eval dataset")
        eval_data = load_dataset(
            data_args.dataset_name,
            subset,
            split=data_args.dataset_split,
        )

        # for ColPali, pad only once for candidates
        if model_args.model_backbone == COLPALI:
            #  in case of ColPali, query_tensor/tgt_tensor have not been flattened
            if len(qry_tensor) != len(qry_index):
                qry_tensor = [t for l in qry_tensor for t in l.tolist()]
                # qry_tensor = qry_tensor[:500]
                # qry_index = qry_index[:500]
            if len(tgt_tensor) != len(tgt_index):
                tgt_tensor = [t for l in tgt_tensor for t in l.tolist()]
                # tgt_tensor = tgt_tensor[:500]
                # tgt_index = tgt_index[:500]

            # Initialize mappings
            tgtkey_to_rowid = {}
            rowid_to_tgtkey = {}
            for rowid, tt in enumerate(tgt_index):
                key = (tt["text"], tt["img_path"])
                tgtkey_to_rowid[key] = rowid
                rowid_to_tgtkey[rowid] = key

            # Convert each [n_token, dim] numpy array to tensor
            tgt_tensor = [torch.from_numpy(np.array(t)) for t in tgt_tensor]  # list of [n_token, dim] tensors
            # Pad to max n_token (dim must be the same)
            tgt_tensor = pad_sequence(tgt_tensor, batch_first=True)  # shape: [num_candidates, max_n_token, dim]
            tgt_tensor = tgt_tensor.to("cuda")

        # build a map for dedup
        qry_key2emb, tgt_key2emb = OrderedDict(), OrderedDict()
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_key2emb[(text, img_path)] = qry_t
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            tgt_key2emb[(text, img_path)] = tgt_t
        # print(f'tgt_tensor.shape = {tgt_tensor.shape}')
        # print(f'len(tkey_to_rowid) = {len(tgtkey_to_rowid)}')
        # print(f'len(rowid_to_tkey) = {len(rowid_to_tgtkey)}')
        print(f'len(tgt_key2emb) = {len(tgt_key2emb)}')

        n_correct = 0
        all_pred = []
        for row in tqdm(eval_data, desc=f"calculate score for {subset}"):
            if model_args.model_backbone == COLPALI:
                qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]  # (dim,)
                qry_t = torch.from_numpy(np.array([qry_t]))  # qry_t.shape=[1,len_q,dim], tgt_tensor.shape=[n_cand,len_cand,dim]
                pos_tkey = (row['tgt_text'][0], row['tgt_img_path'][0])
                print(pos_tkey in tgtkey_to_rowid)
                print(pos_tkey in tgt_key2emb)
                # with time_block("Scoring & prediction"):
                scores = processor.score(qry_t, tgt_tensor, batch_size=1024)
                pred_id = torch.argmax(scores, dim=1).cpu().numpy().tolist()[0]
                pred_tkey = rowid_to_tgtkey[pred_id]
                if pred_tkey == pos_tkey:
                    n_correct += 1
                all_pred.append(pred_tkey)
            else:
                try:
                    qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]  # (dim,)
                    tgt_t, all_candidates = [], []
                except:
                    import ipdb; ipdb.set_trace()
                # with time_block("Target vector & candidate prep"):
                for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                    tgt_t.append(tgt_key2emb[tt])
                    all_candidates.append(tt)
                qry_t = torch.from_numpy(np.array(qry_t))
                tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
                scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
                if pred == 0:
                    n_correct += 1
                all_pred.append(all_candidates[pred])
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)}\033[0m")
        score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data),
                      "num_pred": len(all_pred), "num_data": len(eval_data)}
        print(score_dict)
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            json.dump(score_dict, f, indent=4)
        with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")


if __name__ == "__main__":
    main()
