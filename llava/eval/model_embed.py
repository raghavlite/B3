import json
import sys

import transformers
from transformers import HfArgumentParser, AutoProcessor, ProcessorMixin, AutoTokenizer
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
import re
from datasets import load_dataset
import datasets
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from PIL import Image
import argparse
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images, process_highres_image, process_anyres_image, process_highres_image_crop_split
from llava import conversation as conversation_lib
from llava.train.train import preprocess

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX

def get_pred(qry_t, tgt_t, normalization=True):
    """
    Use L2 norms.
    """
    if normalization:
        qry_t_norm = np.linalg.norm(qry_t)
        tgt_t_norms = np.linalg.norm(tgt_t, axis=1)
        scores = np.dot(tgt_t, qry_t) / (tgt_t_norms * qry_t_norm)
    else:
        scores = np.dot(tgt_t, qry_t)
    pred = np.argmax(scores)
    return scores, pred

def preprocess_multimodal(sources: Sequence[str], data_args) -> Dict:

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources
class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field, tokenizer):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer
        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        image = [self.process_image(img_path)]
        has_image = True if image[0] is not None else False
        # Adapting llava's code
        if has_image:
            text = preprocess_multimodal([[{"from": "human", "value": text.replace("<|image_1|>","<image>")},{"from": "gpt", "value": "\n"}]], self.data_args)
        else:
            text = [[{"from": "human", "value": text},{"from": "gpt", "value": "\n"}]]
        
        data_dict = preprocess(text, self.tokenizer, has_image)

        # print("dict", dict)
        if isinstance(item, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0])

        # print("has_qry_image", has_qry_image, "has_pos_image", has_pos_image)
        if has_image:
            data_dict["image"] = image

        return data_dict

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        if image_file is None or image_file == "":
            crop_size = self.data_args.image_processor.crop_size
            return torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"
        
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.model_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.model_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.model_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"
    
    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif isinstance(row[text_field], List):
                assert isinstance(row[img_path_field], List) and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data

@dataclass
class EvalCollator:

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def _get_batch_inputs(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = tuple([instance["input_ids"] for instance in instances])
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch = dict(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch
    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs


    
def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--encode_output_path", type=str, default="encode_output")
    parser.add_argument("--dataset_name", type=str, default="llava_v1")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--subset_name", type=str, nargs="+", default=["CIRR"])
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--version", type=str, default="qwen_1_5")
    parser.add_argument("--mm_use_im_start_end", action="store_true")
    parser.add_argument("--use_lora", type=bool, default=False)

    args = parser.parse_args()

 
    os.makedirs(args.encode_output_path, exist_ok=True)

    model_path = os.path.expanduser(args.model_name_or_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    args.image_processor = image_processor
    model_args = model.config
    model.eval()
    # tokenizer.padding_side = "le
    if args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    eval_collator = EvalCollator(tokenizer=tokenizer)

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(args.subset_name):
        score_path = os.path.join(args.encode_output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
            except Exception as e:
                pass

        print(f"{idx+1}/{len(args.subset_name)}: Processing {subset} now!")
        encode_qry_path = os.path.join(args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(args.encode_output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            continue

        eval_qry_dataset = EvalDataset(
            data_args=args,
            model_args=model_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
            tokenizer=tokenizer,
        )
        eval_tgt_dataset = EvalDataset(
            data_args=args,
            model_args=model_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
            tokenizer=tokenizer,
        )

        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=args.dataloader_num_workers,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=args.dataloader_num_workers,
        )

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_qry_loader, desc="Encode query"):
                # batch = {key: value.to(args.device) for key, value in batch.items()}
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(model.device)
                        elif isinstance(batch[key], list):
                            if isinstance(batch[key][0], torch.Tensor):
                                batch[key] = [x.to(model.device) for x in batch[key]]
                    output = model.encode_multimodal_embeddings(**batch)
                encoded_tensor.append(output.cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_qry_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                # batch = {key: value.to(args.device) for key, value in batch.items()}
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(model.device)
                        elif isinstance(batch[key], list):
                            if isinstance(batch[key][0], torch.Tensor):
                                batch[key] = [x.to(model.device) for x in batch[key]]
                    output = model.encode_multimodal_embeddings(**batch)
                encoded_tensor.append(output.cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_tgt_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    for subset in tqdm(args.subset_name, desc="calculate score"):
        encode_qry_path = os.path.join(args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(args.encode_output_path, f"{subset}_tgt")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)
        qry_dict, tgt_dict = {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            tgt_dict[(text, img_path)] = tgt_t

        eval_data = load_dataset(
            args.dataset_name,
            subset,
            split=args.dataset_split,
        )
        n_correct = 0
        all_pred = []
        for row in eval_data:
            qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
            tgt_t, all_candidates = [], []
            for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            scores, pred = get_pred(qry_t, tgt_t, normalization=args.normalize)
            if pred == 0:
                n_correct += 1
            all_pred.append(all_candidates[pred])
        with open(os.path.join(args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")
        score_path = os.path.join(args.encode_output_path, f"{subset}_score.json")
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data)}
            json.dump(score_dict, f, indent=4)
        print(f"{subset} accuracy: {n_correct/len(eval_data)}")


if __name__ == "__main__":
    main()
