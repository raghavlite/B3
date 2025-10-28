import os
import sys

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.data.utils.vision_utils import save_frames, load_frames, sample_frames
from src.model.processor import process_input_text


TASK_INST_QRY = "Represent the given image with the following question:"
TASK_INST_TGT = ""
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_img_path, tgt_texts in (
            zip(batch_dict['qry_img_path'], batch_dict['tgt_text'])):

        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text="")])
        qry_img_path = os.path.join(image_root, qry_img_path)
        query_images.append([{"bytes": [None], "paths": [qry_img_path],
                            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}])

        cand_texts.append(tgt_texts)
        cand_images.append([None] * len(tgt_texts))
        dataset_infos.append({
            "cand_names": tgt_texts,
            "label_name": tgt_texts[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "mmeb_qa"
DATASET_HF_PATH = "ziyjiang/MMEB_Test_Instruct"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_mmeb_qa_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]

    dataset = load_dataset(DATASET_HF_PATH, dataset_name, split="test")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch = False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset
