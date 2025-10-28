import os
import sys

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.data.utils.vision_utils import save_frames, load_frames, sample_frames
from src.model.processor import process_input_text


TASK_INST_QRY = "Crop the image to to isolate the object labeled as "
TASK_INST_TGT = "Represent the given cropped image of the object"
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_text, tgt_img_path in (
            zip(batch_dict["qry_text"], batch_dict['tgt_img_path'])):

        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=qry_text)])
        query_images.append([None])

        cand_texts.append([None] * len(tgt_img_path))
        cand_images.append([{"bytes": [None], "paths": [os.path.join(image_root, img_path)],
                             "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for img_path in tgt_img_path])
        dataset_infos.append({
            "cand_names": tgt_img_path,
            "label_name": tgt_img_path[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "mmeb_t2i"
DATASET_HF_PATH = "ziyjiang/MMEB_Test_Instruct"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_mmeb_t2i_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]
    eval_num_sample_per_subset = kwargs["eval_num_sample_per_subset"]

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
