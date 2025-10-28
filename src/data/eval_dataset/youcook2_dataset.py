import os
import sys

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.data.utils.vision_utils import save_frames, load_frames, sample_frames, process_video_frames
from src.model.processor import process_input_text


TASK_INST_QRY = "Find a video that demonstrates the following action while making a recipe:"
TASK_INST_TGT = "Understand the content of the provided video."
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    model_backbone = kwargs["model_backbone"]

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_name, video_path, text in zip(batch_dict['id'], batch_dict["video_path"], batch_dict['sentence']):
        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=text)])
        query_images.append([None])

        video_path = os.path.join(video_root, os.path.basename(video_path))
        frame_dir = os.path.join(frame_root, video_name)
        save_frames(video_path=video_path,
                    frame_dir=frame_dir,
                    max_frames_saved=max_frames_saved)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)


        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)])
        cand_images.append([{"bytes": [None] * len(video_frame_paths), "paths": video_frame_paths,
                             "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths)}])
        dataset_infos.append({
            "cand_names": [video_name],
            "label_name": video_name,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "youcook2"
DATASET_HF_PATH = "lmms-lab/YouCook2"  # slightly less than the official one: https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/csv/validation_youcook.csv?plain=1
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_youcook2_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset = load_dataset(DATASET_HF_PATH, split="val")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if type(num_sample_per_subset) is int and num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64, drop_last_batch = False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset
