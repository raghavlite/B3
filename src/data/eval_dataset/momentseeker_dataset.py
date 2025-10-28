import os
import sys

from datasets import load_dataset
from src.data.dataset.vidore_dataset import DATASET_PARSER_NAME
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.data.utils.vision_utils import sample_frames, load_frames, VID_EXTENSIONS, save_frames
from src.model.processor import process_input_text


TASK_INST_QRY_TEXT = "Find the clip that corresponds to the given text:"
TASK_INST_QRY_IMG = "Select the video clip that aligns with the given text and image:"
TASK_INST_QRY_VIDEO = "Find the clip that corresponds to the given sentence and video segment:"
TASK_INST_TGT = "Understand the content of the provided video clip."
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution = kwargs['image_resolution']

    ## metadata
    num_negative_clips = kwargs["num_negative_clips"]
    num_video_frames = kwargs["num_video_frames"]
    model_backbone = kwargs["model_backbone"]
    frame_root = kwargs["frame_root"]
    ms_root = kwargs["ms_root"]

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for query, positive_frames, negative_frames, input_frames in \
            zip(batch_dict['query'], batch_dict["positive_frames"], batch_dict["negative_frames"],
                batch_dict["input_frames"]):

        if (input_frames.endswith(".mp4")):
            query_texts.append([process_input_text(TASK_INST_QRY_VIDEO, model_backbone, text=query, add_video_token=True)])
            query_video_name = input_frames.split(".mp4")[0].replace("/", "_")
            if query_video_name == 'movie101_77':
                pass
            query_frame_dir = os.path.join(frame_root, query_video_name)
            query_video_path = os.path.join(ms_root, input_frames)
            save_frames(video_path=query_video_path,
                        frame_dir=query_frame_dir,
                        max_frames_saved=num_video_frames)
            qry_frame_paths = load_frames(query_frame_dir)
            query_images.append([{"bytes": [None] * len(qry_frame_paths), "paths": qry_frame_paths,
                                  "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths)}])
        elif (input_frames.endswith(".jpg")):
            query_texts.append([process_input_text(TASK_INST_QRY_IMG, model_backbone, text=query, add_image_token=True)])
            input_image_path = os.path.join(ms_root, f"query_{input_frames}")
            query_images.append([{"bytes": [None] * 1, "paths": [input_image_path],
                                  "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * 1}])
        else:
            query_texts.append([process_input_text(TASK_INST_QRY_TEXT, model_backbone, text=query)])
            query_images.append([None])

        pos_clip_paths = [entry["output_path"] for entry in positive_frames]
        neg_clip_paths = [entry["output_path"] for entry in negative_frames]

        label_name, cand_names, cand_frames = [], [], []
        for path in pos_clip_paths:
            cand_clip_name = path.replace("/", "_").split(".mp4")[0]
            cand_clip_abs_path = os.path.join(ms_root, path)
            cand_clip_frame_dir = os.path.join(frame_root, cand_clip_name)
            os.makedirs(cand_clip_frame_dir, exist_ok=True)
            save_frames(video_path=cand_clip_abs_path, frame_dir=cand_clip_frame_dir, max_frames_saved=num_video_frames)
            pos_clip_frames = load_frames(cand_clip_frame_dir)
            cand_frames.append({"bytes": [None] * num_video_frames, "paths": pos_clip_frames,
                                "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(pos_clip_frames)})
            cand_names.append(cand_clip_frame_dir)
            label_name.append(cand_clip_frame_dir)
        for path in neg_clip_paths:
            cand_clip_name = path.replace("/", "_").split(".mp4")[0]
            cand_clip_abs_path = os.path.join(ms_root, path)
            cand_clip_frame_dir = os.path.join(frame_root, cand_clip_name)
            os.makedirs(cand_clip_frame_dir, exist_ok=True)
            save_frames(video_path=cand_clip_abs_path, frame_dir=cand_clip_frame_dir, max_frames_saved=num_video_frames)
            neg_clip_frames = load_frames(cand_clip_frame_dir)
            cand_frames.append({"bytes": [None] * num_video_frames, "paths": neg_clip_frames,
                                "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(neg_clip_frames)})
            cand_names.append(cand_clip_frame_dir)

        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)] * len(
            pos_clip_paths + neg_clip_paths))
        cand_images.append(cand_frames)
        dataset_infos.append({
            "cand_names": cand_names,
            "label_name": label_name,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "momentseeker"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_momentseeker_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]
    dataset = load_dataset("json", data_files=kwargs["data_path"])
    # dataset = load_dataset("json", data_files="/data/yuepeng/moment_retrieval/dataset/debug_qv.jsonl")
    dataset = dataset["train"]
    def fileter_data(example):
        return example['input_frames'] == ""
    dataset = dataset.filter(fileter_data)
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'
    
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch=False,
                          load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
    return dataset
