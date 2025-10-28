import os
import sys

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.data.utils.vision_utils import sample_frames, load_frames, VID_EXTENSIONS, save_frames
from src.model.processor import process_input_text


TASK_INST_QRY = "Find the clip that corresponds to the described scene in the given video:"
TASK_INST_TGT = "Understand the content of the provided video."
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution = kwargs['image_resolution']

    ## metadata
    num_negative_clips = kwargs["num_negative_clips"]
    max_video_frames_saved = kwargs["max_video_frames_saved"]
    max_clip_frames_saved = kwargs["max_clip_frames_saved"]
    num_video_frames = kwargs["num_video_frames"]
    num_clip_frames = kwargs["num_clip_frames"]
    model_backbone = kwargs["model_backbone"]
    video_dir, clip_dir, frame_dir = kwargs["video_dir"], kwargs["clip_dir"], kwargs["frame_dir"]

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    
    for query, query_video_path, clips_dir_path in \
        zip(batch_dict['query'], batch_dict['video_path'], batch_dict['clips_dir_path']):    

        ## load frames
        video_name = os.path.splitext(os.path.basename(query_video_path))[0]
        frames_dir = os.path.join(frame_dir, video_name)

        # Load query video
        query_video_path = os.path.join(video_dir, os.path.basename(query_video_path)) if video_dir else None
        query_frame_dir = os.path.join(frames_dir, "query")
        save_frames(video_path=query_video_path,
                    frame_dir=query_frame_dir,
                    max_frames_saved=max_video_frames_saved)
        qry_frame_paths = load_frames(query_frame_dir)
        qry_frame_paths = sample_frames(qry_frame_paths, num_segments=num_video_frames)

        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=query, add_video_token=True)])
        query_images.append([{"bytes": [None] * len(qry_frame_paths), "paths": qry_frame_paths,
                             "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths)}])

        # Load pos and neg clip, save the frames if only the raw video is provided.
        cand_names, cand_frames = [], []
        clip_video_dir = os.path.join(clip_dir, video_name) if clip_dir else None
        clip_video_paths = [f for f in os.listdir(clip_video_dir) if os.path.splitext(f)[1].lower() in VID_EXTENSIONS]
        for clip_video_path in clip_video_paths:
            clip_name = os.path.splitext(clip_video_path)[0]
            clip_frame_dir_or_file = os.path.join(frames_dir, clip_name)
            clip_video_path_abs = os.path.join(clip_video_dir, clip_video_path)
            save_frames(video_path=clip_video_path_abs,
                        frame_dir=clip_frame_dir_or_file,
                        max_frames_saved=max_clip_frames_saved)
        for clip_frame_dir_or_file in os.listdir(frames_dir):
            clip_frame_dir_abs = os.path.join(frames_dir, clip_frame_dir_or_file)
            if clip_frame_dir_or_file == 'query' or os.path.isfile(clip_frame_dir_abs):
                continue
            cand_names.append(clip_frame_dir_abs)  # use absolute path here instead of file name to keep it unique
            if clip_frame_dir_or_file.startswith("positive"):
                label_name = clip_frame_dir_abs
            cand_frame_paths = load_frames(clip_frame_dir_abs)
            cand_frame_paths = sample_frames(cand_frame_paths, num_segments=num_clip_frames)
            cand_frames.append({"bytes": [None] * len(cand_frame_paths), "paths": cand_frame_paths,
                                "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(cand_frame_paths)})

        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)] * len(cand_names))
        cand_images.append(cand_frames)
        dataset_infos.append({
            "cand_names": cand_names,
            "label_name": label_name,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "moment_retrieval"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_moment_retrieval_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]

    dataset = load_dataset("json", data_files=kwargs["data_path"], split=kwargs["dataset_split"])
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'
    
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64, drop_last_batch = False,
                          # load_from_cache_file=False
                          )
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
    return dataset
