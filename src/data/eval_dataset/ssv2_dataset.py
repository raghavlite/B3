import os
import sys

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.data.utils.vision_utils import save_frames, load_frames, sample_frames
from src.model.processor import process_input_text


TASK_INST_QRY = "Select the sentence that best describes the actions and scenes in the given video."
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution = kwargs['image_resolution']
    num_frames = kwargs['num_frames']
    video_dir, frame_root = kwargs['video_dir'], kwargs['frame_dir']
    dataset_name = kwargs['dataset_name']
    model_backbone = kwargs['model_backbone']
    max_frames_saved = kwargs['max_frames_saved']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_name, pos_text, cand_text in \
        zip(batch_dict['video_id'], batch_dict['pos_text'], batch_dict['neg_text']):

        video_path = os.path.join(video_dir, str(video_name) + '.mp4')
        frame_dir = os.path.join(frame_root, str(video_name))
        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = load_frames(frame_dir)
        video_frame_paths = sample_frames(video_frame_paths, num_segments=num_frames)

        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, add_video_token=True)])
        query_images.append([{"bytes": [None] * len(video_frame_paths), 'paths': video_frame_paths,
                              'resolutions': [RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths)}])

        cand_text = cand_text
        cand_texts.append(cand_text)
        cand_images.append([None] * len(cand_text))

        dataset_info = {
            "cand_names": cand_text,
            "label_name": pos_text,
        }
        dataset_infos.append(dataset_info)

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "ssv2"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_ssv2_dataset(model_args, data_args, training_args, *args, **kwargs):
    """
    ssv2-mc setup for zero-shot evaluation.
    """
    data_path = kwargs["data_path"]

    dataset = load_dataset('json', data_files=data_path)['train']
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch=False,
                          load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset
