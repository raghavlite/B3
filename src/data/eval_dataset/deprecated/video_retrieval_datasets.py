import os

from datasets import load_dataset
from src.data.dataset.vidore_dataset import DATASET_PARSER_NAME
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.data.utils.vision_utils import save_frames, process_video_frames


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution = kwargs['image_resolution']
    eval_num_hard_neg = kwargs['eval_num_hard_neg']
    num_frames = kwargs['num_frames']
    video_dir, frame_dir = kwargs['video_dir'], kwargs['frame_dir']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for query_instruction, query_text, query_video, pos_text, pos_video, neg_text, neg_video in \
        zip(batch_dict['query_instruction'], batch_dict['query_text'], batch_dict['query_video'], batch_dict['pos_text'], batch_dict['pos_video'], batch_dict['neg_text'], batch_dict['neg_video']):
        cand_video_path = [pos_video] + neg_video[:eval_num_hard_neg]
        cand_video_path = [video_dir + file_name for file_name in cand_video_path]

        cand_frames, cand_video_names = [], []
        for video_path in cand_video_path:
            save_frames(video_path, **kwargs)
            video_name = video_path.split('/')[-1].split('.')[0]
            cand_video_names.append(video_name)
            video_frame_path = os.path.join(frame_dir, video_name)
            cand_frames_path = process_video_frames(video_frame_path, num_frames=num_frames)
            cand_frames.append({"bytes": [None] * num_frames, "paths": cand_frames_path,
                                "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames})

        query_texts.append(query_instruction + query_text)
        query_images.append(None)
        cand_texts.append([pos_text] + neg_text[:eval_num_hard_neg])
        cand_images.append(cand_frames)
        dataset_infos.append({
            "cand_video_names": cand_video_names,
            "label_video_name": cand_video_names[0],
        })

    return {"query_text": [query_texts], "query_image": [query_images],
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "video_retrieval"
DATASET_HF_PATH = "ziyjiang/MMEB_Pro_Video_Retrieval"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_video_retrieval_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]
    eval_num_sample_per_subset = kwargs["eval_num_sample_per_subset"]

    dataset = load_dataset(DATASET_HF_PATH, dataset_name, split="test")
    if eval_num_sample_per_subset is not None:
        dataset = dataset.select(range(eval_num_sample_per_subset))
    dataset = dataset.select(range(eval_num_sample_per_subset))
    dataset = dataset.to_iterable_dataset(num_shards=4)  # convert to IterableDataset and multiple shards

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch=False,
                          load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset
