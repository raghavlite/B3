import os

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance
from src.data.utils.dataset_utils import sample_dataset, load_qrels_mapping
from src.model.processor import process_input_text


TASK_INST_QRY = "Find a document image that matches the given query:"
TASK_INST_TGT = "Understand the content of the provided video."

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    query_id_to_corpus_id = kwargs['query_id_to_corpus_id']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for query_id, query in zip(batch_dict['query-id'], batch_dict['query']):
        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=query)])
        query_images.append([None])

        corpus_id = query_id_to_corpus_id[query_id]
        image_path = f'{image_root}/{corpus_id}.png'
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image path {image_path} not found.')
        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True)])
        cand_images.append([ImageVideoInstance(
            bytes=[None],
            paths=[image_path],
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
        ).to_dict()])
        dataset_infos.append({
            "cand_names": [corpus_id],
            "label_name": corpus_id,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


def corpus_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    cand_texts, cand_images = [], []
    for corpus_id, image in zip(batch_dict['corpus-id'], batch_dict['image']):
        image_path = f'{image_root}/{corpus_id}.png'
        if not os.path.exists(image_path):
            image.save(image_path)
        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True)])
        cand_images.append([ImageVideoInstance(
            bytes=[None],
            paths=[image_path],
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
        ).to_dict()])

    return {"cand_text": cand_texts, "cand_image": cand_images}


DATASET_PARSER_NAME = "vidore"
DATASET_HF_PATH = "vidore/arxivqa_test_subsampled_beir"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_vidore_dataset(model_args, data_args, training_args, *args, **kwargs):
    # BEIR format
    qrels = load_dataset(DATASET_HF_PATH, "qrels", split="test")
    corpus = load_dataset(DATASET_HF_PATH, "corpus", split="test")
    dataset = load_dataset(DATASET_HF_PATH,  "queries", split="test")
    query_id_to_corpus_id = load_qrels_mapping(qrels)
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['query_id_to_corpus_id'] = query_id_to_corpus_id

    corpus = corpus.map(lambda x: corpus_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch = False, load_from_cache_file=False)
    corpus = corpus.select_columns(['cand_text', 'cand_image'])
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch = False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, corpus
