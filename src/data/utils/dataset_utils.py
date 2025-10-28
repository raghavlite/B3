import sys

from datasets import load_dataset
from src.utils import print_rank



def sample_dataset(dataset, **kwargs):
    dataset_name = kwargs["dataset_name"]
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)

    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if type(num_sample_per_subset) is int and num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
    print_rank(f"Subsample {dataset_name} to {len(dataset)} samples")

    return dataset


def load_qrels_mapping(qrels):
    """
    Load query_id to corpus_id for BEIR format.
    Returns:
        {
        "qid1": ["docA", "docB"],
        "qid2": ["docC"],
        ...
        }
    """
    query_id_to_corpus_id = {}

    for row in qrels:
        if row["score"] == 1:
            query_id_to_corpus_id[row["query-id"]] = row["corpus-id"]

    return query_id_to_corpus_id


def load_hf_dataset(hf_path):
    repo, subset, split = hf_path
    if subset and split:
        return load_dataset(repo, subset, split=split)
    elif subset:
        return load_dataset(repo, subset)
    elif split:
        return load_dataset(repo, split=split)
    else:
        return load_dataset(repo)
