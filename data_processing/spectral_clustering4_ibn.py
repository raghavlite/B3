import ast
import time
import pandas as pd
import networkx as nx
import metis
from collections import defaultdict
from tqdm import tqdm
import math
from collections import Counter
import numpy as np
import os
import random
import sys
import argparse
import numpy as np
import random


# Load preferences from file
def load_preferences(file_path, nmax=1000000, max_count=1000000):
    if file_path.endswith(".npy"):
        return np.load(file_path)[:max_count, :nmax]  # Directly load numpy array if .npy

    preferences = []
    with open(file_path, "r") as f:
        for count, line in enumerate(tqdm(f), 1):
            if count > max_count:
                break
            data = ast.literal_eval(line.strip())  # Parse JSON-like data
            preferences.append(data[-1][:nmax])  # Extract rank-score pairs
            preferences[-1] += [(count - 1, 0)] * (nmax - len(preferences[-1]))  # Start index from zero

    return np.array(preferences)[:max_count, :]

# Create an adjacency list for the graph
def create_adjacency_list(preferences, n_candidates):
    edges = []
    # for i, rank_scores in enumerate(preferences):
    #     for rank, score in rank_scores:
    #         if i != rank:  # Avoid self-loops
    #             edges.append((i, int(rank)))
    for i, rank_scores in tqdm(enumerate(preferences), total=len(preferences), desc="Processing Preferences"):
        for rank in rank_scores:
            if i != rank:  # Avoid self-loops
                edges.append((i, int(rank)))


    edges_set = set(edges)
    # Build adjacency list

    adjacency_list = [[] for _ in range(n_candidates)]
    for i, j in tqdm(edges):
        if(j, i) in edges_set:
            adjacency_list[i].append(j)
            adjacency_list[j].append(i)
    
    return adjacency_list

# Partition the graph using Metis
def metis_partition(adjacency_list, n_clusters):
    edgecut, parts = metis.part_graph(adjacency_list, n_clusters)
    return parts




def compute_cluster_preferences(
    clusters: np.ndarray, 
    preference_lists: np.ndarray, 
    K: int,
    start_point: int
):
    cluster_samples = []
    
    increased_repeats = []

    for cluster in tqdm(clusters):
        # Flatten preference lists for this cluster
        preferences = preference_lists[cluster].flatten()
        # Count occurrences, ensuring array length == dataset_size
        scores = np.bincount(np.concatenate((preferences, cluster)))

        # Penalize elements that appear in top 30 preferences for any cluster member
        top_30_prefs = preference_lists[cluster, :start_point].flatten()
        np.subtract.at(scores, top_30_prefs, 2)  # in-place subtract 1

        
        # Zero out the cluster elements themselves
        scores[cluster] = 0
        # import ipdb; ipdb.set_trace()
        valid_indices = np.where(scores > 0)[0]
        # Normalize the scores for valid indices to create a probability distribution
        score_probs = scores[valid_indices] / scores[valid_indices].sum()

        sample_size = len(cluster) * K
        # sampled_array = np.random.choice(valid_indices, size=sample_size, replace=False, p=score_probs)
       
        repeat_times =1
        increased_repeats.append(False)
        sampled_array=None
        while(sampled_array is None):
            try:
                doubled_valid_indices = np.repeat(valid_indices, repeat_times)
                doubled_probs = np.repeat(score_probs, repeat_times)
                doubled_probs = doubled_probs / doubled_probs.sum()
                sampled_array = np.random.choice(doubled_valid_indices, size=sample_size, replace=False, p=doubled_probs)
            except:
                repeat_times+=1
                increased_repeats[-1]=True

        sampled_elements = sampled_array.reshape(len(cluster), K)
        cluster_samples.append(sampled_elements)
    
    print(f">>>>>>>>>>> Increased repeat for {np.sum(increased_repeats)} clusters out of {len(clusters)}", flush=True)
    all_cluster_samples = np.concatenate(cluster_samples, axis=0)

    return all_cluster_samples.tolist()





def compute_ind_cluster_preferences(
    clusters: np.ndarray, 
    preference_lists: np.ndarray, 
    K: int,
    start_point: int
):
    cluster_samples = []
    

    for cluster in tqdm(clusters):
        # Flatten preference lists for this cluster
        preferences = preference_lists[cluster][:, start_point:]

        m, n = preferences.shape
        
        cols = np.array([np.random.choice(n, size=5, replace=False) for _ in range(m)])
        
        sampled_elements = preferences[np.arange(m)[:, None], cols]        
        # import ipdb; ipdb.set_trace()
        cluster_samples.append(sampled_elements)
    
    all_cluster_samples = np.concatenate(cluster_samples, axis=0)

    return all_cluster_samples.tolist()














def select_random_negatives(current_cluster, neg_text, neg_image_path, num_random = None, cluster_ids=None, final_clusters=None, df=None):
    # Exclude the current cluster
    available_clusters = [c for c in cluster_ids if c != current_cluster]
    
    # Ensure there are enough clusters
    if len(available_clusters) < num_random:
        raise ValueError(f"Not enough clusters to select {num_random} random values.")
    
    # Randomly select distinct clusters
    selected_clusters = random.sample(available_clusters, num_random)
    
    # From each selected cluster, pick a random index
    selected_indices = [random.choice(final_clusters[c]) for c in selected_clusters]
    
    # Extract 'pos_text' and 'pos_image_path' from the original DataFrame
    new_neg_text = df.loc[selected_indices, "pos_text"].tolist()
    new_neg_image_path = df.loc[selected_indices, "pos_image_path"].tolist()
    
    K = len(neg_text)
    neg_text = np.concatenate((neg_text[:-num_random], np.array(new_neg_text)),axis=0)
    neg_image_path = np.concatenate((neg_image_path[:-num_random], np.array(new_neg_image_path)),axis=0)
    try:
        assert K==len(neg_text) and K==len(neg_image_path)
    except:
        import ipdb; ipdb.set_trace()
    return pd.Series([neg_text, neg_image_path])












# Main function to process the file and partition the graph
def main(dataset=None, negs=None, nmax=None, batch_size=None, K=None):
    # dataset = "MSCOCO_i2t"
    # assert dataset in ["MSCOCO_i2t", "MSCOCO_t2i", "VisualNews_i2t", "VisualNews_t2i", "VisDial", "CIRR", "NIGHTS", "WebQA"]
    parquet_file = f"./MMEB-train/{dataset}/train-00000-of-00001.parquet"
    # parquet_file = f"./MMEB-train/{dataset}_HN40.60.80.100/train-00000-of-00001.parquet"
    cluster_size = batch_size
    
    assert nmax>0 and nmax<5000 
    # START_POINT=0
    # file_path = f"./MMEB-evaloutputs/0218_vlm2vec_labelling2500_processed_all/{dataset}/{dataset}_pred_score.txt"
    # file_path = f"./MMEB-evaloutputs/0218_vlm2vec_labelling2500_processed_all/{dataset}/{dataset}_pred_score.txt"
    # file_path = f"./MMEB-evaloutputs/0218_vlm2vec_labelling2500_processed_all/{dataset}/{dataset}_pred_score.txt"
    
    #!change this
    MODEL_BASE_PATH="./MMEB-evaloutputs/Teacher_vlm2vec/"
    MODEL_SHORTFORM="qwen7b"
    # MODEL_BASE_PATH="./MMEB-evaloutputs/0313_labelling10k_rebuttal/CLIP"
    # MODEL_SHORTFORM="clip"
    START_POINT=30
    # START_POINT=70
    # START_POINT=0

    file_path = f"{MODEL_BASE_PATH}/{dataset}/{dataset}_pred_rank.npy"
    print(f">>>>>>>>>>>>>>>>> Start Point Selected {START_POINT}")
    # START_POINT=1
    # file_path = f"./MMEB-evaloutputs/0218_vlm2vec_labelling2500_processed_posrank/{dataset}/{dataset}_pred_score.txt"
    

    # MOD = f"PS_Metis_bs{cluster_size}bi_{START_POINT}.{nmax+START_POINT}.1.{nmax+START_POINT}"
    MOD = f"PS_Metis_bs{cluster_size}bi_{START_POINT}.{nmax+START_POINT}_{MODEL_SHORTFORM}"
    if(negs=="hn"):
        output_parquet_file = f"./MMEB-train/{dataset}_HN{MOD}/train-00000-of-00001.parquet"
        # output_parquet_file = f"./MMEB-train/{dataset}_HN40.60.80.100{MOD}/train-00000-of-00001.parquet"
        # if os.path.exists(output_parquet_file):
        #     return
    elif(negs=='ihn'):
        output_parquet_file = f"./MMEB-train/{dataset}_HNI{MOD}/train-00000-of-00001.parquet"
    elif(negs=="rand"):
        output_parquet_file = f"./MMEB-train/{dataset}_HNrand{MOD}/train-00000-of-00001.parquet"
    elif(negs=="rand1"):
        output_parquet_file = f"./MMEB-train/{dataset}_HNrand1{MOD}/train-00000-of-00001.parquet"
    elif(negs=="rand2"):
        output_parquet_file = f"./MMEB-train/{dataset}_HNrand2{MOD}/train-00000-of-00001.parquet"
    elif(negs=="rand3"):
        output_parquet_file = f"./MMEB-train/{dataset}_HNrand3{MOD}/train-00000-of-00001.parquet"
    elif(negs=="rand4"):
        output_parquet_file = f"./MMEB-train/{dataset}_HNrand4{MOD}/train-00000-of-00001.parquet"
    else:
        assert False, "dataset nit accepted"


    # Step 1: Load preferences
    #!startpoint is zero here
    preferences = load_preferences(file_path, nmax=(START_POINT+nmax))
    
    n_candidates = len(preferences)
    n_clusters = n_candidates//cluster_size

    non_conforming_clusters = math.ceil(n_candidates/cluster_size)-n_clusters


    # Step 2: Create the adjacency list
    #!
    adjacency_list = create_adjacency_list(preferences[:,START_POINT:(START_POINT+nmax)], n_candidates)

 
    # Step 3: Partition the graph
    print("Partitioning with Metis...", flush=True)
    st = time.time()
    cluster_labels = metis_partition(adjacency_list, n_clusters)
    et = time.time()
    print(et-st)

    cluster_sizes = Counter(cluster_labels)
    for cluster_id, size in cluster_sizes.items():
        print(f"Cluster {cluster_id} size: {size}")


    # Step 4: Organize candidates into clusters
    clusters = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_labels):

        clusters[cluster_id].append(idx)

    #sorting by length important to maintain the constraint on reamining clusters.
    clusters_values = clusters.values()
    clusters_values = sorted(clusters_values, key=len, reverse=True)

    # Step 5: Create the final clusters ensuring size `cluster_size`
    final_clusters = []
    remaining_candidates = []

    # Collect clusters with exact size and prepare remaining candidates
    for cluster in clusters_values:
        while len(cluster) > cluster_size:
            # Pop one element from oversized cluster and add to remaining
            remaining_candidates.append(cluster.pop())

        # If cluster is smaller than cluster_size, fill it with items from remaining_candidates
        while len(cluster) < cluster_size and remaining_candidates:
            cluster.append(remaining_candidates.pop())

        # Add the cluster to final clusters
        final_clusters.append(cluster)

    final_clusters = sorted(final_clusters, key=lambda x: min(x))
    # If there are any leftover remaining candidates, make a final cluster for them
    if remaining_candidates:
        final_clusters.append(remaining_candidates)


    if len(final_clusters[-1]) != cluster_size:
        big_cluster = final_clusters.pop()  # remove that big leftover cluster
        # Split into chunks of size cluster_size
        splitted = []
        for i in range(0, len(big_cluster), cluster_size):
            splitted.append(big_cluster[i:i + cluster_size])

        # If the last chunk is smaller, fill it from earlier full-size clusters
        if len(splitted[-1]) < cluster_size:
            leftover_chunk = splitted.pop()
            needed = cluster_size - len(leftover_chunk)

            # Take random items from the *earlier* final_clusters (which are presumably full).
            while(needed>0):
                for c in final_clusters:
                    if(needed==0):
                        break;
                    # Only take if the cluster is strictly > cluster_size (meaning it can spare at least one).
                    idx_to_move = random.randrange(len(c))
                    leftover_chunk.append(idx_to_move)
                    needed -= 1

            # Put back our now-filled chunk
            splitted.append(leftover_chunk)

        # Finally, put all these splitted pieces back
        final_clusters.extend(splitted)


    # Ensure correctness

    assert (sum(len(group) for group in final_clusters) >= n_candidates) and (sum(len(group) for group in final_clusters) < (n_candidates+cluster_size)), "Some candidates are unassigned!"
    # assert np.sum([len(group)!=cluster_size for group in final_clusters]) == non_conforming_clusters, "non conforming clusters issue"
    # import ipdb; ipdb.set_trace()
    assert np.all([len(group)==cluster_size for group in final_clusters]), f"all clusters are not {cluster_size} length"



    index_to_cluster = {idx: cluster_id for cluster_id, cluster in enumerate(final_clusters) for idx in cluster}
    
    ordered_indices = [idx for cluster in final_clusters for idx in cluster]
    # ordered_indices = [idx for cluster in final_clusters for idx in cluster for i in range()]

    if negs=='ihn':
        hn_indices_all = compute_ind_cluster_preferences(np.array(final_clusters),preferences[:,:(START_POINT+100)].astype(int), K=K, start_point=START_POINT)
    else:
        hn_indices_all = compute_cluster_preferences(np.array(final_clusters),preferences[:,:(START_POINT+100)].astype(int), K=K, start_point=START_POINT)

    df = pd.read_parquet(parquet_file)
    hn_text = [df.loc[hn_indices, "pos_text"].tolist() for hn_indices in hn_indices_all]
    hn_image_path = [df.loc[hn_indices, "pos_image_path"].tolist() for hn_indices in hn_indices_all]
    df_reordered = df.iloc[ordered_indices].reset_index(drop=True)
    df_reordered["neg_text"] = hn_text
    df_reordered["neg_image_path"] = hn_image_path






    if("rand" in negs):
        df_reordered['cluster_id'] = [index_to_cluster[idx] for idx in ordered_indices]
        total_clusters = len(final_clusters)
        cluster_ids = list(range(total_clusters))
        if("1" in negs):
            df_reordered[['neg_text', 'neg_image_path']] = df_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=1, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df), axis=1)

        elif("2" in negs):
            df_reordered[['neg_text', 'neg_image_path']] = df_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=2, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df ), axis=1)
        elif("3" in negs):
            df_reordered[['neg_text', 'neg_image_path']] = df_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=3, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df ), axis=1)
        elif("4" in negs):
            df_reordered[['neg_text', 'neg_image_path']] = df_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=3, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df ), axis=1)
        else:
            df_reordered[['neg_text', 'neg_image_path']] = df_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=5, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df ), axis=1)
        # parquet_file =  parquet_file.replace("_HN40.60.80.100", "_HNrand")
        # Step 7: Save the reordered DataFrame

    # output_parquet_file = parquet_file.replace('/train', f'{MOD}/train')
    folder_name = os.path.dirname(output_parquet_file)
    os.makedirs(folder_name, exist_ok=True)
    
    df_reordered.to_parquet(output_parquet_file, index=False)

    print(f"Reordered dataset saved to {output_parquet_file}", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process dataset and partition the graph.")
    
    parser.add_argument("--dataset", type=str, choices=[
        "MSCOCO_i2t", "MSCOCO_t2i", "VisualNews_i2t", "VisualNews_t2i", 
        "VisDial", "CIRR", "NIGHTS", "WebQA", "OK-VQA", "A-OKVQA", "ChartQA", "DocVQA", "InfographicsVQA", "Visual7W", "MSCOCO"
    ], help="Dataset name")
    

    parser.add_argument("--negs", type=str, choices=["hn", "ihn", "rand", "rand1", "rand2", "rand3", "rand4"], help="Negative sampling method")
    
    parser.add_argument("--nmax", type=int, help="Maximum value (1-4999)")
    parser.add_argument("--batch_size", type=int, help="Maximum value (1-4999)")
    parser.add_argument("--K", type=int, help="Maximum value (1-4999)")
    
    
    args = parser.parse_args()
    
    main(args.dataset, args.negs, args.nmax, args.batch_size, args.K)
