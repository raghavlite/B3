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
import orjson


# def load_preferences(file_path, nmax=None, max_count=50000):
#     preferences = []
#     # max_count=100
#     with open(file_path, "rb") as f:
#         # preferences = json.loads(f.read())
#         for count, line in enumerate(tqdm(f), 1):
#             if count > max_count:
#                 break
#             data = orjson.loads(line.strip())  # Parse JSON
#             #! change this to start from zero
#             # preferences.append([num for num in data[-1][:nmax] if num[0] < max_count])  # Extract rank-score pairs
#             preferences.append(data[-1][:nmax])  # Extract rank-score pairs
#             # preferences[-1]+=[(count,0)]*(nmax-len(preferences[-1]))
#     return np.array(preferences)[:max_count,:, :]




def load_preferences(file_path):
    """
    Loads ranking lists (preferences) from a file and filters them to retain only ranking pairs
    whose indices are present in the given subset_indices.
    
    Args:
        file_path (str): Path to the file containing the JSON lines.
        nmax (int, optional): Maximum number of ranking pairs to retain for each data point.
        max_count (int): Maximum number of lines (data points) to process.
        subset_indices (iterable, optional): Collection of indices that should be retained in each ranking list.
                                               Only pairs with a first element (index) in this subset are kept.
    
    Returns:
        np.array: An array of filtered ranking lists for each data point.
                  Note: If ranking lists vary in length, the array will be of dtype=object.
    """   
    ranking_matrix = np.load(file_path)  # Directly load numpy array if .npy
    return ranking_matrix




def first_n_unique(seq, n, remove_list=None):
    # Convert remove_list to a set for fast membership tests, if provided.
    remove_set = set(remove_list) if remove_list is not None else set()
    
    unique_items = []
    seen = set()  # Track items we've already added.
    
    for item in seq:
        # Skip if the item is in the removal set.
        if item in remove_set:
            continue
        # Only add the item if it hasn't been seen yet.
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
        # Stop once we have n unique items.
        if len(unique_items) == n:
            break
    return unique_items


def create_adjacency_list(preferences, df, num_unique):
    edges = []
    
    # Precompute a mapping from pos_text value to list of DataFrame indices for efficiency.
    pos_text_to_indices = {}
    for idx, pos in df['pos_text'].items():
        try:
            pos_text_to_indices.setdefault(pos, []).append(idx)
        except:
            import ipdb; ipdb.set_trace()
    # Loop over each point i (each row in the preferences array)
    for i in tqdm(range(preferences.shape[0]), desc="creating unique labels"):
        # Get the indices from the i-th row
        indices = preferences[i, :]
        # Retrieve the corresponding pos_text values from the DataFrame.
        # Assuming that the indices in the preferences array are valid row labels in df.
        try:
            pos_texts = df.loc[indices, 'pos_text'].tolist()
        except:
            import ipdb; ipdb.set_trace()
        golden_label = df.loc[i,"pos_text"]
        # Get the first num_unique unique pos_text values
        #!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>change this IMP IMP IMP IMP IMP
        unique_items = first_n_unique(pos_texts, num_unique,[golden_label])
        # unique_items = first_n_unique(pos_texts, num_unique)
        
        # For each unique pos_text, get all DataFrame indices (points j) that have that value.
        for pos in unique_items:
            j_indices = pos_text_to_indices.get(pos, [])
            # Create an edge from i to each j
            for j in j_indices:
                edges.append((i, j))
    
    edges_set = set(edges)
    n_candidates = len(preferences)

    adjacency_list = [[] for _ in range(n_candidates)]
    for i, j in tqdm(edges, desc="Adding Edges"):
        if(j<i):
            continue
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
    num_unique,
    df):
    
    cluster_samples = []
    increased_repeats = []

    for cluster in tqdm(clusters, desc="Adding Negatives"):
        aggregated_scores = {}  # key: pos_text, value: aggregated score
        
        # 1. Aggregate the first K unique pos_text values for each datapoint.
        for datapoint in cluster:
            pos_texts = df.loc[preference_lists[datapoint], 'pos_text'].tolist()
            golden_label = df.loc[datapoint,"pos_text"]
            #!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>change this IMP IMP IMP IMP IMP
            unique_texts = first_n_unique(pos_texts, num_unique, [golden_label])
            # unique_texts = first_n_unique(pos_texts, num_unique)
            for text in unique_texts:
                aggregated_scores[text] = aggregated_scores.get(text, 0) + 1

        #! this is the major change from old version                       
        # 3. Exclude pos_text values that belong to datapoints in the cluster.
        # for datapoint in cluster:
        #     cluster_text = df.loc[datapoint, 'pos_text']
        #     # Remove or zero out this candidate if present
        #     if (cluster_text in aggregated_scores) and len(aggregated_scores)>len(cluster):
        #         aggregated_scores[cluster_text] = 0
        

        candidates = [(k, v) for k, v in aggregated_scores.items() if v > 0]
        if len(candidates) == 0:
            # import ipdb; ipdb.set_trace()
            # If no candidates are available, return an empty array.
            assert False, "Something wrong"
        
        # Split candidates into keys and their corresponding scores.
        keys = np.array([k for k, _ in candidates])
        values = np.array([v for _, v in candidates])

        score_probs = values / values.sum()
        sample_size = len(cluster) * K
        repeat_times = 1
        increased_repeats.append(False)
        sampled_array = None

        # Attempt to sample without replacement. If an error occurs, repeat with duplication.
        while sampled_array is None:
            try:
                doubled_keys = np.repeat(keys, repeat_times)
                doubled_probs = np.repeat(score_probs, repeat_times)
                doubled_probs = doubled_probs / doubled_probs.sum()
                sampled_array = np.random.choice(doubled_keys, size=sample_size, replace=False, p=doubled_probs)
            except Exception as e:
                repeat_times += 1
                increased_repeats[-1] = True
        # Reshape the sampled array to (number of datapoints in cluster, K)
        sampled_elements = sampled_array.reshape(len(cluster), K)

        cluster_samples.append(sampled_elements)

    print(f">>>>>>>>>>> Increased repeat for {np.sum(increased_repeats)} clusters out of {len(clusters)}", flush=True)
    all_cluster_samples = np.concatenate(cluster_samples, axis=0)

    # import ipdb; ipdb.set_trace()
    return all_cluster_samples.tolist()



def compute_ind_cluster_preferences(clusters: np.ndarray, preference_lists: np.ndarray, K: int, num_unique, df):

    cluster_samples = []
    increased_repeats = []

    for cluster in tqdm(clusters, desc="Adding Negatives"):
        aggregated_scores = {}  # key: pos_text, value: aggregated score
        
        # 1. Aggregate the first K unique pos_text values for each datapoint.
        cluster_unique_texts = []
        for datapoint in cluster:
            pos_texts = df.loc[preference_lists[datapoint], 'pos_text'].tolist()
            golden_label = df.loc[datapoint,"pos_text"]
            unique_texts = first_n_unique(pos_texts, num_unique, [golden_label])
            cluster_unique_texts.append(unique_texts)
        
        try:
            sampled_elements = [np.random.choice(unique_texts, size=5, replace=False) for unique_texts in cluster_unique_texts]
        except:
            sampled_elements = [np.random.choice(unique_texts, size=5, replace=True) for unique_texts in cluster_unique_texts]

        cluster_samples.append(sampled_elements)

    print(f">>>>>>>>>>> Increased repeat for {np.sum(increased_repeats)} clusters out of {len(clusters)}", flush=True)
    all_cluster_samples = np.concatenate(cluster_samples, axis=0)

    # import ipdb; ipdb.set_trace()
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
def main(dataset=None, negs=None, nmax=None, batch_size=None, K=None, top_KC=None):
    # dataset = "MSCOCO_i2t"
    # assert dataset in ["MSCOCO_i2t", "MSCOCO_t2i", "VisualNews_i2t", "VisualNews_t2i", "VisDial", "CIRR", "NIGHTS", "WebQA"]
    parquet_file = f"./MMEB-train/{dataset}/train-00000-of-00001.parquet"
    # parquet_file = f"./MMEB-train/{dataset}_HN40.60.80.100/train-00000-of-00001.parquet"
    cluster_size = batch_size
    


    #!change these two>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMP IMP IMP
    # BASE_FOLDER_PATH = "./MMEB-evaloutputs/0313_labelling10k/VLM2Vec-Qwen2VL-7B"
    # MODEL_SHORTFORM="qwen7b"
    BASE_FOLDER_PATH = "./MMEB-evaloutputs/0313_labelling10k_rebuttal/CLIP"
    MODEL_SHORTFORM="clip"

    rank_path = f"{BASE_FOLDER_PATH}/{dataset}/{dataset}_pred_rank{{}}.npy"
    score_path = f"{BASE_FOLDER_PATH}/{dataset}/{dataset}_pred_score{{}}.npy"
    indices_path = f"{BASE_FOLDER_PATH}/{dataset}/{dataset}_batch_indices{{}}.npy"


    #!change this>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMP IMP IMP
    MOD = f"PS_Metis_bs{cluster_size}bi_{top_KC}P_{2*K}_{K}_{MODEL_SHORTFORM}"
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
    df = pd.read_parquet(parquet_file)
    num_unique_classes = df['pos_text'].nunique()
    num_unique_graph = math.ceil(num_unique_classes*top_KC/100)
    df_list = []
    n_total = len(df)     # should be 50,000
    n_subset = 10000

    num_parts = math.ceil(n_total/n_subset)
    # Randomly shuffle indices from 0 to 49,999
    # shuffled_indices = np.random.permutation(n_total)
    # start_idx=0
    for partidx in range(1, 1+num_parts):
        # print(f"Doing Subset {start_idx} - {(start_idx + n_subset)}")
        # subset_indices = shuffled_indices[start_idx: (start_idx + n_subset)]
        # subset_indices = sorted(subset_indices)
        subset_indices = np.load(indices_path.format(partidx))
        df_subset = df.iloc[subset_indices].reset_index(drop=True)

        
        preferences = load_preferences(rank_path.format(partidx)).astype(int)
        

        n_candidates = len(preferences)
        n_clusters = n_candidates//cluster_size

        non_conforming_clusters = math.ceil(n_candidates/cluster_size)-n_clusters


        # Step 2: Create the adjacency list
        #!
        adjacency_list = create_adjacency_list(preferences[:,:], df_subset, num_unique=num_unique_graph)

    
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
        # n_candidates=100
        assert (sum(len(group) for group in final_clusters) >= n_candidates) and (sum(len(group) for group in final_clusters) < (n_candidates+cluster_size)), "Some candidates are unassigned!"
        assert np.all([len(group)==cluster_size for group in final_clusters]), f"all clusters are not {cluster_size} length"



        index_to_cluster = {idx: cluster_id for cluster_id, cluster in enumerate(final_clusters) for idx in cluster}
        
        ordered_indices = [idx for cluster in final_clusters for idx in cluster]
        # ordered_indices = [idx for cluster in final_clusters for idx in cluster for i in range()]


        if(negs=='ihn'):
            hn_pos_text_all = compute_ind_cluster_preferences(np.array(final_clusters), preferences[:,:].astype(int), K=K, num_unique=2*K, df=df_subset)
        else:
            hn_pos_text_all = compute_cluster_preferences(np.array(final_clusters), preferences[:,:].astype(int), K=K, num_unique=2*K, df=df_subset)


        df_subset_reordered = df_subset.iloc[ordered_indices].reset_index(drop=True)
        df_subset_reordered["neg_text"] = hn_pos_text_all
        df_subset_reordered["neg_image_path"] = [[""]*len(ll) for ll in hn_pos_text_all]

        if("rand" in negs):
            df_subset_reordered['cluster_id'] = [index_to_cluster[idx] for idx in ordered_indices]
            total_clusters = len(final_clusters)
            cluster_ids = list(range(total_clusters))
            if("1" in negs):
                df_subset_reordered[['neg_text', 'neg_image_path']] = df_subset_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=1, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df_subset), axis=1)
            elif("2" in negs):
                df_subset_reordered[['neg_text', 'neg_image_path']] = df_subset_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=2, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df_subset ), axis=1)
            elif("3" in negs):
                df_subset_reordered[['neg_text', 'neg_image_path']] = df_subset_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=3, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df_subset), axis=1)
            elif("4" in negs):
                df_subset_reordered[['neg_text', 'neg_image_path']] = df_subset_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=3, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df_subset), axis=1)
            else:
                df_subset_reordered[['neg_text', 'neg_image_path']] = df_subset_reordered.apply(lambda x: select_random_negatives(current_cluster=x['cluster_id'], neg_text=x["neg_text"], neg_image_path=x["neg_image_path"], num_random=5, cluster_ids=cluster_ids, final_clusters=final_clusters, df=df_subset), axis=1)
            # parquet_file =  parquet_file.replace("_HN40.60.80.100", "_HNrand")
            # Step 7: Save the reordered DataFrame
        df_list.append(df_subset_reordered)

    df_reordered = pd.concat(df_list, ignore_index=True)
    # output_parquet_file = parquet_file.replace('/train', f'{MOD}/train')
    folder_name = os.path.dirname(output_parquet_file)
    os.makedirs(folder_name, exist_ok=True)
    
    df_reordered.to_parquet(output_parquet_file, index=False)

    print(f"Reordered dataset saved to {output_parquet_file}", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process dataset and partition the graph.")
    
    parser.add_argument("--dataset", type=str, choices=[
        "MSCOCO_i2t", "MSCOCO_t2i", "VisualNews_i2t", "VisualNews_t2i", 
        "VisDial", "CIRR", "NIGHTS", "WebQA", "OK-VQA", "A-OKVQA", "ChartQA", "DocVQA", "InfographicsVQA", "Visual7W", "ImageNet_1K", "HatefulMemes", "SUN397", "N24News", "VOC2007"
    ], help="Dataset name")
    

    parser.add_argument("--negs", type=str, choices=["hn", "ihn", "rand", "rand1", "rand2", "rand3", "rand4"], help="Negative sampling method")
    
    parser.add_argument("--nmax", default=5000, type=int, help="Maximum value (1-4999)")
    parser.add_argument("--batch_size", type=int, help="Maximum value (1-4999)")
    parser.add_argument("--K", type=int, help="Maximum value (1-4999)")
    parser.add_argument("--topKC", type=int, help="Between 1 and 100")
    
    
    args = parser.parse_args()
    
    main(args.dataset, args.negs, args.nmax, args.batch_size, args.K, args.topKC)
