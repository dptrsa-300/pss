from google.cloud import storage

import argparse
import gzip
import os
import sys
import time
from datetime import datetime
import string
import io
import itertools
import pickle

import numpy as np
import pandas as pd

import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score

from google.cloud import storage
from google.cloud.storage import Blob

# Ask how to work with this one 
sys.path.append(f"{os.path.dirname(__file__)}/")

import gcs_utils as gcs

import urllib.parse
import urllib.request

import matplotlib
from matplotlib.pyplot import figure
from matplotlib import pyplot
import matplotlib.pyplot as plt 

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from kneed import KneeLocator

def deepfold_file_processor(key):    
    """Download and parse a DeepFold embedding file. """
    df_emb_decode = pd.read_csv(io.BytesIO(gcs.download_blob(key)))

    protein = np.array(df_emb_decode.dropna().protein_id)
    X = np.vstack(df_emb_decode["deepfold"].dropna(
                  ).apply(lambda embed: np.array(str(embed)[1:-1].split()
                                                )
                         )
                 )
    missing_protein = list(df_emb_decode[df_emb_decode.deepfold.isna()]["protein_id"])
    
    return X, protein, missing_protein 

def DO_NOT_USE_deepfold_file_processor(key):
    """DO NOT USE. Version with a bug."""
    
    X = np.empty((0,398), dtype=float)
    protein = np.empty((0,1), dtype=str)
    
    # Download and split the str into list 
    df_emb_decode = gcs.download_text(key).split(",")    
    
    # If any embedding hasn't been generated, then put it into missing 
    missing_protein=[]

    # The first two items are metadata. Start on index 2. 
    i = 2
    while i < len(df_emb_decode):
        pair = df_emb_decode[i].rsplit('\n', 1)
        
        # parse feature vec. Remove double quotes and brackets. Split and cast as float.
        feature_vec = np.array(pair[0][2:-2].split()).astype(float)
        protein_id = pair[1]

        # Only take the vector if we have a feature vec of length 398 
        if len(feature_vec) != 398:
            missing_protein.append(protein_id)
            i+=1
            continue

        X = np.concatenate((X, feature_vec.reshape(1,398)))
        protein = np.append(protein, protein_id)
        i+=1
    
    return X, protein, missing_protein 

def import_deepfold_embeddings(keys):
    missing_full=np.empty((0,1), dtype=str)
    X_hold = [] 
    protein_name_full=np.empty((0,1), dtype=str)
    z=0

    for key in keys:
        # I actually only need the file path once in the right storage.
        key = gcs.uri_to_bucket_and_key(key)[1]

        # parse the file 
        X, protein, missing_protein = deepfold_file_processor(key)

        # Put it into list 
        X_hold.append(X)
        protein_name_full = np.append(protein_name_full, protein)
        missing_full = np.append(missing_full, missing_protein)
        print(key)
        

    # Stack all the X 
    X_full = np.vstack(X_hold)
    
    return X_full, missing_full, protein_name_full 

def find_elbow(n_neighbors, 
               X_embed, 
               curve='convex',   # convex or concave 
               direction="decreasing"   # increasing or decreasing
              ):
    beg = datetime.now()
    ts = beg.strftime("%Y-%m-%d-%H:%M")
    print("###############")
    print(ts)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_embed)
    distances, indices = nbrs.kneighbors(X_embed)
    distance_desc = sorted(distances[:,-1])
#     distance_desc = sorted(distances[:,-1], reverse=True)
    plt.plot(distance_desc)
    plt.show()

    kneedle = KneeLocator(range(1,len(distance_desc)+1),  #x values
                          distance_desc, # y values
                          S=1.0, #parameter suggested from paper
                          curve=curve, #parameter from figure
                          direction=direction, #parameter from figure
                          online=True
                         )

    kneedle.plot_knee_normalized()
    plt.show()
    
    print("Run time:",  datetime.now() - beg )
    print()
    print("n_neighbors = {}".format(n_neighbors))
    print("Elbow:", kneedle.elbow)
    print("Elbow Y:", kneedle.elbow_y)
    
    return kneedle.elbow, kneedle.elbow_y
    
def merge_cluster_stats(stats_1, stats_2):
    """If there are two dataframes with cluster_label and corresponding stats,
    merge the two into one df."""
    
    return stats_1.join(stats_2)



def sequence_stats(clusters,
                   sequences=None,
                   seq_parquet="structure_files/sequences/sequences.parquet"):
    """Get amino acid sequence-related stats for each cluster.
    The input `clusters` is a pd.df that has two columns: 'protein' and 'cluster_label' """
    
    # Download sequence stats if not exists
    if isinstance(sequences, type(None)):
        sequences = gcs.download_parquet(seq_parquet)
        sequences["seq_len"] = sequences["pdbx_seq_one_letter_code"].str.len()

    # For each protein in sequence, map in the cluster label
    cluster_stats = clusters\
            .merge(sequences,
                  how='outer',
                  left_on='protein',
                  right_on='protein_id'
                 ).reset_index()[['protein', 'cluster_label', 'db_code', 'db_name','pdbx_seq_one_letter_code', 'protein_filename', 'seq_len']]

    cluster_stats=cluster_stats.rename(columns={'index': "protein"})

    # Summarize stats by cluster. For each cluster, show min, max, avg etc of seq length. 
    cluster_stats = pd.pivot_table(cluster_stats, values="seq_len",
                             index="cluster_label",
                             aggfunc={"seq_len": [len, np.mean, np.std, np.min, np.median, np.max,
                                                 lambda x: list(x)]}).reset_index()
    
    cluster_stats = cluster_stats.rename(columns={'<lambda_0>': "seq_len_arr", 
                              'amax': "max_seq_len", 
                              'amin': "min_seq_len", 
                              'mean': "mean_seq_len", 
                              'median': "median_seq_len", 
                              'len': "num_proteins",
                              'std': "std_seq_len"})[['cluster_label', "num_proteins", 'mean_seq_len', 'min_seq_len',  'median_seq_len', 'max_seq_len', 
       'std_seq_len','seq_len_arr']]
    
    return cluster_stats 


def download_asp(prefix='structure_files/atom_sites', 
                 n=None,
                 print_status=False):
    """Downloads protein files, then summarizes amino acid sequence-level confidence information."""
    keys = gcs.list_file_paths(prefix)
    
    if not n:
        n = len(keys)

    # Download, dedupe, and add 
    asp = pd.DataFrame(columns=["protein_id", "label_seq_id", "pdbx_sifts_xref_db_res", "confidence_pLDDT"])

    for key in keys[:n]:
        if print_status:
            print(key)
        # Download file 
        asp_i = gcs.download_parquet(gcs.uri_to_bucket_and_key(key)[1])

        # Dedupicate it at the amino acid level 
        asp_i = asp_i[["protein_id", "label_seq_id", "pdbx_sifts_xref_db_res", "confidence_pLDDT"]
                   ].drop_duplicates()
        # Convert data type 
        asp_i["label_seq_id"] = asp_i["label_seq_id"].astype('float64')
        asp_i["confidence_pLDDT"] = asp_i["confidence_pLDDT"].astype('float64')

        # Add to the all dataset
        asp = asp.append(asp_i)

    # Deduplicate asp because there may be data present across different files 
    asp = asp[["protein_id", "label_seq_id", "pdbx_sifts_xref_db_res", "confidence_pLDDT"]
                   ].drop_duplicates()

    return asp
    
def protein_confidence_agg(clusters, asp):
    '''
    Confidence level at sequence level 
    '''
   
    
    # Find avg confidence per protein
    avg_conf_protein = pd.pivot_table(asp,
                                      index="protein_id",
                                      values="confidence_pLDDT",
                                      aggfunc=np.mean
                                     ).reset_index()
    avg_conf_protein.columns=["protein", "protein_confidence"]

    # Add confidence category for amino acid
    asp["confidence"] = pd.cut(asp["confidence_pLDDT"], 
                                       [0, 50, 70, 90, 100], 
                                       labels=['D', 'L', 'M', 'H'],
                                       right=False)

    # Show count of amino acids under each confidence category
    asp_pivot = pd.pivot_table(asp,
                              index=["protein_id"],
                              columns=["confidence"],
                              values=["label_seq_id"],
                              aggfunc=[len]
                             ).fillna(0).reset_index()

    asp_pivot.columns=["protein", "D", "L", "M", "H"]
    
    asp = asp_pivot.set_index("protein").join(avg_conf_protein.set_index("protein"), how="outer").reset_index()
    
    asp["D_perc"] = asp["D"]/(asp["D"] + asp["L"] + asp["M"] + asp["H"])
    
    # Join cluster numbers to each protein 
    asp = asp.set_index(["protein"])\
            .join(clusters.set_index(["protein"]),
                  how='outer'
                 ).reset_index()
    
    # Summarize by cluster
    cluster_conf = pd.pivot_table(asp, index="cluster_label",
              values=["protein_confidence", "D_perc"],
              aggfunc=np.mean)
    
    # Return cluster-level confidence data
    return cluster_conf

def protein_confidence_agg_protein_level(n=None):
    """Downloads protein files, then summarizes protein-level confidence information."""
    
    prefix = 'structure_files/sequences'
    keys = gcs.list_file_paths(prefix)
    
    if not n:
        n = len(keys)

    # Download, dedupe, and add 
    asp = gcs.download_parquet(gcs.uri_to_bucket_and_key(keys[0])[1])

    # Deduplicate asp because there may be data present across different files 
    asp = asp[["protein_id", "confidence_pLDDT"]
                   ].drop_duplicates()
    
    # Find avg confidence per protein
    avg_conf_protein = pd.pivot_table(asp,
                                      index="protein_id",
                                      values="confidence_pLDDT",
                                      aggfunc=np.mean
                                     ).reset_index()
    avg_conf_protein.columns=["protein", "protein_confidence"]

    # Add confidence category for amino acid
    asp["confidence"] = pd.cut(asp["confidence_pLDDT"], 
                                       [0, 50, 70, 90, 100], 
                                       labels=['D', 'L', 'M', 'H'],
                                       right=False)
    
    # Return df with the conf data for amino acids and protein level 
    return asp

def get_go_dict(fname='c5.go.mf.v7.4.symbols.gmt'):
    """Download GO functions as Gene Symbols from GSEA"""
    go = gcs.download_text(fname)
    go_dict = {e.split("\t")[0]: e.split("\t")[2:] for e in go.split("\n")}
    
    return go_dict

def get_protein_id_for_genename(go_dict=None):
    """Once gene ontology dict is downloaded, get protein ID for each gene & function"""
    
    protein_lookup = {}
    
    # If go_dict exists, no need to re-download. 
    if not go_dict:
        go_dict = get_go_dict()
    
    # Parse out the results inside the dictionary 
    url = 'https://www.uniprot.org/uploadlists/'
    
    for function, proteins in go_dict.items():
        
        # Use uniprot API to map protein ID into gene name 
        # Question - can I filter this by just human? 
        # Gotta handle the ones that could not be found in the api
        params = {
            'from': 'GENENAME',
            'to': 'SWISSPROT',
            'format': 'tab',
            'query': ' '.join(proteins)
            }
    
        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req) as f:
            response = f.read()
            
        response = response.decode('utf-8')
        
        for r in response:
            protein = r.split()[1]
            if protein_lookup[protein]:
                protein_lookup[protein].append(function)
            else:
                protein_lookup[protein] = [function]
            


    return protein_lookup

def find_all_protein_combos_per_cluster(clusters, exclude_unclustered=True, max_clus_size=100, rand_seed=1710):
    '''For each cluster, find all combos of proteins. 
    Return a dataframe of all possible query and target protein pairs.'''
    np.random.seed(rand_seed)
    
    # Find all combinations of proteins WITHIN clusters
    all_protein_combos_per_cluster = pd.DataFrame()
    
    if exclude_unclustered:
        n = 1
    else: 
        n = 0
        
    # If a cluster has too many items, then just sample. 
    stack_cluster_counts = np.stack(np.unique(clusters.cluster_label, return_counts=True))
        # array([[   -1,     0,     1, ...,  1603,  1604,  1605],
        # [12966,     2,     5, ...,     2,     2,     2]])
    
    big_clusters = stack_cluster_counts[0, stack_cluster_counts[1, :] > max_clus_size]
        # Looks at the counts only.
        # Big cluster at this point can include the noise.

    # Loop through each cluster 
    for clust in sorted(clusters.cluster_label.unique())[n:]:
        # sample if too many items
        if clust in big_clusters:
            cluster_subset = clusters[clusters.cluster_label==clust].sample(max_clus_size)
        else:
            cluster_subset = clusters[clusters.cluster_label==clust]
        # Find all possible combinations of proteins within it 
        clust_combos = pd.DataFrame(itertools.product(cluster_subset.protein, repeat=2),
                                    columns=['query_protein', 'target_protein'])
        # Eliminate pairs of the same protein 
        clust_combos = clust_combos[clust_combos.query_protein != clust_combos.target_protein]
        # Fill its cluster value with the current cluster number 
        clust_combos['cluster'] = clust
        all_protein_combos_per_cluster = all_protein_combos_per_cluster.append(clust_combos)

    return all_protein_combos_per_cluster


def join_blast(clusters, pairwise_metrics, all_protein_combos_per_cluster):
    '''Return clusters with all possible combos and their blast scores.
    pairwise_metrics is the pre-calculated blast file.'''
    
    # if pairwise_metrics wasn't inputted, download the file. 
    if isinstance(pairwise_metrics, type(None)):
        pairwise_metrics = pd.read_csv(io.StringIO(gcs.download_text('annotations/blast_annotations.csv')))
    
    # If all_protein_combos_per_cluster wasn't inputted, calculate from scratch
    if isinstance(all_protein_combos_per_cluster, type(None)):
        all_protein_combos_per_cluster = find_all_protein_combos_per_cluster(clusters)

    # Note: all_protein_combos_per_cluster will only show a subset of proteins in large clusters (up to 200 proteins) 
    clusters_w_blast = all_protein_combos_per_cluster.set_index(['query_protein','target_protein'])\
                         .join(pairwise_metrics.set_index(['query_protein','target_protein']), 
                               on=['query_protein', 'target_protein'], 
                               how='left'
                              ).reset_index()
    
    return clusters_w_blast

def cluster_blast (clusters_w_blast):
    '''Summarize blats stats per cluster '''
    stats_by_cluster = pd.DataFrame(columns=['cluster', 
                                             'bitscore_mean', 'bitscore_std_dev', 
                                             'evalue_mean', 'evalue_std_dev', 'ratio_pairs_wo_blast'])
    
    # Loop through each cluster and calculate cluster-level summary of BLAST
    for clust in clusters_w_blast.cluster.unique():
        # Note: clusters_w_blast only contains up to max sampled proteins if the cluster is bigger than that.
        # This is fine as we're just aggregating the stats at the cluster level. 
        slc = clusters_w_blast[clusters_w_blast.cluster == clust]

        num_combos_in_clust = len(clusters_w_blast[clusters_w_blast.cluster == clust])
        num_null_blast_combos = len(slc[slc.bitscore.isnull()])
        
        # If bitscore and e-value are missing, fill NA with 0 and 10, respectively. 
        slc[slc.cluster==clust].fillna({'bitscore':0,
                                         'evalue':10},
                                        inplace=True)
        
        # The stats by cluster will assume missing values are 0 
        stats_by_cluster.loc[len(stats_by_cluster)] = [clust,  
                                                       slc.bitscore.mean(), slc.bitscore.std(), slc.evalue.mean(), 
                                                       slc.evalue.std(), num_null_blast_combos / num_combos_in_clust]

    return stats_by_cluster


def silhouette_n_davies(X, cluster_labels):
    sil_sc = silhouette_score(X, cluster_labels)
    db_sc = davies_bouldin_score(X, cluster_labels)
    
    return sil_sc, db_sc
    

def dbscan_gridsearch(X, range_eps, range_min_samples, metric='euclidean',
                     X_original=None):
    """For a set of values for `eps` and `range_min_samples`, run grid search in DBSCAN."""
    
    search_results = pd.DataFrame(columns=["eps", "min_samples", "metric", 
                                           "Num. Clusters", "Noise Size", "Max Cluster Size",
                                           "DB_sc", "Silhouette_sc",
                                           "DB_sc excl. noise", "Silhouette_sc excl. noise"])
    
    num_proteins = len(X)
    
    # Even if we use reduced embeddings, do silhouette score based on 
    if isinstance(X_original, type(None)):
        X_v2 = X
    else:
        X_v2 = X_original
    
    # Loop through grid and generate clustering models 
    for i in range_eps:
        for j in range_min_samples:
            print(i, j) 
            
            # Run model 
            clustering = DBSCAN(eps=i, 
                                min_samples=j,
                                metric=metric).fit(X)
            cluster_labels = clustering.labels_
            noise_size = sum(cluster_labels==-1)
            
            # If everything is a noise or there's only one cluster, don't bother calculating scores. 
            if len(np.unique(cluster_labels))<=2:
                sil_sc = db_sc = sil_sc_nonoise = db_sc_nonoise = max_clus_size = None
                
            # Otherwise, calculate scores and save the results. 
            else:
                # Find cluster metrics 
                sil_sc, db_sc = silhouette_n_davies(X_v2, cluster_labels)
                
                # Find cluster metrics after excluding noise (or unclustered)
                sil_sc_nonoise, db_sc_nonoise = silhouette_n_davies(X_v2[cluster_labels!=-1], cluster_labels[cluster_labels!=-1])
                max_clus_size = np.unique(cluster_labels, return_counts=True)[1][1:].max()
                
            # Append results
            search_results = search_results.append({"eps": i, 
                                   "min_samples": j,
                                   "metric": metric, 
                                   "Num. Clusters": len(np.unique(cluster_labels))-1, 
                                   "Noise Size": noise_size, 
                                   "Max Cluster Size": max_clus_size,
                                   "DB_sc": db_sc, 
                                   "Silhouette_sc": sil_sc,
                                   "DB_sc excl. noise": db_sc_nonoise,
                                   "Silhouette_sc excl. noise": sil_sc_nonoise 
                                  }, ignore_index=True)

    return search_results.sort_values(by="Noise Size")



def hdbscan_gridsearch(X, 
                       min_cluster_sizes,
                       min_samples,
                       cluster_selection_epsilons,
                       alphas=[1.0],
                       metrics=['cosine']):
    """grid search in HDBSCAN."""
    
    search_results = pd.DataFrame(columns=["min_cluster_size", "min_sample", "cluster_selection_epsilon", "alpha", "metric",
                                           "Num. Clusters", "Noise Size", "Max Cluster Size",
                                           "DB_sc", "Silhouette_sc",
                                           "DB_sc excl. noise", "Silhouette_sc excl. noise"])
    
    num_proteins = len(X)
    params = list(itertools.product(min_cluster_sizes,
                                   min_samples,
                                   cluster_selection_epsilons,
                                   alphas,
                                   metrics))
    
    # Loop through grid and generate clustering models 
    for param in params:
        min_cluster_size, min_sample, cluster_selection_epsilon, alpha, metric = param

        # Run model 
        clustering = hdbscan.HDBSCAN(algorithm='generic', 
                                     alpha=alpha, 
                                     approx_min_span_tree=True,
                                     gen_min_span_tree=False, 
                                     leaf_size=40, 
                                     metric=metric, 
                                     min_cluster_size=min_cluster_size, 
                                     min_samples=min_sample, 
                                     p=None)
        clustering.fit(X)
        cluster_labels = clustering.labels_
        noise_size = sum(cluster_labels==-1)

        # If everything is a noise or there's only one cluster, don't bother calculating scores. 
        if len(np.unique(cluster_labels))<=2:
            sil_sc = db_sc = sil_sc_nonoise = db_sc_nonoise = None
        # Otherwise, calculate scores and save the results. 
        else:
            # Find cluster metrics 
            sil_sc, db_sc = silhouette_n_davies(X, cluster_labels)

            # Find cluster metrics after excluding noise (or unclustered)
            sil_sc_nonoise, db_sc_nonoise = silhouette_n_davies(X[cluster_labels!=-1], cluster_labels[cluster_labels!=-1])

            
        # Append results
        search_results = search_results.append({"min_cluster_size": min_cluster_size, 
                                                "min_sample": min_sample, 
                                                "cluster_selection_epsilon": cluster_selection_epsilon,
                                                "alpha": alpha,
                                                "metric": metric,
                               "Num. Clusters": len(np.unique(cluster_labels))-1, 
                               "Noise Size": noise_size, 
                               "Max Cluster Size": np.unique(cluster_labels, return_counts=True)[1][1:].max(),
                               "DB_sc": db_sc, 
                               "Silhouette_sc": sil_sc,
                               "DB_sc excl. noise": db_sc_nonoise,
                               "Silhouette_sc excl. noise": sil_sc_nonoise 
                              }, ignore_index=True)

    return search_results.sort_values(by="Noise Size")


def get_tmalign_for_pairs(pairs, get_latest_statsfile=True):
    '''
    Accepts a (2, n) array of protein pairs (full filenames in form query_protein, target_protein). Optional bool to download the current
    version of pairwise_evaluation_metrics.parquet from GCS to pwd (requires "PSS GCS Storage Key.json" in pwd) before getting tmalign scores.
    If False, uses pairwise_evaluation_metrics.parquet from pwd.
    
    Returns a (7, n) pandas dataframe of input protein pairs and their associated TM-Align stats. NaN stats values for pairs not found in 
    pairwise_evaluation_metrics.parquet.
    '''
    if get_latest_statsfile is True:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/jupyter/pss/PSS GCS Storage Key.json"
        storage_client = storage.Client()
        blob = storage_client.get_bucket('capstone-fall21-protein').get_blob('annotations/pairwise_evaluation_metrics.parquet')
        blob.download_to_filename('pairwise_evaluation_metrics.parquet')
    
    stats = pd.read_parquet('pairwise_evaluation_metrics.parquet')
    stats.set_index(['query_protein', 'target_protein'], inplace=True)
    
    return pd.DataFrame(pairs, columns=['query_protein', 'target_protein']).join(stats, on=['query_protein', 'target_protein'], how='left')


def tmalign_hist(metrics, num_results=10, show_control=False, bin_size=100, version='top_score', select_clusters=None, cmap='viridis', name=''):
    '''
    Arguments:
    - metrics: A Pandas DataFrame in the form pairwise_evaluation_metrics.parquet + a "cluster" column.
    
    Optional Arguments:
    - num_results: Default 10. Number of clusters to plot. Ignored if version is set to "select".
    - show_control: Default False. Whether to plot tmalign scores from the random control dataset.
    - bin_size: Default 100. Number of bins on historgram.
    - version: Default "top_score". Options also include "top_pair_count", "random" and "select" (if passed, argument select_clusters is required).
    - select_clusters: A list of integer cluster labels. Required if version is set to "select".
    - cmap: Default "viridis". The name of the Matplotlib colormap to apply to the plot.
    - name: Default "". Name of experiment, ex.: "SEQVEC + HDBSCAN".
    
    Output:
    - A Pyplot histogram to stdout.
    - List of cluster labels shown on plot.
    '''
    subtitle = ''
    figure(figsize=(10, 6))
    pyplot.style.use('default')

    if version == 'top_score':
        interesting_clusters = metrics[['cluster', 'tmalign_score']].groupby(['cluster']).mean(['tmalign_score']).reset_index().sort_values(
        by=['tmalign_score'], ascending=False)[:num_results].cluster.values
        subtitle = f'\nTop {num_results} Clusters by Score'
    elif version == 'top_pair_count':
        interesting_clusters = metrics[['cluster', 'target_protein']].groupby(['cluster']).count().reset_index().sort_values(by=['target_protein'], ascending=False)[:num_results].cluster.values
        subtitle = f'\nTop {num_results} Clusters by Num. of Protein Pairs'
    elif version == 'random':
        interesting_clusters = metrics[['cluster', 'target_protein']].groupby(['cluster']).count().reset_index().sample(num_results).cluster.values
        subtitle = f'\nRandom {num_results} Clusters'
    elif version == 'select':
        interesting_clusters = select_clusters
        subtitle = f'\nSelect {len(interesting_clusters)} Clusters'
    else:
        return None

    ssize = int(len(metrics[metrics.cluster.isin(interesting_clusters)])*0.5)
    colors = pyplot.get_cmap(cmap, len(interesting_clusters))
    
    interesting_clusters.sort()
    
    if name != "":
        name = ' (' + name + ')'
    
    for i, cluster in enumerate(interesting_clusters):
        pyplot.hist(metrics[metrics.cluster == cluster].tmalign_score, bins=bin_size, alpha=0.5, label=f'Cluster {cluster}', range=(0,1), color=colors(i))

    if show_control:
        sample_all = pd.read_parquet('/home/jupyter/pss/tmalign_rmsd_full.parquet')
        pyplot.hist(sample_all.sample(ssize).tm_score_norm_ref_p1.astype(float), bins=bin_size, alpha=0.5, label='Control', range=(0,1))
        pyplot.title('TM-Align Score Distributions: Control vs Experiment' + name + subtitle)
    else:
        pyplot.title('TM-Align Score Distributions' + name + subtitle)

    p = pyplot.ylim()[1] * 0.75
    pyplot.legend(loc='upper left')
    pyplot.xlabel('TM-Align Score')
    pyplot.ylabel('Num. Protein Pairs')
    pyplot.vlines(0.5, 0, pyplot.ylim()[1], linestyles='dashed', colors='k')
    pyplot.text(0.51, p, 'Similarity Significance\nThreshold')
    pyplot.vlines(0.17, 0, pyplot.ylim()[1], linestyles='dashed', colors='k')
    pyplot.text(0.18, p, 'Random Noise')
    pyplot.show();
    
    return interesting_clusters


def tmalign_scatter(metrics, num_results=10, version='top_score_tmalign', select_clusters=None, cmap='viridis', name=""):
    '''
    Arguments:
    - metrics: A Pandas DataFrame in the form pairwise_evaluation_metrics.parquet + a "cluster" column.
    
    Optional Arguments:
    - num_results: Default 10. Number of clusters to plot. Ignored if version is set to "select".
    - version: Default "top_score_tmalign". Options also include "top_score_rmsd", "top_pair_count", "random" and "select" (if passed, argument select_clusters is required).
               Note: Only the equivalent of num_results = len(metrics.cluster.unique()) is currently implemented.
    - select_clusters: A list of integer cluster labels. Required if version is set to "select".
    - cmap: Default "viridis". The name of the Matplotlib colormap to apply to the plot.
    - name: Default "". Name of experiment, ex.: "SEQVEC + HDBSCAN".
    
    Output:
    - A Pyplot scatter plot to stdout.
    - List of cluster labels shown on plot.
    '''
    subtitle = ''
    figure(figsize=(10, 6))
    pyplot.style.use('default')
    
    subtitle = f'\nSelect {len(metrics.cluster.unique())} Clusters'
    focus = metrics.cluster.unique()
    focus.sort()
    colors = pyplot.get_cmap(cmap, len(focus))
    #colors = list(mcolors.cnames.keys())[::-1]
    
    if name != "":
        name = ' (' + name + ')'
    
    for i, cluster in enumerate(focus):
        t = metrics[metrics.cluster == cluster]
        pyplot.scatter(x=t.tmalign_score, y=t.rmsd, alpha=0.5, label=f'Cluster {cluster}', color=colors(i))

    pyplot.legend(loc='upper left')
    pyplot.xlim(left=0, right=1)
    pyplot.ylim(bottom=0, top=10)
    pyplot.vlines(0.5, 0, 3, linestyles='dashed', colors='k')
    pyplot.hlines(3, 0.5, 1, linestyles='dashed', colors='k')
    pyplot.text(0.5, 3.2, 'Strong Similarity Zone')
    pyplot.xlabel('TM-Align Score')
    pyplot.ylabel('RMSD')
    pyplot.title('TM-Align Score vs RMSD' + name + subtitle)
    pyplot.show();
    
    return focus


def model_overview(model, X):
    """
    Given a clustering model and the original embeddings, generate stats related to the model.
    """
    
    labels_all = model.labels_
    labels = np.unique(model.labels_, return_counts=True)
    noise_ct = labels[1][0]
    max_cluster_size = labels[1][1:].max()
    num_proteins=model.labels_.shape[0]
    sil_sc_nonoise, db_sc_nonoise = silhouette_n_davies(X[labels_all!=-1], labels_all[labels_all!=-1])


    result= {"Model": str(model),
            "Number of clusters categories (incl. noise)": np.unique(model.labels_, return_counts=True)[1].shape[0],
            "Number of clusters (excl. noise)": np.unique(model.labels_, return_counts=True)[1].shape[0]-1,
            "Noise": noise_ct,
            "Largest non-noise cluster": max_cluster_size,
            "Noise as % of total":  noise_ct/num_proteins,
            "Noise and largest cluster as % of total": (noise_ct+max_cluster_size)/num_proteins,
            "Silhouette score": sil_sc_nonoise,
            "DB score": db_sc_nonoise
           }
    
    try: 
        result["Length of embedding"] = len(model.weighted_cluster_centroid(0))
    except:
        pass
    
    return result 
            
def map_gomf_to_cluster(clusters, alphafold_protein_to_gomf):
    """
    Given clusters and protein-to-GOMF mapping, 
    Show all proteins in clusters with GOMF and parent GOMF. 
    Left join to clusters rather than inner or outer join, because 
    for some embeddings (e.g., DeepFold), not all the proteins will have an
    embedding generated and therefore will not have been in the clustering model.
    """
                
    clusters_with_gomf = clusters.merge(alphafold_protein_to_gomf,
                   how='left',   # In case clusters excludes the proteins that did not have any embeddings
                   left_on='protein',
                   right_on='protein_id'
                  )

    return clusters_with_gomf

def map_uniprot_data(cluster_data, left_on='protein'):
    """
    Given clusters, map in the information available on UniProtKB. 
    
    ABOUT 'structure_files/uniprot-organism Homo+sapiens+(Human)+9606-AlphaFoldFiltered.parquet'
        Downloaded [UniProtKB 2021_03 results (Filtered by Human)]
        (https://www.uniprot.org/uniprot/?query=organism%3A%22Homo+sapiens+%28Human%29+%5B9606%5D%22&sort=id&desc=no)
        on Nov 14, 2021. Filtered by the overlapping proteins with AlphaFold. 
        Added an indicator for whether a 3d structure exists. If there is a structure information,
        that means uniprot had a 3D structure other than those predicted by AlphaFold.

    `has_3d` marks whether there is a 3D structure data outside of AlphaFold2, which would be one 
    of ['X-ray crystallography',  'NMR spectroscopy', 'Electron microscopy', 'Model', 'Infrared spectroscopy']
    
    """
    uni = gcs.download_parquet('structure_files/uniprot-organism Homo+sapiens+(Human)+9606-AlphaFoldFiltered.parquet')
    cluster_w_uniprot = cluster_data.merge(uni,
                      how='left',
                      left_on=left_on,
                      right_on='Entry'
                     )    
    
    cluster_w_uniprot["has_3d"].fillna(False, inplace=True)
    return cluster_w_uniprot
    

def gen_cluster_uniprot_stats(cluster_data):
    """
    Given clusters, map in the information available on UniProtKB. 
    
    `has_3d` marks whether there is a 3D structure data outside of AlphaFold2, which would be one 
    of ['X-ray crystallography',  'NMR spectroscopy', 'Electron microscopy', 'Model', 'Infrared spectroscopy']
    
    Summarizes the results by cluster.
    
    """

    cluster_w_uniprot = map_uniprot_data(cluster_data)
    
    uniprot_stats= cluster_w_uniprot.pivot_table(index='cluster_label',
                              values='has_3d',
                              aggfunc=[len, sum, np.mean]
                             ).reset_index()
    uniprot_stats.columns=['cluster_label', 'num_proteins', 'num_proteins_has_3d', 'perc_proteins_has_3d']
    
    return uniprot_stats

class funsim_evaluator():
    def __init__(self, all_protein_combos_per_cluster, goa=None):
        self.all_protein_combos_per_cluster = all_protein_combos_per_cluster
        self.goa = goa 
        
        self.go_term_names = pd.read_csv(io.StringIO(
            gcs.download_text('functional_sim/data/yeastmine_results_goa_goid_mf.tsv')),
            sep='\t'
            )
        self.go_term_names.columns = [col[10:] for col in self.go_term_names.columns]
        self.go_term_names_dict = self.go_term_names.set_index("Identifier").to_dict()
        
        # If goa df is not provided, download from GCS 
        if isinstance(self.goa, type(None)):
            print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "No GO annotations provided. Downloading from google cloud.")
            self.goa = pd.read_csv( io.BytesIO(gcs.download_blob("functional_sim/data/goa_human.gaf.gz")), 
                                compression='gzip', 
                                header=None,
                                skiprows=41,    # hard-coded. May be different for other gaf files.
                                sep='\t')
            self.goa.columns=["DB", "DB Object ID", "DB Object Symbol", "Qualifier", "GO ID", "Reference", 
                         "Evidence Code", "With or From", "Aspect", "Name", "Synonym", "Type", 
                         "Taxon", "Date", "Assigned By", "Annotation Extension", "Gene Product Form ID"]


        ##################
        # Calculate IC (information content) of each term

        # Identify molecular functions in GO
        self.shortest_from_root = gcs.download_pkl('functional_sim/shortest_from_root.pkl')
        self.goa_goid_mf = [goid for goid in set(self.goa["GO ID"]) if goid in self.shortest_from_root['GO:0003674'] ]

        # IC calculation 
        self.M = self.goa[self.goa['GO ID'].isin(self.goa_goid_mf)].pivot_table(index='GO ID',
                    values='DB Object ID',
                    aggfunc=pd.Series.nunique
                   ).to_dict()['DB Object ID']

        self.N = len(self.goa[self.goa['GO ID'].isin(self.goa_goid_mf)]['DB Object ID'].unique())
        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Total number of proteins in GO annotations:", self.N)

        self.IC_t = {t: -np.log(m/self.N) for t, m in self.M.items()}
        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "IC_t created")

        ##################

        # Lookup dictionary of proteins and their GO terms 
        self.goa_by_protein = self.goa[self.goa['GO ID'].isin(self.goa_goid_mf)].pivot_table(
                                index=["DB Object ID"],
                                values=["GO ID"],
                                aggfunc=lambda x:set(x)
                            ).to_dict()['GO ID']
        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Dictionary of proteins and their GO terms lookup created")
        

        ##################
        # Eliminate duplicates in the pairs of proteins from cluster output, 
        # since the jaccard pairwise metric is symmetrical.

        self.all_protein_combos_per_cluster['protein_A'] = self.all_protein_combos_per_cluster[
            ['query_protein','target_protein']].min(axis=1)

        self.all_protein_combos_per_cluster['protein_B'] = self.all_protein_combos_per_cluster[
            ['query_protein','target_protein']].max(axis=1)

        self.protein_pair_funsim = self.all_protein_combos_per_cluster[
            ['protein_A', 'protein_B', 'cluster']].drop_duplicates()



    def funsim(self):
        """
        Find functional similarities for all protein pairs in each cluster 

        Inputs:
            - self.all_protein_combos_per_cluster: possible protein pair combinations per cluster 
            - goa: Gene ontology annotation file that maps proteins to gene ontology 
        """


        ##################
        # Find Jaccard sim

        self.protein_pair_funsim['funsim'] = \
            self.protein_pair_funsim.apply(
                lambda x: self.jaccard_sim_protein_go(x['protein_A'], x['protein_B']), 
                axis=1
            )
        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Funsim calculated.")

        # Pivot by cluster 
        self.cluster_funsim = self.protein_pair_funsim.pivot_table(
            index="cluster",
            values="funsim",
            aggfunc=[len, "count", np.mean]
        )
        self.cluster_funsim.columns = ["num_pairs", "num_pairs_with_funsim", "funsim"]
        self.cluster_funsim["perc_pairs_w_funsim"] = self.cluster_funsim.num_pairs_with_funsim/self.cluster_funsim.num_pairs

        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Funsim summary by cluster done.")

        #################
        # Identify top common GO terms per cluster 
        cluster_common_go = self.common_go_in_cluster()
        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Common GO term sumary per cluster processed.")
        
        self.cluster_funsim = self.cluster_funsim.merge(cluster_common_go,
                                                       left_index=True,
                                                        right_index=True)
        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Merged cluster-level funsim score with GO summary.")


#         return self.cluster_funsim, self.protein_pair_funsim



    def jaccard_sim_protein_go(self, protein_A, protein_B):
        """Calculate the GIC or the Jaccard index of terms between two proteins.
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S5-S4

        - self.goa_by_protein: Dictionary where key is protein ID and value is list of GO annotation terms for that protein
        - self.IC_t: Dictionary where key is GO term and value is its information content (IC)

        """
        if protein_A not in self.goa_by_protein or protein_B not in self.goa_by_protein:
            return None

        go_intersection = self.goa_by_protein[protein_A].intersection(self.goa_by_protein[protein_B])
        go_union        = self.goa_by_protein[protein_A].union(       self.goa_by_protein[protein_B])

        numerator = 0
        denominator = 0

        for goid in go_intersection:
            numerator += self.IC_t[goid]

        denominator = numerator
        for goid in go_union - go_intersection:
            denominator += self.IC_t[goid]

        return numerator/denominator

    
    def common_go_in_cluster(self):
        """ 
        For each cluster, returns the list of GO terms that are associated with all proteins
        in that cluster and provides a summary statistic. 
        """

        self.unique_proteins = self.all_protein_combos_per_cluster[["query_protein", "cluster"]].drop_duplicates()

        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Get NP Arr of GO terms for each protein")
        self.unique_proteins["go"] = self.unique_proteins["query_protein"].apply(self.get_nparr_of_go_terms)

        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Turn GO terms into dict")
        cluster_info = self.unique_proteins.pivot_table(
                                    index='cluster',
                                    values='go',
                                    aggfunc=self.make_go_ct_dict
                                )

        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Map GO desc...")
        cluster_info["go_summary"] = cluster_info["go"].map(self.map_go_desc)
        print(datetime.now().strftime("%Y-%b-%d %H:%M:%S"), "Mapping GO desc done.")

        return cluster_info.reset_index()
    
    def get_nparr_of_go_terms(self, protein_id):
        '''
        Looks up protein_id from self.goa_by_protein and returns the list of GO terms as a numpy array.
        '''
        try:
            return np.array(list(self.goa_by_protein[protein_id]) )
        except:
            return np.array([])

    def make_go_ct_dict(self, go_terms):
        '''
        Given a list of GO terms, stack them all together and return a dictionary
        where the key is GO values and the value is their counts.
        '''
        go_ct = np.vstack(np.unique(np.hstack(go_terms), return_counts = True))
        pairs = list(zip(go_ct[0], go_ct[1]))
        go_ct_dict = {go: int(ct) for go, ct in pairs}

        return go_ct_dict

    def map_go_desc(self, go_ct_dict):
        '''
        Given a dictionary containing just go_id and count of proteins, 
        pull in GO name and description as well. Return a dictionary where
        the key is GO ID and the value is a dictionary containing 
        num. proteins, GO name, and GO desc. 
        '''

        new_go_ct_dict = {}
        temp_dict={}
        for go_identifier, ct_protein in go_ct_dict.items():
            new_go_ct_dict[go_identifier]= {}
            new_go_ct_dict[go_identifier]["Num. Protein"] = ct_protein
            new_go_ct_dict[go_identifier]["Name"] = self.go_term_names_dict["Name"][go_identifier]
            new_go_ct_dict[go_identifier]["Description"] = self.go_term_names_dict["Description"][go_identifier]

        return {k: v for k, v in
            sorted(new_go_ct_dict.items(), key=lambda item: item[1]["Num. Protein"], reverse=True)}
    
    def get_go_summary_df(self, clusterno):
        return pd.DataFrame.from_dict(self.cluster_funsim.loc[clusterno]["go_summary"],
                               orient='index')
