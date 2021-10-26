from google.cloud import storage

import argparse
import gzip
import os
import sys
import time
import string
import io
import itertools

import numpy as np
import pandas as pd

#from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score

from google.cloud import storage
from google.cloud.storage import Blob

# Ask how to work with this one 
sys.path.append(f"{os.path.dirname(__file__)}/")

#import gcs_utils as gcs

import urllib.parse
import urllib.request

import matplotlib
from matplotlib.pyplot import figure
from matplotlib import pyplot

    
def deepfold_file_processor(key):
    """Download and parse a DeepFold embedding file. """
    
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

    for key in keys[1:]:
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
    
def protein_confidence_agg(clusters,
                           asp):
   
    
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

def find_all_protein_combos_per_cluster(clusters, exclude_unclustered=True, max_clus_size=100):
    '''For each cluster, find all combos of proteins. 
    Return a dataframe of all possible query and target protein pairs.'''
    
    # Find all combinations of proteins WITHIN clusters
    all_protein_combos_per_cluster = pd.DataFrame()
    
    if exclude_unclustered:
        n = 1
    else: 
        n = 0
        
    # If a cluster has too many items, then just sample. 
    stack_cluster_counts = np.stack(np.unique(clusters.cluster_label, return_counts=True))
    big_clusters = stack_cluster_counts[0, stack_cluster_counts[1, :] > max_clus_size]

    # Loop through each cluster 
    for clust in sorted(clusters.cluster_label.unique()[n:]):
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
    

def dbscan_gridsearch(X, range_eps, range_min_samples, metric='euclidean'):
    """For a set of values for `eps` and `range_min_samples`, run grid search in DBSCAN."""
    
    search_results = pd.DataFrame(columns=["eps", "min_samples", "metric", 
                                           "Num. Clusters", "Noise Size", "Max Cluster Size",
                                           "DB_sc", "Silhouette_sc",
                                           "DB_sc excl. noise", "Silhouette_sc excl. noise"])
    
    num_proteins = len(X)
    
    
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
                sil_sc, db_sc, sil_sc_nonoise, db_sc_nonoise = None
                
            # Otherwise, calculate scores and save the results. 
            else:
                # Find cluster metrics 
                sil_sc, db_sc = silhouette_n_davies(X, cluster_labels)
                
                # Find cluster metrics after excluding noise (or unclustered)
                sil_sc_nonoise, db_sc_nonoise = silhouette_n_davies(X[cluster_labels!=-1], cluster_labels[cluster_labels!=-1])
                
            # Append results
            search_results = search_results.append({"eps": i, 
                                   "min_samples": j,
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
            sil_sc, db_sc, sil_sc_nonoise, db_sc_nonoise = None
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