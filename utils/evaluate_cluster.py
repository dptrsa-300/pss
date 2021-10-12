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

# Ask how to work with this one 
sys.path.append(f"{os.path.dirname(__file__)}/")

import gcs_utils as gcs

import urllib.parse
import urllib.request

    
def merge_cluster_stats(stats_1, stats_2):
    """If there are two dataframes with cluster_label and corresponding stats,
    merge the two into one df."""
    
    return stats_1.join(stats_2)



def sequence_stats(clusters,
                  seq_parquet="structure_files/sequences/sequences.parquet"):
    """Get amino acid sequence-related stats for each cluster.
    The input `clusters` is a pd.df that has two columns: 'protein' and 'cluster_label' """
    
    # Download sequence stats
    sequences = gcs.download_parquet(seq_parquet)
    sequences["seq_len"] = sequences["pdbx_seq_one_letter_code"].str.len()
    
    # For each protein in sequence, map in the cluster label
    sequences = clusters.set_index(["protein"])\
            .join(sequences.set_index(["protein_id"]),
                  how='outer'
                 ).reset_index()[['protein', 'cluster_label', 'db_code', 'db_name','pdbx_seq_one_letter_code', 'protein_filename', 'seq_len']]

    sequences=sequences.rename(columns={'index': "protein"})
    
    # Summarize stats by cluster. For each cluster, show min, max, avg etc of seq length. 
    cluster_stats = pd.pivot_table(sequences, values="seq_len",
                             index="cluster_label",
                             aggfunc={"seq_len": [len, np.mean, np.std, np.min, np.median, np.max,
                                                 lambda x: list(x)]}).reset_index()
    
    cluster_stats = cluster_stats.rename(columns={'<lambda_0>': "seq_len_arr", 
                              'amax': "max_seq_len", 
                              'amin': "min_seq_len", 
                              'mean': "mean_seq_len", 
                              'median': "median_seq_len", 
                              'len': "count",
                              'std': "std_seq_len"})[['cluster_label', "count", 'mean_seq_len', 'min_seq_len',  'median_seq_len', 'max_seq_len', 
       'std_seq_len','seq_len_arr']]
    
    return cluster_stats 



def protein_confidence_agg(clusters, 
                           n=None,
                           prefix='structure_files/atom_sites'):
    """Downloads protein files, then summarizes amino acid sequence-level confidence information."""
    
    keys = gcs.list_file_paths(prefix)
    if not n:
        n = len(keys)

    # Download, dedupe, and add 
    asp = pd.DataFrame(columns=["protein_id", "label_seq_id", "pdbx_sifts_xref_db_res", "confidence_pLDDT"])

    for key in keys[:n]:
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

def find_all_protein_combos_per_cluster(clusters, exclude_unclustered=True):
    '''For each cluster, find all combos of proteins. 
    Return a dataframe of all possible query and target protein pairs.'''
    
    # Find all combinations of proteins WITHIN clusters
    all_protein_combos_per_cluster = pd.DataFrame()
    
    if exclude_unclustered:
        n = 1
    else: 
        n = 0

    # Loop through each cluster 
    for clust in sorted(clusters.cluster_label.unique()[n:]):
        # Find all possible combinations of proteins within it 
        clust_combos = pd.DataFrame(itertools.product(clusters[clusters.cluster_label == clust].protein, repeat=2),
                                    columns=['query_protein', 'target_protein'])
        # Eliminate pairs of the same protein 
        clust_combos = clust_combos[clust_combos.query_protein != clust_combos.target_protein]
        # Fill its cluster value with the current cluster number 
        clust_combos['cluster'] = clust
        all_protein_combos_per_cluster = all_protein_combos_per_cluster.append(clust_combos)

    return all_protein_combos_per_cluster


def join_blast(clusters, pairwise_metrics=None):
    '''Return clusters with all possible combos and their blast scores.
    pairwise_metrics is the pre-calculated blast file.'''
    
    if not pairwise_metrics:
        pairwise_metrics = pd.read_csv(io.StringIO(gcs.download_text('annotations/blast_annotations.csv')))
    
    all_protein_combos_per_cluster = find_all_protein_combos_per_cluster(clusters)
     
    clusters_w_blast = all_protein_combos_per_cluster.set_index(['query_protein','target_protein'])\
                         .join(pairwise_metrics.set_index(['query_protein','target_protein']), 
                               on=['query_protein', 'target_protein'], 
                               how='left'
                              ).reset_index()

    return clusters_w_blast

def cluster_blast (embedding_name, 
                   model_name, 
                   clusters_w_blast):
    '''Summarize blats stats per cluster '''
    stats_by_cluster = pd.DataFrame(columns=['embedding', 'model', 'cluster', 
                                             'bitscore_mean', 'bitscore_std_dev', 
                                             'evalue_mean', 'evalue_std_dev', 'ratio_pairs_wo_blast'])
    
    # print('cluster / len left join / nulls from join / all combos / ratio of nulls')
    for clust in clusters_w_blast.cluster.unique():
        slc = clusters_w_blast[clusters_w_blast.cluster == clust]

        num_combos_in_clust = len(clusters_w_blast[clusters_w_blast.cluster == clust])
        num_null_blast_combos = len(slc[slc.bitscore.isnull()])

        stats_by_cluster.loc[len(stats_by_cluster)] = [embedding_name, model_name, clust,  
                                                       slc.bitscore.mean(), slc.bitscore.std(), slc.evalue.mean(), 
                                                       slc.evalue.std(), num_null_blast_combos / num_combos_in_clust]
        
    return stats_by_cluster
