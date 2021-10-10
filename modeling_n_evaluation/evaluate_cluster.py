from google.cloud import storage

import argparse
import gzip
import os
import sys
import time
import string

from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Ask how to work with this one 
sys.path.append(f"{os.path.dirname(__file__)}/modeling_n_evaluation/")

from utils import gcs_utils as gcs

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import utils.gcs_utils as gcs


def sequence_stats(clusters):
    """Get amino acid sequence-related stats for each cluster."""
    # Download sequence stats
    sequences = gcs.download_parquet("structure_files/sequences/sequences.parquet")
    sequences["seq_len"] = sequences["pdbx_seq_one_letter_code"].str.len()
    
    # For each protein in sequence, map in the cluster label
    sequences = clusters.set_index(["protein"])\
            .join(sequences.set_index(["protein_id"]),
                  how='outer'
                 ).reset_index()[['index', 'cluster_label', 'db_code', 'db_name','pdbx_seq_one_letter_code', 'protein_filename', 'seq_len']]

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

def protein_confidence_agg(n=None):
    """Downloads protein files, then summarizes amino acid sequence-level confidence information."""
    
    prefix = 'structure_files/atom_sites'
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate cluster outputs.')
    parser.add_argument('--model-output', metavar='model-output', help='Cluster model outputs')
    parser.add_argument('--eval-results', metavar='eval-results', help='Evaluation results')
    args = parser.parse_args()
    
    # Grab model results
    clusters = 
    
    # Sequence stats per cluster
    cluster_seq_stats = sequence_stats(clusters)
    
    # Get confidence level stats per protein
    cluster_conf = protein_confidence_agg(clusters)
    
    