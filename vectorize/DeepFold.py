import argparse
import gzip
import os
import sys
import time
import string
from multiprocessing import Pool
from collections import defaultdict
from functools import partial

sys.path.append(f"{os.path.dirname(__file__)}/DeepFold/scripts/")
sys.path.append(f"{os.path.dirname(__file__)}/..")

import pandas as pd
from tqdm.auto import tqdm

import utils.gcs_utils as gcs
import utils.proteins as pr
from vectorize.DeepFold.scripts.network import DeepFold

from distance_matrix import get_distance_matrix_from_structure
from Bio.PDB import PDBParser

model_file = os.path.join(os.path.dirname(__file__), './DeepFold/models/deepfold.model')


def combine_structures(list_of_structures):
    """https://lists.open-bio.org/pipermail/biopython/2014-June/015345.html"""
    main_structure = list_of_structures[0]
    # Set chains in structures and move to first structure
    index = 0
    for i, structure in enumerate(list_of_structures):
        for chain in structure:
            chain.id = string.ascii_uppercase[index]
            index += 1
            # Don't move chains of first structure
            if i != 0:
                chain.detach_parent()
                main_structure.add(chain)
    return main_structure

def vectorize_protein(files, open_method):
    structures = []
    parser = PDBParser()
    for file in files:
        with open_method(file) as f:
            struct = parser.get_structure('structure', f).get_list()[0]
            structures.append(struct)
            
    structure = combine_structures(structures)
    distance_matrix = get_distance_matrix_from_structure(structure).astype("float32")
    model = DeepFold(max_length=distance_matrix.shape[0], projection_level=1)
    model.load_from_file(model_file)
    embedding = model.get_embedding(distance_matrix)
    return embedding


def vectorize_protein_file_from_local(protein, files):
    open_fn = lambda f: gzip.open(f, 'rt')
    try:
        vector = vectorize_protein(files, open_fn)
    except ValueError as e:
        print(f"Vectorizing with DeepFold failed for {protein}, {e}")
        vector = None
    return protein, vector


def vectorize_protein_file_from_gs(protein, files):
    open_fn = lambda f: gcs.download_gzip(f)
    try:
        vector = vectorize_protein(files, open_fn)
    except ValueError as e:
        print(f"Vectorizing with DeepFold failed for {protein}, {e}")
        vector = None
    return protein, vector


def apply_vector_fn(protein_and_files, fn):
    return fn(protein_and_files[0], protein_and_files[1])


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create embeddings for a protein structure. Output a numpy embedding.')
    parser.add_argument('--pdb-file-directory', metavar='pdb-file-directory', help='directory containing pdb gzip files')
    parser.add_argument('--output-file', metavar='output-file', help='an output numpy embedding')
    args = parser.parse_args()

    print(f"vectorizing files from {args.pdb_file_directory}")

    if args.pdb_file_directory.startswith("gs:/"):
        bucket, prefix = gcs.uri_to_bucket_and_key(args.pdb_file_directory)
        files = gcs.list_keys(prefix)
        fn = vectorize_protein_file_from_gs
    else:
        local_dir = args.pdb_file_directory # "/Users/skyler.roh/Downloads/UP000005640_9606_HUMAN"
        files = [f"{local_dir}/{f}" for f in os.listdir(local_dir) if f.endswith(".pdb.gz")]
        fn = vectorize_protein_file_from_local

    grouped_files = defaultdict(list)
    for f in files:
        protein = pr.get_protein_id_from_filename(f)
        grouped_files[protein].append(f)

    sorted_and_grouped_files = \
        [(k, sorted(v, key=lambda x: pr.protein_file_number(pr.get_stripped_protein_filename(x)))) 
         for k, v in grouped_files.items()]
    
    for i in tqdm(range(len(sorted_and_grouped_files) // 1000 + 1)):
        print(f"****{i*1000} to {(i+1)*1000}****")
        start = time.time()
        with Pool() as p:
            embeddings = p.map(partial(apply_vector_fn, fn=fn), sorted_and_grouped_files[i*1000:(i+1)*1000])
        print(f"took {int(time.time() - start)} seconds")
        pd.DataFrame(embeddings, columns=["protein_id", "deepfold"]).to_csv(f"{args.output_file}_{i}.csv", index=False)