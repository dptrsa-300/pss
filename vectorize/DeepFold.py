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
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import utils.gcs_utils as gcs
import utils.proteins as pr
from vectorize.DeepFold.scripts.network import DeepFold

from distance_matrix import get_distance_matrix_from_structure, get_distance_matrix_from_positions
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

def vectorize_protein_from_file(files, open_method):
    # structures = []
    parser = PDBParser()
    embeddings = []
    for file in files:
        with open_method(file) as f:
            struct = parser.get_structure('structure', f).get_list()[0]
        #     structures.append(struct)
        # structure = combine_structures(structures)
        distance_matrix = get_distance_matrix_from_structure(struct).astype("float32")
        model = DeepFold(max_length=distance_matrix.shape[0], projection_level=1)
        model.load_from_file(model_file)
        emb = model.get_embedding(distance_matrix)
        embeddings.append(emb)
    embeddings = np.mean(np.vstack(embeddings), axis=0)
    return embeddings


def filter_residue_positions(atoms, mask_by_shape=False):
    atoms_filtered = atoms[
        atoms.label_atom_id == "CA"]  # proxy for residue position, see get_residue_positions from original DeepFold code
    if mask_by_shape:
        atoms_filtered = atoms_filtered[~atoms_filtered["shape.conf_type_id"].isnull()]
    residue_positions = atoms_filtered[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype("float32")
    return residue_positions


def vectorize_protein_from_atom_locations(protein_id, residue_positions):
    try:
        embeddings = []
        distance_matrix = get_distance_matrix_from_positions(residue_positions).astype("float32")
        model = DeepFold(max_length=distance_matrix.shape[0], projection_level=1)
        model.load_from_file(model_file)
        emb = model.get_embedding(distance_matrix)
        embeddings.append(emb)
        vector = np.mean(np.vstack(embeddings), axis=0)
    except ValueError as e:
        print(f"Vectorizing with DeepFold failed for {protein_id}, {e}")
        vector = None
    return protein_id, vector


def vectorize_protein_file_from_local(protein, files):
    open_fn = lambda f: gzip.open(f, 'rt')
    try:
        vector = vectorize_protein_from_file(files, open_fn)
    except ValueError as e:
        print(f"Vectorizing with DeepFold failed for {protein}, {e}")
        vector = None
    return protein, vector


def vectorize_protein_file_from_gs(protein, files):
    open_fn = lambda f: gcs.download_gzip(f)
    try:
        vector = vectorize_protein_from_file(files, open_fn)
    except ValueError as e:
        print(f"Vectorizing with DeepFold failed for {protein}, {e}")
        vector = None
    return protein, vector


def apply_vector_fn(protein_and_files, fn, **kwargs):
    return fn(protein_and_files[0], protein_and_files[1], **kwargs)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create embeddings for a protein structure. Output a numpy embedding.')
    parser.add_argument('--pdb-file-directory', metavar='pdb-file-directory', help='directory containing pdb gzip files')
    parser.add_argument('--output-file', metavar='output-file', help='an output numpy embedding')
    parser.add_argument('--file-format', choices=['pdb', 'parquet'], default='pdb',
                        help="specify whether files in directory are pdb or already parsed into parquet format")
    parser.add_argument('--mask-by-shape', action='store_true', help='whether or not to mask more uncertain/floppy parts of protein structure')
    args = parser.parse_args()

    print(f"vectorizing files from {args.pdb_file_directory}")

    if args.pdb_file_directory.startswith("gs:/"):
        bucket, prefix = gcs.uri_to_bucket_and_key(args.pdb_file_directory)
        files = gcs.list_keys(prefix)
        fn = vectorize_protein_file_from_gs
    elif args.file_format == "pdb":
        local_dir = args.pdb_file_directory # "/Users/skyler.roh/Downloads/UP000005640_9606_HUMAN"
        files = [f"{local_dir}/{f}" for f in os.listdir(local_dir) if f.endswith(".pdb.gz")]
        fn = vectorize_protein_file_from_local
    elif args.file_format == "parquet":
        local_dir = args.pdb_file_directory
        files = [f"{local_dir}/{f}" for f in os.listdir(local_dir) if f.endswith(".parquet")]
        fn = vectorize_protein_from_atom_locations

    if args.file_format == "pdb":
        grouped_files = defaultdict(list)
        for f in files:
            protein = pr.get_protein_id_from_filename(f)
            grouped_files[protein].append(f)

        sorted_and_grouped_files = \
            [(k, sorted(v, key=lambda x: pr.protein_file_number(pr.get_stripped_protein_filename(x))))
             for k, v in grouped_files.items()]

        for i in tqdm(range(len(sorted_and_grouped_files) // 1000 + 1)):
            print(f"****{i*1000} to {(i+1)*1000}****")
            embeddings = process_map(partial(apply_vector_fn, fn=fn), sorted_and_grouped_files[i*1000:(i+1)*1000], max_workers=8)
            pd.DataFrame(embeddings, columns=["protein_id", "deepfold"]).to_csv(f"{args.output_file}_{i}.csv", index=False)

    elif args.file_format == "parquet":
        for f in tqdm(files[:1]):
            atoms = pd.read_parquet(f)
            atoms_by_protein = [(p, filter_residue_positions(a, args.mask_by_shape)) for p, a in atoms.groupby("protein_id")]
            embeddings = process_map(partial(apply_vector_fn, fn=fn, mask=args.mask),
                                     atoms_by_protein, max_workers=8)
            pd.DataFrame(embeddings, columns=["protein_id", "deepfold"]).to_csv(f"{args.output_file}_{i}.csv",
                                                                                index=False)
