import argparse
import gzip
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd
from tqdm.auto import tqdm
sys.path.append(f"{os.path.dirname(__file__)}/..")
sys.path.append(f"{os.path.dirname(__file__)}/DeepFold/scripts/")

import utils.gcs_utils as gcs
from network import DeepFold

from distance_matrix import get_distance_matrix_from_structure
from Bio.PDB import PDBParser

model_file = os.path.join(os.path.dirname(__file__), './DeepFold/models/deepfold.model')

def vectorize_protein(file_like_object):
	try:
		parser = PDBParser()
		structure = parser.get_structure('structure', file_like_object).get_list()[0]
		distance_matrix = get_distance_matrix_from_structure(structure).astype("float32")
		model = DeepFold(max_length=distance_matrix.shape[0], projection_level=1)
		model.load_from_file(model_file)
		embedding = model.get_embedding(distance_matrix)
		return embedding
	except ValueError as e:
		print(f"Vectorizing with DeepFold failed for {file_like_object.name}, {e}")
		return None


def get_protein_filename(path):
	return path.strip("pdb.gz").split("/")[-1]


def vectorize_protein_file_from_local(pdb_file):
	with gzip.open(pdb_file, 'rt') as f:
		return get_protein_filename(pdb_file), vectorize_protein(f)


def vectorize_protein_file_from_gs(gs_key):
	with gcs.download_gzip(gs_key) as f:
		return get_protein_filename(gs_key), vectorize_protein(f)


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Create embeddings for a protein structure. Output a numpy embedding.')
	parser.add_argument('--pdb-file-directory', metavar='pdb-file-directory', help='directory containing pdb gzip files')
	parser.add_argument('--output-file', metavar='output-file', help='an output numpy embedding')
	args = parser.parse_args()

	print(f"vectorizing files from {args.pdb_file_directory}")

	if args.pdb_file_directory.startswith("gs:/"):
		bucket, prefix = gcs.uri_to_bucket_and_key(args.pdb_file_directory)
		files = sorted(gcs.list_keys(prefix))
		fn = vectorize_protein_file_from_gs
	else:
		local_dir = args.pdb_file_directory # "/Users/skyler.roh/Downloads/UP000005640_9606_HUMAN"
		files = sorted([f"{local_dir}/{f}" for f in os.listdir(local_dir) if f.endswith(".pdb.gz")])
		fn = vectorize_protein_file_from_local

	for i in tqdm(range(len(files) // 1000 + 1)):
		print(f"****{i*1000} to {(i+1)*1000}****")
		start = time.time()
		with Pool() as p:
			embeddings = p.map(fn, files[i*1000:(i+1)*1000])
		print(f"took {int(time.time() - start)} seconds")
		pd.DataFrame(embeddings).to_csv(f"{args.output_file}_{i}.csv")