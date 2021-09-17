import os
import sys
sys.path.append(f"{os.path.dirname(__file__)}/..")
sys.path.append(f"{os.path.dirname(__file__)}/DeepFold/scripts/")

import gzip
import utils.spark_utils
import utils.gcs_utils as gcs
from network import DeepFold
from distance_matrix import get_distance_matrix_from_structure
import Bio
from Bio.PDB import PDBParser

model_file = os.path.join(os.path.dirname(__file__), './DeepFold/models/deepfold.model')
# pdb_file = os.path.join(os.path.dirname(__file__), './../AF-A0A0A0MRZ7-F1-model_v1.pdb')
pdb_file = "gs://capstone-fall21-protein/UP000005640_9606_HUMAN/pdb/AF-A0A0A0MRZ7-F1-model_v1.pdb.gz"

def vectorize_protein(file_like_object):
	parser = PDBParser()
	structure = parser.get_structure('structure', file_like_object).get_list()[0]
	distance_matrix = get_distance_matrix_from_structure(structure).astype("float32")
	model = DeepFold(max_length=distance_matrix.shape[0], projection_level=1)
	model.load_from_file(model_file)
	embedding = model.get_embedding(distance_matrix)
	print(embedding)


if pdb_file.startswith("gs://"):
	file_context= gcs.download_gzip_with_context(gcs.uri_to_bucket_and_key(pdb_file)[1])
	with gzip.open(file_context, mode="rt") as f:
		vectorize_protein(f)
else:
	with open(pdb_file, 'r') as f:
		vectorize_protein(f)