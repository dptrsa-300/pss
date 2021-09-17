import os
import sys
sys.path.append("..")
sys.path.append("./DeepFold/scripts")

import utils.spark_utils as 
from network import DeepFold
from distance_matrix import get_distance_matrix

model_file = os.path.join(os.path.dirname(__file__), './DeepFold/models/deepfold.model')
pdb_file = os.path.join(os.path.dirname(__file__), './../AF-A0A0A0MRZ7-F1-model_v1.pdb')

distance_matrix = get_distance_matrix(pdb_file).astype("float32")
model = DeepFold(max_length=distance_matrix.shape[0], projection_level=1)
model.load_from_file(model_file)
embedding = model.get_embedding(distance_matrix)
print(embedding)