from google.cloud import storage
import io
import numpy as np
import pandas as pd
import pdbx
from pdbx.reader import PdbxReader
from pdbx.reader import DataContainer


def get_protein_sequence_from_cif(cif_contents):
    data = []
    pRd = PdbxReader(io.StringIO(cif_contents))
    pRd.read(data)
    block = data[0]

    obj = block.get_object("struct_ref")
    df = pd.DataFrame(obj._row_list, columns=obj._attribute_name_list)
    return df


def sequence_to_fasta_format(protein_name, sequence):
    n_amino = 60
    sequences_split = [sequence[index : index + n_amino] for index in range(0, len(sequence), n_amino)]
    fasta_sequence = '\n'.join(sequences_split)
    return f">{protein_name}\n{fasta_sequence}"