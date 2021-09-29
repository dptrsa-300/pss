from google.cloud import storage
import io
import numpy as np
import pandas as pd
import pdbx
from pdbx.reader import PdbxReader
from pdbx.reader import DataContainer


def parse_cif(cif_contents):
    data = []
    pRd = PdbxReader(io.StringIO(cif_contents))
    pRd.read(data)
    block = data[0]
    return block


def get_protein_sequence_from_cif(parsed_cif):
    obj = parsed_cif.get_object("struct_ref")
    df = pd.DataFrame(obj._row_list, columns=obj._attribute_name_list)
    return df


def get_atom_sites_from_cif(parsed_cif):
    obj = parsed_cif.get_object("atom_site")
    df = pd.DataFrame(obj._row_list, columns=obj._attribute_name_list)
    return df


def reduce_sequence_df(df):
    return df.sort_values(['pdbx_db_accession', 'pdbx_align_begin'])\
        .groupby(['pdbx_db_accession', 'db_code', 'db_name'])\
        .agg({"pdbx_seq_one_letter_code": reduce_sequences, "protein_name": lambda x: x.iloc[0]})\
        .reset_index()


def reduce_sequences(seqs):
    return ''.join(seqs.str.strip().replace('\n','', regex=True))


def sequence_to_fasta_format(protein_name, sequence):
    n_amino = 60
    sequences_split = [sequence[index : index + n_amino] for index in range(0, len(sequence), n_amino)]
    fasta_sequence = '\n'.join(sequences_split)
    return f">{protein_name}\n{fasta_sequence}"