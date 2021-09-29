from google.cloud import storage
import io
import numpy as np
import pandas as pd
import pdbx
from pdbx.reader import PdbxReader
from pdbx.reader import DataContainer


def protein_file_number(name):
    """e.g. AF-Q8WZ42-F101-model_v1 --> 101"""
    return int(name.split("-")[2].strip("F"))


def get_stripped_protein_filename(filename):
    return filename.split("/")[-1].replace(".cif.gz", "")


def get_protein_id_from_filename(filename):
    """e.g. AF-W5XKT8-F1-model_v1.cif.gz"""
    stripped = get_stripped_protein_filename(filename)
    return stripped.split("-")[1]


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


def sort_by_file_number_and_index(atom_df):
    """
    This utility method is used to make sure that atoms are returned in order based on the file part that they came in and the id within that file
    """
    atom_df["file_number"] = atom_df.protein_name.apply(protein_file_number)
    sorted_atom_df = atom_df.sort_values(["file_number", "id"]).drop(columns=["file_number"])
    return sorted_atom_df


def sequence_to_fasta_format(protein_name, sequence):
    n_amino = 60
    sequences_split = [sequence[index : index + n_amino] for index in range(0, len(sequence), n_amino)]
    fasta_sequence = '\n'.join(sequences_split)
    return f">{protein_name}\n{fasta_sequence}"