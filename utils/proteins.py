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


def get_protein_shapes_from_cif(parsed_cif):
    obj = parsed_cif.get_object("struct_conf")
    if obj:
        struct_shape = pd.DataFrame(obj._row_list, columns=obj._attribute_name_list)

        index_col = "label_seq_id"
        struct_shape[index_col] = struct_shape.apply(lambda x: list(range(int(x["beg_label_seq_id"]), int(x["end_label_seq_id"]) + 1)), axis=1)
        struct_shape_explode = struct_shape[["beg_label_comp_id", "beg_label_seq_id", "conf_type_id", "end_label_comp_id", "end_label_seq_id", "id", index_col]].explode(index_col, ignore_index=True)
        struct_shape_explode = struct_shape_explode.rename(columns = {n: f"shape.{n}" if n != index_col else n for n in struct_shape_explode.columns})
        return struct_shape_explode
    return None


def get_protein_sequence_from_cif(parsed_cif):
    obj = parsed_cif.get_object("struct_ref")
    df = pd.DataFrame(obj._row_list, columns=obj._attribute_name_list)
    return df


def get_atom_sites_from_cif(parsed_cif):
    obj = parsed_cif.get_object("atom_site")
    df = pd.DataFrame(obj._row_list, columns=obj._attribute_name_list)
    return df


def get_global_confidence_from_cif(parsed_cif):
    obj = parsed_cif.get_object("ma_qa_metric_global")
    return pd.DataFrame(obj._row_list, columns=obj._attribute_name_list).iloc[0]["metric_value"]


def get_local_confidence_from_cif(parsed_cif):
    # local confidence is indexed by the amino acid sequence
    obj = parsed_cif.get_object("ma_qa_metric_local")
    df = pd.DataFrame(obj._row_list, columns=obj._attribute_name_list)
    return df


def join_atoms_with_confidence(atoms, local_confidence):
    min_amino_index = atoms.label_seq_id.astype(int).min() - 1
    amino_index = atoms.label_seq_id.astype(int) - min_amino_index
    
    atoms_index = atoms.label_comp_id + amino_index.astype(str)
    atoms_w_index = atoms.set_index(atoms_index)
    
    confidence_index = local_confidence.label_comp_id + local_confidence.label_seq_id
    confidence_w_index = pd.DataFrame(local_confidence.metric_value.rename("confidence_pLDDT")).set_index(confidence_index)
    
    return atoms_w_index.join(confidence_w_index, how="left")


def join_atoms_with_shape(atoms, struct_shape_explode):
    index_col = "label_seq_id"
    atoms[index_col] = atoms[index_col].astype(int)
    if struct_shape_explode is not None:
        struct_shape_explode[index_col] = struct_shape_explode[index_col].astype(int)
        return atoms.merge(struct_shape_explode, how="left", on=index_col)
    else: 
        return atoms


def reduce_sequence_df(df):
    df['pdbx_align_begin'] = df['pdbx_align_begin'].astype(int)
    df['file_number'] = df.protein_filename.apply(protein_file_number)
    assert df['file_number'].dtype == int
    return df.sort_values(['pdbx_db_accession', 'pdbx_align_begin', 'file_number'])\
        .groupby(['pdbx_db_accession', 'db_code', 'db_name', "protein_id"])\
        .agg({"pdbx_seq_one_letter_code": reduce_sequences, "protein_filename": lambda x: x.iloc[0], "confidence_pLDDT": lambda x: np.mean(x.astype(float))})\
        .reset_index()


def reduce_sequences(seqs):
    return ''.join(seqs.str.strip().replace('\n','', regex=True))


def sort_by_file_number_and_index(atom_df):
    """
    This utility method is used to make sure that atoms are returned in order based on the file part that they came in and the id within that file
    """
    atom_df["id"] = atom_df["id"].astype(int)
    atom_df["file_number"] = atom_df.protein_filename.apply(protein_file_number)
    sorted_atom_df = atom_df.sort_values(["protein_id", "file_number", "id"]).drop(columns=["file_number"])
    return sorted_atom_df.reset_index(drop=True)


def sequence_to_fasta_format(protein_name, sequence):
    n_amino = 60
    sequences_split = [sequence[index : index + n_amino] for index in range(0, len(sequence), n_amino)]
    fasta_sequence = '\n'.join(sequences_split)
    return f">{protein_name}\n{fasta_sequence}"