from typing import Any
import ase
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
import json
import os
from tqdm import tqdm
import lmdb
import torch
import torch_scatter
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.common.utils import radius_graph_pbc


_JSON_PATH = './MPtrj_2022.9_full.json'
_OUTPUT_DIR = './aselmdb_uncorrected_total_energy'
_NUM_CHUNK_LMDB_FILES = 15

_cutoff = 6.0


def chgnet_to_ase_atoms(id, data):
    """
        1.  Reference: https://github.com/janosh/matbench-discovery/blob/main/models/sevennet/train_sevennet/convert_mptrj_to_xyz.py
    """
    # Convert stress from kBar to eV/A^3 and use ASE sign convention
    kbar_to_evpa3 = -0.1 * ase.units.GPa
    info_keys = [
        "uncorrected_total_energy",
        "ef_per_atom",
        "e_per_atom_relaxed",
        "ef_per_atom_relaxed",
        "magmom",
        "bandgap",
        "mp_id",
    ]

    energy = data["uncorrected_total_energy"]
    #energy = data["corrected_total_energy"]
    force = data["force"]
    #stress = full_3x3_to_voigt_6_stress(dtm["stress"])  # internal stress
    stress = data['stress']
    stress = np.array(stress)
    stress = stress.astype(np.float32)
    stress = stress * kbar_to_evpa3  # to eV/Angstrom^3

    struct = data["structure"]
    cell = struct["lattice"]["matrix"]
    sites = struct["sites"]
    species = [ase.data.atomic_numbers[site["species"][0]["element"]] for site in sites]
    pos = [site["xyz"] for site in sites]

    atoms = Atoms(species, pos, cell=cell, pbc=True)
    calc_results = {
        'energy': energy,
        'free_energy': energy,
        'forces': force,
        'stress': stress,
    }
    calculator = SinglePointCalculator(atoms, **calc_results)
    atoms = calculator.get_atoms()

    #info = {
    #    "data_from": "MP-CHGNet",
    #    "material_id": id.split("-")[0] + "-" + id.split("-")[1],
    #    "calc_id": id.split("-")[2],
    #    "ionic_step_id": id.split("-")[3],
    #}
    #for key in info_keys:
    #    info[key] = data[key]
    #atoms.info = info

    atoms.info = {'sid': id}
    
    return atoms


def check_all_atoms_not_isolated(atoms, a2g):
    data_object = a2g.convert(atoms)
    num_atoms = data_object.natoms
    data_object.natoms = torch.tensor([data_object.natoms]).int()
    edge_index, _, _ = radius_graph_pbc(data_object, _cutoff, 1000, True)
    one_tensor = torch.ones((num_atoms, ))
    one_tensor = one_tensor[edge_index[0]]
    degree_tensor = torch_scatter.scatter(src=one_tensor, index=edge_index[1], dim_size=num_atoms, dim=0)
    if torch.all(degree_tensor):
        return True
    else:
        return False


if __name__ == '__main__':

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=_cutoff,
        r_energy=True,
        r_forces=True,
        r_stress=True,
        r_distances=False,
        r_fixed=True,
        r_edges=False,
        r_pbc=True
    )

    print('Loading: {}'.format(_JSON_PATH))
    with open(_JSON_PATH, 'r') as f:
        data = json.load(f)
    dataset = []
    for temp in data.values():
        for k, v in temp.items():
            dataset.append((k, v))
    print('Loaded MPTrj dataset with length: {}'.format(len(dataset)))

    chunk_size = len(dataset) // _NUM_CHUNK_LMDB_FILES
    remainder = len(dataset) % _NUM_CHUNK_LMDB_FILES
    start_chunk_idx = 0
    end_chunk_idx = 0    
    
    natoms_list = []
    
    for chunk_idx in range(_NUM_CHUNK_LMDB_FILES):
        db_path = os.path.join(_OUTPUT_DIR, 'data_{}.aselmdb'.format(chunk_idx))
        db = ase.db.connect(db_path)
        
        end_chunk_idx = start_chunk_idx + chunk_size + (1 if chunk_idx < remainder else 0)
        chunk_data = dataset[start_chunk_idx : end_chunk_idx]
        start_chunk_idx = end_chunk_idx
        
        idx = 0
        for entry in tqdm(chunk_data, desc=f'Processing chunk {chunk_idx}'):
            atoms = chgnet_to_ase_atoms(entry[0], entry[1])
            if not check_all_atoms_not_isolated(atoms, a2g):
                continue
            db.write(atoms, data=atoms.info)
            natoms_list.append(len(atoms))
            idx = idx + 1
        db.close()

        print('Finish processing chunk {}: original data length {} and data written {}'.format(
            chunk_idx, len(chunk_data), idx
        ))

    np.savez(
        os.path.join(_OUTPUT_DIR, 'metadata.npz'),
        natoms=(np.array(natoms_list))
    )

    #print(f'Conversion complete. LMDB files saved in {_OUTPUT_DIR}. Total entries: {data_count}')