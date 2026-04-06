import os
from ase import db as ase_db
from tqdm import tqdm
import numpy as np
from matbench_discovery.data import DataFiles, ase_atoms_from_zip


_output_aselmdb_dir = '/data/NFS/radish/ylliao/omat24_dataset/matbench_discovery'


if __name__ == '__main__':
    wbm_init_atoms = ase_atoms_from_zip(DataFiles.wbm_initial_atoms.path)
    os.makedirs(_output_aselmdb_dir, exist_ok=True)
    
    db = ase_db.connect(os.path.join(_output_aselmdb_dir, 'WBM_IS2RE.aselmdb'))
    natoms_list = []
    for i in tqdm(range(len(wbm_init_atoms))):
        atoms = wbm_init_atoms[i]
        db.write(atoms, data=atoms.info)
        natoms_list.append(len(atoms))

    np.savez(
        os.path.join(_output_aselmdb_dir, 'metadata.npz'),
        natoms=(np.array(natoms_list))
    )