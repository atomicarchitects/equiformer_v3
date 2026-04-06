import os
import torch
import ase
from tqdm import tqdm
import numpy as np
from fairchem.core.datasets import AseDBDataset


_NUM_SAMPLES = 10_000
_SOURCE_DIR = './aselmdb_uncorrected_total_energy'
_TARGET_DIR = './aselmdb_uncorrected_total_energy_10k'


if __name__ == '__main__':

    torch.random.manual_seed(0)
    
    dataset = AseDBDataset(
        {
            'src': _SOURCE_DIR,
            'a2g_args': {
                'r_energy': True,
                'r_forces': True,
                'r_stress': True,
            }
        }
    )
    length = len(dataset)
    print('Dataset length: {}'.format(length))
    idx_list = torch.randperm(length)

    os.makedirs(_TARGET_DIR, exist_ok=True)
    db = ase.db.connect(os.path.join(_TARGET_DIR, 'data.aselmdb'))
    natoms_list = []
    for i in tqdm(range(_NUM_SAMPLES)):
        atoms = dataset.get_atoms(idx_list[i])
        db.write(atoms, data=atoms.info)
        natoms_list.append(len(atoms))
    np.savez(
        os.path.join(_TARGET_DIR, 'metadata.npz'),
        natoms=(np.array(natoms_list))
    )