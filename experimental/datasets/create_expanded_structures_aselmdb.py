import typer
from typing import Annotated
from pathlib import Path
import os
from tqdm import tqdm
import ase

import torch
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from fairchem.core.datasets import AseDBDataset
from fairchem.core.preprocessing import AtomsToGraphs


_num_structures_per_aselmdb_file = 50_000


def get_expanded_structures(atoms, a2g):
    num_atoms = len(atoms)

    # hard-coded expansion of unit cells 
    if num_atoms <= 2:
        scaling = (3, 3, 3)
    else:
        scaling = (2, 2, 2)
    
    expanded_atoms = atoms.repeat(scaling)
    scaling_factor = int(np.prod(scaling))

    # update labels
    data_object = a2g.convert(atoms)
    energy = data_object.energy
    energy = energy * scaling_factor
    forces = data_object.forces
    forces = forces.repeat((scaling_factor, 1))
    stress = data_object.stress
    calc_results = {
        'energy': energy,
        'free_energy': energy,
        'forces': forces,
        'stress': stress,
    }
    calculator = SinglePointCalculator(expanded_atoms, **calc_results)
    expanded_atoms = calculator.get_atoms()
    expanded_atoms.info = {'sid': atoms.info['sid']}
    
    return expanded_atoms
    

def create_expanded_structures_aselmdb(
    input_dir: Annotated[
        str, typer.Option(help="Input directory to .aselmdb files")
    ],
    min_num_atoms: Annotated[
        int, typer.Option(help="Minimum number of atoms in a unit cell")
    ] = 20,
) -> None:
    path_obj = Path(input_dir)
    new_path = path_obj.parent / (str(path_obj.name) + '_expand-min@{}'.format(min_num_atoms))
    output_dir = str(new_path)
    os.makedirs(output_dir, exist_ok=True)

    dataset = AseDBDataset(
        {
            'src': input_dir,
            'a2g_args': {
                'r_energy': True,
                'r_forces': True,
                'r_stress': True,
            }
        }
    )
    length = len(dataset)
    print('Dataset length: {}'.format(length))
    
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=6.0,
        r_energy=True,
        r_forces=True,
        r_stress=True,
        r_distances=False,
        r_fixed=True,
        r_edges=False,
        r_pbc=True
    )

    natoms_list = []
    num_expanded_structures = 0
    db = None
    
    for i in tqdm(range(len(dataset))):
        atoms = dataset.get_atoms(i)
        if len(atoms) < min_num_atoms:
            if num_expanded_structures % _num_structures_per_aselmdb_file == 0:
                file_index_str = str(num_expanded_structures // _num_structures_per_aselmdb_file)
                file_index_str = file_index_str.zfill(4)
                if db is not None:
                    db.close()
                db = ase.db.connect(
                    os.path.join(
                        output_dir, 
                        ('data_' + file_index_str + '.aselmdb')
                    )
                )
            expanded_atoms = get_expanded_structures(atoms, a2g)
            db.write(expanded_atoms, data=expanded_atoms.info)
            natoms_list.append(len(expanded_atoms))

            num_expanded_structures += 1
    
    if db is not None:
        db.close()
        
    np.savez(
        os.path.join(output_dir, 'metadata.npz'),
        natoms=(np.array(natoms_list))
    )

    print('Save {} expanded structures'.format(num_expanded_structures))
    

if __name__ == '__main__':
    typer.run(create_expanded_structures_aselmdb)