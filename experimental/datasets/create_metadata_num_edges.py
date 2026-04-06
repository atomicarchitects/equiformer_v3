import ase
from ase import Atoms
import numpy as np
import json
import os
from tqdm import tqdm
import torch
import typer
from typing import Annotated

from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.common.utils import radius_graph_pbc
from fairchem.core.datasets import AseDBDataset


def compute_cost(atoms, a2g, cutoff):
    data_object = a2g.convert(atoms)
    data_object.natoms = torch.tensor([data_object.natoms]).int()
    edge_index, _, _ = radius_graph_pbc(data_object, cutoff, 1000, True)
    num_edges = len(edge_index[0])
    num_nodes = int(data_object.natoms.item())
    cost = num_edges
    return cost


def create_metadata(
    input_dir: Annotated[
        str, typer.Option(help="Input directory to .aselmdb files")
    ],
    cutoff: Annotated[
        float, typer.Option(help="Cutoff radius for creating edges")
    ] = 6.0,
) -> None:

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
        radius=cutoff,
        r_energy=True,
        r_forces=True,
        r_stress=True,
        r_distances=False,
        r_fixed=True,
        r_edges=False,
        r_pbc=True
    )

    natoms_list = []
    for i in tqdm(range(len(dataset))):
        atoms = dataset.get_atoms(i)
        num_edges = compute_cost(atoms, a2g, cutoff)
        natoms_list.append(num_edges)
    np.savez(
        os.path.join(input_dir, 'metadata_num-edges.npz'),
        natoms=(np.array(natoms_list))
    )


if __name__ == '__main__':
    typer.run(create_metadata)