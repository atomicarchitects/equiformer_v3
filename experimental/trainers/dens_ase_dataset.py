from typing import Any, Callable
import ase
import os
from glob import glob
import torch
import bisect

from fairchem.core.common.registry import registry
from fairchem.core.datasets.ase_datasets import AseDBDataset, apply_one_tags
from fairchem.core.datasets._utils import rename_data_object_keys


_SKIP_DENS_FILENAME = 'skip_dens.txt'


@registry.register_dataset("dens_ase_db")
class DeNSAseDBDataset(AseDBDataset):
    """
    1.  This dataset class uses inherits from `AseDBDataset` so that we can add `skip_dens` attribute
        to graph data to skip applying DeNS for certain structures.
    2.  To skip applying DeNS, we need to add `skip_dens.txt` under the same directory as `.aselmdb` files.

    args:
        config (dict):
            src (str): Either
                    - the path an ASE DB,
                    - the connection address of an ASE DB,
                    - a folder with multiple ASE DBs,
                    - a list of folders with ASE DBs
                    - a glob string to use to find ASE DBs, or
                    - a list of ASE db paths/addresses.
                    If a folder, every file will be attempted as an ASE DB, and warnings
                    are raised for any files that can't connect cleanly

                    Note that for large datasets, ID loading can be slow and there can be many
                    ids, so it's advised to make loading the id list as easy as possible. There is not
                    an obvious way to get a full list of ids from most ASE dbs besides simply looping
                    through the entire dataset. See the AseLMDBDataset which was written with this usecase
                    in mind.

            connect_args (dict): Keyword arguments for ase.db.connect()

            select_args (dict): Keyword arguments for ase.db.select()
                    You can use this to query/filter your database

            a2g_args (dict): Keyword arguments for fairchem.core.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True, "r_forces":True, and/or "r_stress":True as appropriate
                    In that case, energy/forces must be in the database

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transforms (dict[str, dict]): Dictionary specifying data transforms as {transform_function: config}
                    where config is a dictionary specifying arguments to the transform_function

            key_mapping (dict[str, str]): Dictionary specifying a mapping between the name of a property used
                in the model with the corresponding property as it was named in the dataset. Only need to use if
                the name is different.

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
                    object. Useful for applying tags, for example.

        transform (callable, optional): deprecated?
    """

    def __init__(
        self,
        config: dict,
        atoms_transform: Callable[[ase.Atoms, Any, ...], ase.Atoms] = apply_one_tags,
    ) -> None:
        super().__init__(config, atoms_transform)

        # contruct a list recording whether structures in each .aselmdb file
        # should skip applying DeNS 
        self.aselmdb_skip_dens_list = []
        
        if isinstance(config["src"], list):
            filepaths = []
            for path in sorted(config["src"]):
                if os.path.isdir(path):
                    filepaths.extend(sorted(glob(f"{path}/*")))
                elif os.path.isfile(path):
                    filepaths.append(path)
                else:
                    raise RuntimeError(f"Error reading dataset in {path}!")
        elif os.path.isfile(config["src"]):
            filepaths = [config["src"]]
        elif os.path.isdir(config["src"]):
            filepaths = sorted(glob(f'{config["src"]}/*'))
        else:
            filepaths = sorted(glob(config["src"]))

        for path in filepaths:
            if '.aselmdb' in path and '.aselmdb-lock' not in path:
                directory = os.path.dirname(path)
                if os.path.exists(os.path.join(directory, _SKIP_DENS_FILENAME)):
                    self.aselmdb_skip_dens_list.append(True)
                else:
                    self.aselmdb_skip_dens_list.append(False)

        assert len(self.aselmdb_skip_dens_list) == len(self._idlen_cumulative)

    
    def __getitem__(self, idx):
        # Handle slicing
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        # Get atoms object via derived class method
        atoms = self.get_atoms(self.ids[idx])

        # Transform atoms object
        if self.atoms_transform is not None:
            atoms = self.atoms_transform(
                atoms, **self.config.get("atoms_transform_args", {})
            )

        sid = atoms.info.get("sid", self.ids[idx])
        fid = atoms.info.get("fid", torch.tensor([0]))

        # Convert to data object
        data_object = self.a2g.convert(atoms, sid)
        data_object.fid = fid
        data_object.natoms = len(atoms)

        # apply linear reference
        if self.a2g.r_energy is True and self.lin_ref is not None:
            data_object.energy -= sum(self.lin_ref[data_object.atomic_numbers.long()])

        # Transform data object
        data_object = self.transforms(data_object)

        if self.key_mapping is not None:
            data_object = rename_data_object_keys(data_object, self.key_mapping)

        if self.config.get("include_relaxed_energy", False):
            data_object.energy_relaxed = self.get_relaxed_energy(self.ids[idx])

        # for skipping DeNS
        db_idx = bisect.bisect(self._idlen_cumulative, idx)
        data_object.skip_dens = torch.tensor([self.aselmdb_skip_dens_list[db_idx]])

        return data_object