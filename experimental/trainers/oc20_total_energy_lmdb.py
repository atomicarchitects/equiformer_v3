import bisect
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch

from fairchem.core.common.registry import registry
from fairchem.core.common.typing import assert_is_instance as aii
from fairchem.core.common.utils import pyg2_data_transform
from fairchem.core.datasets._utils import rename_data_object_keys
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.modules.transforms import DataTransforms


@registry.register_dataset("oc20_total_energy_lmdb")
class OC20TotalEnergyLmdbDataset(BaseDataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    The keys in the LMDB must be integers (stored as ascii objects) starting
    from 0 through the length of the LMDB. For historical reasons any key named
    "length" is ignored since that was used to infer length of many lmdbs in the same
    folder, but lmdb lengths are now calculated directly from the number of keys.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None) -> None:
        self.config = config

        self.path = Path(self.config["src"])
        self.paths = [self.path]
        self.data2train = self.config.get("data2train", "all")
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                self.envs.append(cur_env)

                # Get the number of stores data from the number of entries
                # in the LMDB
                num_entries = aii(cur_env.stat()["entries"], int)

                # If "length" encoded as ascii is present, we have one fewer
                # data than the stats suggest
                if cur_env.begin().get("length".encode("ascii")) is not None:
                    num_entries -= 1

                # Append the keys (0->num_entries) as a list
                self._keys.append(list(range(num_entries)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)

            if self.data2train != "all":
                txt_paths = sorted(self.path.glob("*.txt"))
                index = 0
                self.indices = []
                for txt_path in txt_paths:
                    lines = open(txt_path).read().splitlines()
                    for line in lines:
                        if self.data2train == "adslabs":
                            if "clean" not in line:
                                self.indices.append(index)
                        if self.data2train == "slabs":
                            if "clean" in line:
                                self.indices.append(index)
                        index += 1
                self.num_samples = len(self.indices)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)

            num_entries = aii(self.env.stat()["entries"], int)

            # If "length" encoded as ascii is present, we have one fewer
            # data than the stats suggest
            if self.env.begin().get("length".encode("ascii")) is not None:
                num_entries -= 1

            self._keys = list(range(num_entries))
            self.num_samples = num_entries

        self.key_mapping = self.config.get("key_mapping", None)
        self.transforms = DataTransforms(self.config.get("transforms", {}))

        self.use_total_energy = self.config.get("use_total_enery", False)
        self.oc20_ref = False
        self.lin_ref = False

        """
            Since the later version of OC20 S2EF dataset would have `energy` (total energy) and
            `y` (adsorption energy), we want to skip loading `oc20_ref`.
        """
        if self.use_total_energy:
            assert self.config.get("oc20_ref")
        if self.config.get("oc20_ref", False):
            self.oc20_ref = pickle.load(open(self.config["oc20_ref"], "rb"))
        
        if self.config.get("lin_ref", False):
            coeff = np.load(self.config["lin_ref"], allow_pickle=True)["coeff"]
            self.lin_ref = torch.nn.Parameter(
                torch.tensor(coeff), requires_grad=False
            )
        self.subsample = aii(self.config.get("subsample", False), bool)

    def __len__(self) -> int:
        if self.subsample:
            return min(self.subsample, self.num_samples)
        return self.num_samples

    def __getitem__(self, idx):
        if self.data2train != "all":
            idx = self.indices[idx]
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(
                f"{self._keys[idx]}".encode("ascii")
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))

        # make types consistent
        sid = data_object.sid
        if isinstance(sid, torch.Tensor):
            sid = sid.item()
            data_object.sid = sid
        if "fid" in data_object:
            fid = data_object.fid
            if isinstance(fid, torch.Tensor):
                fid = fid.item()
                data_object.fid = fid

        attr = None
        if getattr(data_object, "y_relaxed", None) is not None:
            attr = "y_relaxed"
        elif getattr(data_object, "y", None) is not None:
            attr = "y"
        elif getattr(data_object, 'energy', None) is not None:
            pass
        # if targets are not available, test data is being used
        else:
            return self.transforms(data_object)

        # convert s2ef energies to raw energies
        if attr in ["y", "y_relaxed"]:
            if self.use_total_energy:
                randomid = f"random{sid}"
                data_object[attr] += self.oc20_ref[randomid]
                data_object.nads = 1
                data_object.oc22 = 0

        # This would force using `energy` (total energy) as the only energy label
        if attr in ["y", "y_relaxed"]:
            if getattr(data_object, "energy", None) is None:
                data_object.energy = data_object[attr]
            del data_object[attr]

        if self.lin_ref is not False and getattr(data_object, "energy", False):
            lin_energy = sum(self.lin_ref[data_object.atomic_numbers.long()])
            data_object.energy -= lin_energy

        if self.key_mapping is not None:
            data_object = rename_data_object_keys(
                data_object, self.key_mapping
            )

        # to jointly train on oc22+oc20, need to delete these oc20-only attributes
        # ensure otf_graph=1 in your model configuration
        if "edge_index" in data_object:
            del data_object.edge_index
        if "cell_offsets" in data_object:
            del data_object.cell_offsets
        if "distances" in data_object:
            del data_object.distances

        data_object = self.transforms(data_object)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self) -> None:
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()