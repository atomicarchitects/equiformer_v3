from glob import glob
from typing import Annotated

import pandas as pd
import typer
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatviz.enums import Key
from tqdm import tqdm
import warnings
import os

from matbench_discovery.data import as_dict_handler, df_wbm
from matbench_discovery.energy import get_e_form_per_atom, mp_elemental_ref_energies
from matbench_discovery.enums import DataFiles, MbdKey


def join_predictions(
    input_dir: Annotated[
        str, typer.Option(help="Input directory to predicted relaxed energy files.")
    ],
    apply_mp_corrections: Annotated[
        bool, typer.Option(help="Apply MP 2020 corrections to formation energies")
    ] = True,
) -> None:
    """
        1.  Calculate formation energy per atom
        2.  Apply MP corrections (since we train on `uncorrected_energy`)
        3.  Write a single result file
    """
    e_form_fairchem_col = 'e_form_per_atom_fairchem'
    glob_pattern = '*.json.gz'
    file_paths = sorted(glob(f"{input_dir}/{glob_pattern}"))

    print(f"Found {len(file_paths):,} files for {glob_pattern = }")

    dfs: dict[str, pd.DataFrame] = {}

    for file_path in tqdm(file_paths, desc="Loading prediction files"):
        if file_path in dfs:
            continue
        dfs[file_path] = pd.read_json(file_path, lines=True).set_index(Key.mat_id)

    df_fairchem = pd.concat(dfs.values()).round(4)

    # make sure there is no missing structure
    if len(df_fairchem) != len(df_wbm):
        warnings.warn("Missing structures", stacklevel=2)

    wbm_cse_path = DataFiles.wbm_computed_structure_entries.path
    df_wbm_cse = pd.read_json(wbm_cse_path, lines=True).set_index(Key.mat_id)

    df_wbm_cse[Key.computed_structure_entry] = [
        ComputedStructureEntry.from_dict(dct)
        for dct in tqdm(df_wbm_cse[Key.computed_structure_entry], desc="Hydrate CSEs")
    ]

    # transfer energies and relaxed structures WBM CSEs since MP2020 energy
    # corrections applied below are structure-dependent (for oxides and sulfides)
    cse: ComputedStructureEntry
    for row in tqdm(
        df_fairchem.itertuples(), total=len(df_fairchem), desc="ML energies to CSEs"
    ):
        mat_id, struct_dict, pred_energy, *_ = row
        mlip_struct = Structure.from_dict(struct_dict)
        cse = df_wbm_cse.loc[mat_id, Key.computed_structure_entry]
        # cse._energy is the uncorrected energy
        cse._energy = pred_energy  # noqa: SLF001
        cse._structure = mlip_struct  # noqa: SLF001
        df_fairchem.loc[mat_id, Key.computed_structure_entry] = cse

    # apply corrections for models that were not trained on MP corrected energies
    if apply_mp_corrections:
        # apply energy corrections
        processed = MaterialsProject2020Compatibility().process_entries(
            df_fairchem[Key.computed_structure_entry], verbose=True, clean=True
        )
        if len(processed) != len(df_fairchem):
            raise ValueError(
                f"not all entries processed: {len(processed)=} {len(df_fairchem)=}"
            )

    # compute corrected formation energies
    df_fairchem[Key.formula] = df_wbm[Key.formula]
    df_fairchem[e_form_fairchem_col] = [
        get_e_form_per_atom(
            dict(energy=cse.energy, composition=formula),
            mp_elemental_ref_energies
        )
        for formula, cse in tqdm(
            df_fairchem.set_index(Key.formula)[Key.computed_structure_entry].items(),
            total=len(df_fairchem),
            desc="Computing formation energies",
        )
    ]
    df_wbm[[*df_fairchem]] = df_fairchem

    bad_mask = abs(df_wbm[e_form_fairchem_col] - df_wbm[MbdKey.e_form_dft]) > 5
    n_preds = len(df_wbm[e_form_fairchem_col].dropna())
    print(f"{sum(bad_mask)=} is {sum(bad_mask) / len(df_wbm):.2%} of {n_preds:,}")
    df_fairchem = df_fairchem.round(4)

    #os.makedirs(output_path, exist_ok=True)
    df_fairchem.select_dtypes("number").to_csv(f"{input_dir}/results.csv.gz")
    df_fairchem.reset_index().to_json(
        f"{input_dir}/results.json.gz",
        default_handler=as_dict_handler,
        orient="records",
        lines=True,
    )
    print('Save `results.csv.gz` and `results.json.gz` in {}'.format(input_dir))


if __name__ == "__main__":
    typer.run(join_predictions)