import argparse
import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

from rdkit import Chem
from data.zinc.prepare_dataset import run

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')


def prepare_zinc(zinc_dir):
    val_path = f'{zinc_dir}/data_zinc_final_valid.txt'
    val_data = pd.read_csv(val_path, sep=' ', names=['molecule', 'linker', 'fragments', 'distance', 'angle'])
    val_smiles = val_data.molecule.unique()

    test_path = f'{zinc_dir}/data_zinc_final_test.txt'
    test_data = pd.read_csv(test_path, sep=' ', names=['molecule', 'linker', 'fragments', 'distance', 'angle'])
    test_smiles = test_data.molecule.unique()

    sdf_path = f'{zinc_dir}/zinc_250k_valid_test_only.sdf'
    suppl = Chem.SDMolSupplier(sdf_path)
    val_mols = []
    test_mols = []

    for mol in suppl:
        if mol.GetProp('_Name') in val_smiles:
            val_mols.append(mol)
        if mol.GetProp('_Name') in test_smiles:
            test_mols.append(mol)

    np.savetxt(f'{zinc_dir}/smiles_zinc_final_val.txt', val_smiles, fmt='%s')
    np.savetxt(f'{zinc_dir}/smiles_zinc_final_test.txt', test_smiles, fmt='%s')

    val_sdf_path = f'{zinc_dir}/zinc_final_val.sdf'
    test_sdf_path = f'{zinc_dir}/zinc_final_test.sdf'

    with Chem.SDWriter(open(val_sdf_path, 'w')) as writer:
        for mol in val_mols:
            mol.SetProp('_Energy', '0')
            writer.write(mol)

    with Chem.SDWriter(open(test_sdf_path, 'w')) as writer:
        for mol in test_mols:
            mol.SetProp('_Energy', '0')
            writer.write(mol)

    run(
        table_path=val_path,
        sdf_path=val_sdf_path,
        out_mol_path=f'{zinc_dir}/zinc_final_val_mol.sdf',
        out_frag_path=f'{zinc_dir}/zinc_final_val_frag.sdf',
        out_link_path=f'{zinc_dir}/zinc_final_val_link.sdf',
        out_table_path=f'{zinc_dir}/zinc_final_val_table.csv',
    )
    run(
        table_path=test_path,
        sdf_path=test_sdf_path,
        out_mol_path=f'{zinc_dir}/zinc_final_test_mol.sdf',
        out_frag_path=f'{zinc_dir}/zinc_final_test_frag.sdf',
        out_link_path=f'{zinc_dir}/zinc_final_test_link.sdf',
        out_table_path=f'{zinc_dir}/zinc_final_test_table.csv',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zinc-dir', action='store', type=str, required=True)
    args = parser.parse_args()
    prepare_zinc(zinc_dir=args.zinc_dir)
