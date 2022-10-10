import argparse
import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

from rdkit import Chem
from tqdm import tqdm
from data.zinc.prepare_dataset import run

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')


def prepare_casf(casf_dir):
    casf_table_path = f'{casf_dir}/data_casf_final.txt'
    casf_data = pd.read_csv(casf_table_path, sep=' ', names=['molecule', 'linker', 'fragments', 'distance', 'angle'])
    casf_smiles = casf_data.molecule.unique()

    casf_sdf_path = f'{casf_dir}/casf_structures.sdf'
    suppl = Chem.SDMolSupplier(casf_sdf_path)

    mols = []
    bad_mols = []

    for i, mol in tqdm(enumerate(suppl), total=len(suppl)):
        try:
            newmol = Chem.RemoveAllHs(mol)
            if len(casf_data[casf_data.molecule == Chem.MolToSmiles(newmol)]) > 0:
                newmol.SetProp('_Identifier', newmol.GetProp('_Name'))
                newmol.SetProp('_Name', Chem.MolToSmiles(newmol))
                mols.append(newmol)
        except:
            bad_mols.append(mol)

    np.savetxt(f'{casf_dir}/smiles_casf_final_test.txt', casf_smiles, fmt='%s')
    out_casf_sdf_path = f'{casf_dir}/casf_structures_test.sdf'
    with Chem.SDWriter(open(out_casf_sdf_path, 'w')) as writer:
        for mol in mols:
            mol.SetProp('_Energy', '0')
            writer.write(mol)

    run(
        table_path=casf_table_path,
        sdf_path=out_casf_sdf_path,
        out_mol_path=f'{casf_dir}/casf_final_test_mol.sdf',
        out_frag_path=f'{casf_dir}/casf_final_test_frag.sdf',
        out_link_path=f'{casf_dir}/casf_final_test_link.sdf',
        out_table_path=f'{casf_dir}/casf_final_test_table.csv',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--casf-dir', action='store', type=str, required=True)
    args = parser.parse_args()
    prepare_casf(casf_dir=args.casf_dir)
