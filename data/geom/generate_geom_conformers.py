import argparse
import json
import os
import pandas as pd
import pickle

from rdkit import Chem
from tqdm import tqdm


def generate_conformers(geom_json_path, geom_root_path, generated_splits_path, output_path):
    with open(geom_json_path) as f:
        geom_json = json.load(f)

    splits_table = pd.read_csv(generated_splits_path)
    smiles = splits_table.molecule.unique()

    molecules = []
    for smi in tqdm(smiles):
        pickle_path = geom_json[smi].get('pickle_path')
        if pickle_path is None:
            continue

        mol_path = os.path.join(geom_root_path, pickle_path)
        molecule_data = pickle.load(open(mol_path, 'rb'))
        mol_dict = min(molecule_data['conformers'], key=lambda c: c['totalenergy'])
        mol = mol_dict['rd_mol']
        mol_energy = mol_dict['totalenergy']
        try:
            mol = Chem.RemoveAllHs(mol)
            mol.SetProp('_Name', smi)
            mol.SetProp('_Energy', str(mol_energy))
        except Exception as e:
            print(f'Error with {smi}: {e}')
            continue

        molecules.append(mol)

    with Chem.SDWriter(open(output_path, 'w')) as writer:
        for mol in molecules:
            writer.write(mol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geom-json', action='store', type=str, required=True)
    parser.add_argument('--geom-root', action='store', type=str, required=True)
    parser.add_argument('--generated-splits', action='store', type=str, required=True)
    parser.add_argument('--output', action='store', type=str, required=True)
    args = parser.parse_args()

    generate_conformers(
        geom_json_path=args.geom_json,
        geom_root_path=args.geom_root,
        generated_splits_path=args.generated_splits,
        output_path=args.output,
    )
