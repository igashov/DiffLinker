import argparse
import sys
sys.path.append('../../')

import os
import pandas as pd

from rdkit import Chem
from tqdm import tqdm
from src.datasets import read_sdf


def run(input_dir, output_dir, template, n):

    os.makedirs(output_dir, exist_ok=True)
    out_table_path = os.path.join(output_dir, f'{template}_table.csv')
    out_mol_path = os.path.join(output_dir, f'{template}_mol.sdf')
    out_frag_path = os.path.join(output_dir, f'{template}_frag.sdf')
    out_link_path = os.path.join(output_dir, f'{template}_link.sdf')

    full_table = []
    full_molecules = []
    full_fragments = []
    full_linkers = []

    for idx in range(n):
        mol_path = os.path.join(input_dir, f'{template}_mol_{idx}.sdf')
        frag_path = os.path.join(input_dir, f'{template}_frag_{idx}.sdf')
        link_path = os.path.join(input_dir, f'{template}_link_{idx}.sdf')
        table_path = os.path.join(input_dir, f'{template}_table_{idx}.csv')

        table = pd.read_csv(table_path)
        table['idx'] = table.index
        grouped_table = (
            table
            .groupby(['molecule', 'fragments', 'linker', 'anchor_1', 'anchor_2'])
            .min()
            .reset_index()
            .sort_values(by='idx')
        )
        idx_to_keep = set(grouped_table['idx'].unique())
        table['keep'] = table['idx'].isin(idx_to_keep)

        generator = tqdm(
            zip(table.iterrows(), read_sdf(mol_path), read_sdf(frag_path), read_sdf(link_path)),
            total=len(table),
            desc=str(idx),
        )
        for (_, row), molecule, fragments, linker in generator:
            if row['keep']:
                if molecule.GetProp('_Name') != row['molecule']:
                    print('Molecule _Name:', molecule.GetProp('_Name'), row['molecule'])
                    continue

                full_table.append(row)
                full_molecules.append(molecule)
                full_fragments.append(fragments)
                full_linkers.append(linker)

    full_table = pd.DataFrame(full_table)
    full_table.to_csv(out_table_path, index=False)
    with Chem.SDWriter(open(out_mol_path, 'w')) as writer:
        for mol in tqdm(full_molecules):
            writer.write(mol)
    with Chem.SDWriter(open(out_frag_path, 'w')) as writer:
        writer.SetKekulize(False)
        for frags in tqdm(full_fragments):
            writer.write(frags)
    with Chem.SDWriter(open(out_link_path, 'w')) as writer:
        writer.SetKekulize(False)
        for linker in tqdm(full_linkers):
            writer.write(linker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', action='store', type=str, required=True)
    parser.add_argument('--out-dir', action='store', type=str, required=True)
    parser.add_argument('--template', action='store', type=str, required=True)
    parser.add_argument('--number-of-files', action='store', type=int, required=True)
    args = parser.parse_args()

    run(
        input_dir=args.in_dir,
        output_dir=args.out_dir,
        template=args.template,
        n=args.number_of_files,
    )
