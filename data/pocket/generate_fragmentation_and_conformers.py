import sys
sys.path.append('../../')

import argparse
import os
import pandas as pd

from data.geom.generate_geom_multifrag import check_mmpa_linker, check_mmpa_fragments
from src.utils import disable_rdkit_logging

from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from tqdm import tqdm


def fragment_by_mmpa(mol, mol_name, mol_smiles, min_cuts, max_cuts, min_frag_size, min_link_size):
    mmpa_results = []
    for i in range(min_cuts, max_cuts + 1):
        mmpa_results += FragmentMol(
            mol,
            minCuts=i,
            maxCuts=i,
            maxCutBonds=100,
            pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]",
            resultsAsMols=False
        )

    filtered_mmpa_results = []
    for linker_smiles, fragments_smiles in mmpa_results:
        if check_mmpa_linker(linker_smiles, min_link_size) and check_mmpa_fragments(fragments_smiles, min_frag_size):
            filtered_mmpa_results.append([mol_name, mol_smiles, linker_smiles, fragments_smiles, 'mmpa'])
    return filtered_mmpa_results


def run(ligands_dir, output_table, output_conformers):
    min_frag_size = 5
    min_link_size = 3

    mol_results = []
    conformers = []
    for fname in tqdm(os.listdir(ligands_dir)):
        if fname.endswith('.mol') and not fname.startswith('._'):
            mol_name = fname.split('.')[0]
            try:
                mol = Chem.MolFromMolFile(os.path.join(ligands_dir, fname))
                mol = Chem.RemoveAllHs(mol)
                Chem.SanitizeMol(mol)
            except:
                continue
            if mol is None:
                continue

            if mol.GetNumAtoms() <= 40 and mol.GetRingInfo().NumRings() >= 2:
                try:
                    res = fragment_by_mmpa(
                        mol,
                        mol_smiles=Chem.MolToSmiles(mol),
                        mol_name=mol_name,
                        min_cuts=2,
                        max_cuts=2,
                        min_link_size=min_link_size,
                        min_frag_size=min_frag_size,
                    )
                except:
                    continue
                if len(res) > 0:
                    mol_results += res
                    mol.SetProp('_Name', mol_name)
                    conformers.append(mol)

    table = pd.DataFrame(mol_results, columns=['molecule_name', 'molecule', 'linker', 'fragments', 'method'])
    table = table.drop_duplicates(['molecule_name', 'molecule', 'linker'])
    table.to_csv(output_table, index=False)

    with Chem.SDWriter(open(output_conformers, 'w')) as writer:
        for mol in conformers:
            writer.write(mol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-ligands', action='store', type=str, required=True)
    parser.add_argument('--out-fragmentations', action='store', type=str, required=True)
    parser.add_argument('--out-conformers', action='store', type=str, required=True)
    args = parser.parse_args()

    disable_rdkit_logging()

    run(
        ligands_dir=args.in_ligands,
        output_table=args.out_fragmentations,
        output_conformers=args.out_conformers,
    )
