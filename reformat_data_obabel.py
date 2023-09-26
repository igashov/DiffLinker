import argparse
import os
import pandas as pd
import subprocess

from rdkit import Chem
from src.utils import disable_rdkit_logging

from tqdm import tqdm
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--samples', action='store', type=str, required=True)
parser.add_argument('--checkpoint', action='store', type=str, required=True)
parser.add_argument('--dataset', action='store', type=str, required=True)
parser.add_argument('--true_smiles_path', action='store', type=str, required=True)
parser.add_argument('--formatted', action='store', type=str, required=True)
parser.add_argument('--linker_size_model_name', action='store', type=str, required=False, default=None)


def load_rdkit_molecule(xyz_path, obabel_path, true_frag_smi):
    if not os.path.exists(obabel_path):
        subprocess.run(f'obabel {xyz_path} -O {obabel_path}', shell=True)

    supp = Chem.SDMolSupplier(obabel_path, sanitize=False)
    mol = list(supp)[0]

    # Keeping only the biggest connected part
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    mol_filtered = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    try:
        mol_smi = Chem.MolToSmiles(mol_filtered)
    except RuntimeError:
        mol_smi = Chem.MolToSmiles(mol_filtered, canonical=False)

    # Retrieving linker
    true_frag = Chem.MolFromSmiles(true_frag_smi, sanitize=False)
    match = mol_filtered.GetSubstructMatch(true_frag)
    if len(match) == 0:
        linker_smi = ''
    else:
        elinker = Chem.EditableMol(mol_filtered)
        for atom in sorted(match, reverse=True):
            elinker.RemoveAtom(atom)
        linker = elinker.GetMol()
        Chem.Kekulize(linker, clearAromaticFlags=True)
        try:
            linker_smi = Chem.MolToSmiles(linker)
        except RuntimeError:
            linker_smi = Chem.MolToSmiles(linker, canonical=False)

    return mol_filtered, mol_smi, linker_smi


def load_molecules(folder, true_frag_smi):
    obabel_dir = f'{folder}/obabel'
    os.makedirs(obabel_dir, exist_ok=True)

    pred_mols = []
    pred_mols_smi = []
    pred_link_smi = []
    for fname in os.listdir(folder):
        number = fname.split('_')[0]
        if number.isdigit():
            pred_path = f'{folder}/{fname}'
            pred_obabel_path = f'{obabel_dir}/{number}_.sdf'
            mol, mol_smi, link_smi = load_rdkit_molecule(pred_path, pred_obabel_path, true_frag_smi)
            pred_mols.append(mol)
            pred_mols_smi.append(mol_smi)
            pred_link_smi.append(link_smi)

    return pred_mols, pred_mols_smi, pred_link_smi


def load_sampled_dataset(folder, idx2true_mol_smi, idx2true_frag_smi):
    pred_mols = []
    pred_mols_smi = []
    pred_link_smi = []
    true_mols_smi = []
    true_frags_smi = []
    uuids = []

    for fname in tqdm(os.listdir(folder)):
        if fname.isdigit():
            true_mol_smi = idx2true_mol_smi[int(fname)]
            true_frag_smi = idx2true_frag_smi[int(fname)]

            mols, mols_smi, link_smi = load_molecules(f'{folder}/{fname}', true_frag_smi)
            pred_mols += mols
            pred_mols_smi += mols_smi
            pred_link_smi += link_smi
            true_mols_smi += [true_mol_smi] * len(mols)
            true_frags_smi += [true_frag_smi] * len(mols)
            uuids += [fname] * len(mols)

    return pred_mols, pred_mols_smi, pred_link_smi, true_mols_smi, true_frags_smi, uuids


def reformat(samples, dataset, true_smiles_path, checkpoint, formatted, linker_size_model_name):
    true_smiles_path = os.path.join(true_smiles_path)
    if linker_size_model_name is None:
        input_path = os.path.join(samples, dataset, checkpoint)
        formatted_output_dir = os.path.join(formatted, checkpoint)
        out_smi_path = os.path.join(formatted_output_dir, f'{dataset}.smi')
        out_sdf_path = os.path.join(formatted_output_dir, f'{dataset}.sdf')
    else:
        input_path = os.path.join(samples, dataset, 'sampled_size', linker_size_model_name, checkpoint)
        formatted_output_dir = os.path.join(formatted, checkpoint, 'sampled_size', linker_size_model_name)
        out_smi_path = os.path.join(formatted_output_dir, f'{dataset}.smi')
        out_sdf_path = os.path.join(formatted_output_dir, f'{dataset}.sdf')

    print(f'Sampled SMILES will be saved to {out_smi_path}')
    print(f'Sampled molecules will be saved to {out_sdf_path}')

    true_smiles_table = pd.read_csv(true_smiles_path, sep=' ', names=['molecule', 'fragments'])
    if 'MOAD' in dataset:
        import numpy as np
        uuids = np.loadtxt(os.path.join(formatted, dataset.split('.')[0], 'uuids.txt'), dtype=int)
        idx2true_mol_smi = dict(zip(uuids, true_smiles_table.molecule.values))
        idx2true_frag_smi = dict(zip(uuids, true_smiles_table.fragments.values))
    else:
        idx2true_mol_smi = dict(enumerate(true_smiles_table.molecule.values))
        idx2true_frag_smi = dict(enumerate(true_smiles_table.fragments.values))

    pred_mols, pred_mols_smi, pred_link_smi, true_mols_smi, true_frag_smi, uuids = load_sampled_dataset(
        folder=input_path,
        idx2true_mol_smi=idx2true_mol_smi,
        idx2true_frag_smi=idx2true_frag_smi,
    )

    os.makedirs(formatted_output_dir, exist_ok=True)
    with open(out_smi_path, 'w') as f:
        for i in range(len(pred_mols_smi)):
            f.write(f'{true_frag_smi[i]} {true_mols_smi[i]} {pred_mols_smi[i]} {pred_link_smi[i]} {uuids[i]}\n')

    with Chem.SDWriter(open(out_sdf_path, 'w')) as writer:
        for mol in pred_mols:
            writer.write(mol)


if __name__ == '__main__':
    disable_rdkit_logging()
    args = parser.parse_args()
    reformat(
        samples=args.samples,
        dataset=args.dataset,
        true_smiles_path=args.true_smiles_path,
        checkpoint=args.checkpoint,
        formatted=args.formatted,
        linker_size_model_name=args.linker_size_model_name,
    )
