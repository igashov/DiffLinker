import argparse
import numpy as np
import pandas as pd
import pickle

from rdkit import Chem
from tqdm import tqdm


ALLOWED_TYPES = {'C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P'}
TEST_PDBS_PATH = '../../resources/moad_test_pdbs.txt'
VAL_PDBS_PATH = '../../resources/moad_val_pdbs.txt'


def assign_dataset(name, test_pdbs, val_pdbs):
    pdb = name.split('_')[0]
    if pdb in test_pdbs:
        return 'test'
    if pdb in val_pdbs:
        return 'val'
    return 'train'


def filter_and_split(mol_path, frag_path, link_path, pockets_path, table_path):
    mols_sdf = Chem.SDMolSupplier(mol_path, sanitize=False)
    frags_sdf = Chem.SDMolSupplier(frag_path, sanitize=False)
    link_sdf = Chem.SDMolSupplier(link_path, sanitize=False)
    pocket_data = pickle.load(open(pockets_path, 'rb'))

    table = pd.read_csv(table_path)

    # 1. Filter by molecule size
    table.loc[(table.pocket_full_size + table.molecule_size) >= 1000, 'discard'] = True

    # 2. Filter by allowed ligand atom types
    for i, mol in tqdm(enumerate(mols_sdf), total=len(mols_sdf)):
        types = set()
        for atom in mol.GetAtoms():
            types.add(atom.GetSymbol())

        if len(types.difference(ALLOWED_TYPES)) > 0:
            table.loc[i, 'discard'] = True

    # 3. Filter by pocket atom types
    for i, pdata in tqdm(enumerate(pocket_data), total=len(pocket_data)):
        types = set(pdata['full_types'])
        if len(types.difference(ALLOWED_TYPES)) > 0:
            table.loc[i, 'discard'] = True

    # 4. Filter by pocket size
    for i, pdata in tqdm(enumerate(pocket_data), total=len(pocket_data)):
        if len(pdata['full_coord']) == 0:
            table.loc[i, 'discard'] = True

    # Split in train, test, val
    test_pdbs = np.loadtxt(TEST_PDBS_PATH, fmt='%s')
    val_pdbs = np.loadtxt(VAL_PDBS_PATH, fmt='%s')
    table['dataset'] = table['molecule_name'].apply(lambda x: assign_dataset(x, test_pdbs, val_pdbs))
    print('Train:', len(table[(~table.discard) & (table.dataset == 'train')]))
    print('Test:', len(table[(~table.discard) & (table.dataset == 'test')]))
    print('Val:', len(table[(~table.discard) & (table.dataset == 'val')]))

    mols = {
        'train': [],
        'val': [],
        'test': [],
    }
    frags = {
        'train': [],
        'val': [],
        'test': [],
    }
    link = {
        'train': [],
        'val': [],
        'test': [],
    }
    pockets = {
        'train': [],
        'val': [],
        'test': [],
    }
    idx = {
        'train': [],
        'val': [],
        'test': [],
    }

    for i, (m, f, l, p) in tqdm(enumerate(zip(mols_sdf, frags_sdf, link_sdf, pocket_data)), total=len(mols_sdf)):
        discard = table.loc[i, 'discard']
        dataset = table.loc[i, 'dataset']
        if discard:
            continue

        mols[dataset].append(m)
        frags[dataset].append(f)
        link[dataset].append(l)
        pockets[dataset].append(p)
        idx[dataset].append(i)

    tables = {
        'train': table.loc[idx['train']].copy().reset_index(drop=True),
        'val': table.loc[idx['val']].copy().reset_index(drop=True),
        'test': table.loc[idx['test']].copy().reset_index(drop=True),
    }

    # Saving datasets
    template = mol_path.replace('_mol.sdf', '')
    for dataset in ['train', 'val', 'test']:
        mols_len = len(mols[dataset])
        frags_len = len(frags[dataset])
        link_len = len(link[dataset])
        pockets_len = len(pockets[dataset])
        table_len = len(tables[dataset])
        assert len({mols_len, frags_len, link_len, pockets_len, table_len}) == 1

        mol_sdf_path = f'{template}_{dataset}_mol.sdf'
        frag_sdf_path = f'{template}_{dataset}_frag.sdf'
        link_sdf_path = f'{template}_{dataset}_link.sdf'
        pockets_sdf_path = f'{template}_{dataset}_pockets.pkl'
        table_out_path = f'{template}_{dataset}_table.csv'

        with Chem.SDWriter(open(mol_sdf_path, 'w')) as writer:
            for mol in tqdm(mols[dataset], desc=dataset):
                writer.write(mol)

        with Chem.SDWriter(open(frag_sdf_path, 'w')) as writer:
            for mol in tqdm(frags[dataset], desc=dataset):
                writer.write(mol)

        with Chem.SDWriter(open(link_sdf_path, 'w')) as writer:
            for mol in tqdm(link[dataset], desc=dataset):
                writer.write(mol)

        with open(pockets_sdf_path, 'wb') as f:
            pickle.dump(pockets[dataset], f)

        tables[dataset].to_csv(table_out_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mol-sdf', action='store', type=str, required=True)
    parser.add_argument('--frag-sdf', action='store', type=str, required=True)
    parser.add_argument('--link-sdf', action='store', type=str, required=True)
    parser.add_argument('--pockets-pkl', action='store', type=str, required=True)
    parser.add_argument('--table', action='store', type=str, required=True)
    args = parser.parse_args()

    filter_and_split(
        mol_path=args.mol_sdf,
        frag_path=args.frag_sdf,
        link_path=args.link_sdf,
        pockets_path=args.pockets_pkl,
        table_path=args.table,
    )
