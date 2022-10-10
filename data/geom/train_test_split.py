import argparse
import pandas as pd

from rdkit import Chem
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split(table_path, train_path, val_path, test_path):
    table = pd.read_csv(table_path)
    table = table.drop_duplicates(['molecule', 'linker'])

    linker_sizes = []
    fragment_sizes = []
    number_of_linkers = []
    number_of_fragments = []
    for linker_smi, fragments_smi in tqdm(table[['linker', 'fragments']].values):
        linker = Chem.MolFromSmiles(linker_smi)
        fragments = Chem.MolFromSmiles(fragments_smi)
        linker_sizes.append(linker.GetNumAtoms())
        fragment_sizes.append(fragments.GetNumAtoms())
        number_of_linkers.append(len(linker_smi.split('.')))
        number_of_fragments.append(len(fragments_smi.split('.')))

    table['linker_size'] = linker_sizes
    table['fragment_size'] = fragment_sizes
    table['num_linkers'] = number_of_linkers
    table['num_fragments'] = number_of_fragments
    table = table[table.num_fragments > 2]

    grouped = table[['molecule', 'linker_size', 'num_linkers']].groupby('molecule').max().reset_index()
    grouped['stratify'] = grouped.linker_size.astype(str) + '_' + grouped.num_linkers.astype(str)

    counts = grouped['stratify'].value_counts()
    rare = set(counts[counts < 10].index.values)
    grouped['stratify'] = grouped['stratify'].apply(lambda g: 'rare' if g in rare else g)

    smiles = grouped.molecule.values
    stratify = grouped.stratify.values

    train_smi, test_smi, train_strat, test_strat = train_test_split(
        smiles, stratify,
        test_size=200,
        stratify=stratify,
        random_state=42
    )

    train_smi, val_smi, train_strat, val_strat = train_test_split(
        train_smi, train_strat,
        test_size=200,
        stratify=train_strat,
        random_state=42
    )

    assert len(set(val_smi) & set(test_smi)) == 0
    assert len(set(val_smi) & set(train_smi)) == 0
    assert len(set(test_smi) & set(train_smi)) == 0

    train_data = table[table.molecule.isin(train_smi)]
    val_data = table[table.molecule.isin(val_smi)]
    test_data = table[table.molecule.isin(test_smi)]

    print(f'Train size: {len(train_smi)} molecules, {len(train_data)} examples')
    print(f'Val size: {len(val_smi)} molecules, {len(val_data)} examples')
    print(f'Test size: {len(test_smi)} molecules, {len(test_data)} examples')

    train_data.to_csv(train_path)
    val_data.to_csv(val_path)
    test_data.to_csv(test_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-splits', action='store', type=str, required=True)
    parser.add_argument('--train', action='store', type=str, required=True)
    parser.add_argument('--val', action='store', type=str, required=True)
    parser.add_argument('--test', action='store', type=str, required=True)
    args = parser.parse_args()

    split(
        table_path=args.generated_splits,
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
    )

