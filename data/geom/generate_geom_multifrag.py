import argparse
import json
import numpy as np
import pandas as pd
import re

from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.BRICS import FindBRICSBonds
from tqdm import tqdm


# #################################################################################### #
# ####################################### BRICS ###################################### #
# #################################################################################### #

REGEX = re.compile('\[\d*\*\]')


def split_into_n_fragments(mol, bonds, num_frags):
    num_bonds = num_frags - 1
    bondidx2minfrag = {}
    bondidx2atoms = {}
    for bond in bonds:
        bond_idx = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx()
        frags = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
        frags = Chem.GetMolFrags(frags, asMols=True)
        minfragsize = min([f.GetNumAtoms() for f in frags])
        bondidx2minfrag[bond_idx] = minfragsize
        bondidx2atoms[bond_idx] = bond

    # Selecting only top-N bonds connecting the biggest 6 fragments
    sorted_bonds = sorted(bondidx2minfrag.keys(), key=lambda bidx: -bondidx2minfrag[bidx])
    bonds_to_split = sorted_bonds[:num_bonds]

    # Selecting atoms connected by top-N bonds
    # Note that we add 1 to start numeration from 1 to correctly assign labels (RDKit issue)
    bond_atoms = [
        (bondidx2atoms[bidx][0] + 1, bondidx2atoms[bidx][1] + 1)
        for bidx in bonds_to_split
    ]

    frags = Chem.FragmentOnBonds(mol, bonds_to_split, addDummies=True, dummyLabels=bond_atoms)
    frags = Chem.GetMolFrags(frags, asMols=True)

    return frags, bond_atoms


def check_fragments_brics(frags, min_frag_size):
    for frag in frags:
        num_dummy_atoms = len(re.findall(REGEX, Chem.MolToSmiles(frag)))
        if (frag.GetNumAtoms() - num_dummy_atoms) < min_frag_size:
            return False

    return True


def generate_possible_connected_linkers(neighbors):
    candidates = np.where(neighbors.sum(0) > 2)[0]
    possible_linkers = set([
        (candidate,)
        for candidate in candidates
    ])
    return possible_linkers


def generate_possible_2nd_order_linkers(neighbors):
    # Removing edge fragments
    initial_candidates = np.where(neighbors.sum(0) > 1)[0]
    neighbors = neighbors[initial_candidates][:, initial_candidates]

    # Computing 2nd order neighbors and finding all loops
    second_order_neigh = ((neighbors @ neighbors) > 0).astype(int) * (1 - neighbors) - np.eye(neighbors.shape[0])
    candidates = set(np.where(np.diag(second_order_neigh @ second_order_neigh))[0])

    possible_linkers_pairs = set()
    for first_candidate in candidates:
        for second_candidate in set(np.where(second_order_neigh[first_candidate])[0]) & candidates:
            linker_1 = initial_candidates[first_candidate]
            linker_2 = initial_candidates[second_candidate]
            if linker_1 != linker_2:
                possible_linkers_pairs.add(tuple(sorted([linker_1, linker_2])))

    return possible_linkers_pairs


def generate_possible_3nd_order_linkers(neighbors):
    # Removing edge fragments
    initial_candidates = np.where(neighbors.sum(0) > 1)[0]
    neighbors = neighbors[initial_candidates][:, initial_candidates]

    # Computing 3rd order neighbors and finding all loops
    third_order_neigh = ((neighbors @ neighbors @ neighbors) > 0).astype(int)
    third_order_neigh = third_order_neigh * (1 - neighbors) - np.eye(neighbors.shape[0])
    candidates = set(np.where(np.diag(third_order_neigh @ third_order_neigh @ third_order_neigh))[0])

    possible_linkers_triples = set()
    for first_candidate in candidates:
        rest_candidates = candidates.difference({first_candidate})
        rest_candidates = set(np.where(third_order_neigh[first_candidate])[0]) & rest_candidates
        for second_candidate in rest_candidates:
            remainders = rest_candidates.difference({second_candidate})
            for third_candidate in remainders:
                linker_1 = initial_candidates[first_candidate]
                linker_2 = initial_candidates[second_candidate]
                linker_3 = initial_candidates[third_candidate]
                if linker_1 != linker_2 and linker_1 != linker_3 and linker_2 != linker_3:
                    possible_linkers_triples.add(tuple(sorted([linker_1, linker_2, linker_3])))

    return possible_linkers_triples


def fragment_by_brics(smiles, min_frag_size, num_frags, linker_type=None):
    mol = Chem.MolFromSmiles(smiles)

    # Finding all BRICS bonds
    bonds = [bond[0] for bond in FindBRICSBonds(mol)]
    if len(bonds) == 0:
        return []

    # Splitting molecule into fragments
    frags, bond_atoms = split_into_n_fragments(mol, bonds, num_frags)
    if not check_fragments_brics(frags, min_frag_size):
        return []

    # Building mapping between fragments and connecting atoms
    atom2frag = {}
    for i, frag in enumerate(frags):
        matches = re.findall(REGEX, Chem.MolToSmiles(frag))
        for match in matches:
            atom = int(match[1:-2])
            atom2frag[atom] = i

    # Creating adjacency matrix
    neighbors = np.zeros((len(frags), len(frags)))
    for atom1, atom2 in bond_atoms:
        neighbors[atom2frag[atom1], atom2frag[atom2]] = 1
        neighbors[atom2frag[atom2], atom2frag[atom1]] = 1

    # Generating possible linkers
    if linker_type is None:
        possible_linkers = []
        possible_linkers += generate_possible_connected_linkers(neighbors)
        possible_linkers += generate_possible_2nd_order_linkers(neighbors)
        possible_linkers += generate_possible_3nd_order_linkers(neighbors)
    elif linker_type == 1:
        possible_linkers = generate_possible_connected_linkers(neighbors)
    elif linker_type == 2:
        possible_linkers = generate_possible_2nd_order_linkers(neighbors)
    elif linker_type == 3:
        possible_linkers = generate_possible_3nd_order_linkers(neighbors)
    else:
        raise NotImplementedError

    # Formatting the results
    results = []
    for linkers in possible_linkers:
        linkers_smi = ''
        fragments_smi = ''
        for i in range(len(frags)):
            if i in linkers:
                linkers_smi += Chem.MolToSmiles(frags[i]) + '.'
            else:
                fragments_smi += Chem.MolToSmiles(frags[i]) + '.'

        linkers_smi = linkers_smi[:-1]
        fragments_smi = fragments_smi[:-1]
        results.append([smiles, linkers_smi, fragments_smi, 'brics'])

    return results


# #################################################################################### #
# ####################################### MMPA ####################################### #
# #################################################################################### #

def check_mmpa_linker(linker_smi, min_size):
    mol = Chem.MolFromSmiles(linker_smi)
    num_exits = linker_smi.count('*:')
    return (mol.GetNumAtoms() - num_exits) >= min_size


def check_mmpa_fragment(fragment_smi, min_size):
    mol = Chem.MolFromSmiles(fragment_smi)
    num_exits = fragment_smi.count('*:')
    return (mol.GetNumAtoms() - num_exits) >= min_size


def check_mmpa_fragments(fragments_smi, min_size):
    for fragment_smi in fragments_smi.split('.'):
        if not check_mmpa_fragment(fragment_smi, min_size):
            return False
    return True


def fragment_by_mmpa(smiles, min_cuts, max_cuts, min_frag_size):
    mmpa_results = []
    for i in range(min_cuts, max_cuts + 1):
        mmpa_results += FragmentMol(
            Chem.MolFromSmiles(smiles),
            minCuts=i,
            maxCuts=i,
            maxCutBonds=100,
            pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]",
            resultsAsMols=False
        )

    filtered_mmpa_results = []
    for linker_smiles, fragments_smiles in mmpa_results:
        if check_mmpa_linker(linker_smiles, min_frag_size) and check_mmpa_fragments(fragments_smiles, min_frag_size):
            filtered_mmpa_results.append([smiles, linker_smiles, fragments_smiles, 'mmpa'])
    return filtered_mmpa_results


# #################################################################################### #
# ####################################### Main ####################################### #
# #################################################################################### #

def run(geom_json_path, output_path, min_size, mmpa, brics):
    with open(geom_json_path) as f:
        geom_json = json.load(f)

    all_smiles = list(geom_json.keys())
    mol_results = []

    for i, smiles in tqdm(enumerate(all_smiles), total=len(all_smiles)):
        if mmpa:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule.GetNumAtoms() <= 40 and molecule.GetRingInfo().NumRings() >= 3:
                try:
                    mol_results += fragment_by_mmpa(smiles, min_cuts=3, max_cuts=5, min_frag_size=min_size)
                except Exception as e:
                    print(f'Error [MMPA] with {smiles}: {e}')
        if brics:
            for num_frags in [4, 5, 6, 7]:
                try:
                    mol_results += fragment_by_brics(
                        smiles=smiles,
                        num_frags=num_frags,
                        min_frag_size=min_size,
                        linker_type=None
                    )
                except Exception as e:
                    print(f'Error [BRICS] with {smiles}: {e}')

        if (i + 1) % 5000 == 0:
            table = pd.DataFrame(mol_results, columns=['molecule', 'linker', 'fragments', 'method'])
            table = table.drop_duplicates(['molecule', 'linker'])
            table.to_csv(output_path, index=False)

    table = pd.DataFrame(mol_results, columns=['molecule', 'linker', 'fragments', 'method'])
    table = table.drop_duplicates(['molecule', 'linker'])
    table.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geom-json', action='store', type=str, required=True)
    parser.add_argument('--output', action='store', type=str, required=True)
    parser.add_argument('--min-size', action='store', type=int, required=True)
    parser.add_argument('--mmpa', action='store_true', default=False)
    parser.add_argument('--brics', action='store_true', default=False)
    args = parser.parse_args()

    run(
        geom_json_path=args.geom_json,
        output_path=args.output,
        min_size=args.min_size,
        mmpa=args.mmpa,
        brics=args.brics,
    )
