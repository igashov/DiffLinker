import argparse
import itertools
import numpy as np
import pandas as pd

from rdkit import Chem, Geometry
from tqdm import tqdm

from pdb import set_trace


def get_exits(mol):
    """
    Returns atoms marked as exits in DeLinker data
    """
    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits


def set_anchor_flags(mol, anchor_idx):
    """
    Sets property _Anchor to all atoms in a molecule
    """
    for atom in mol.GetAtoms():
        if atom.GetIdx() == anchor_idx:
            atom.SetProp('_Anchor', '1')
        else:
            atom.SetProp('_Anchor', '0')


def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        if atom.GetProp('_Anchor') == '1':
            anchors_idx.append(atom.GetIdx())

    return anchors_idx


def update_fragment(frag):
    """
    Removes exit atoms with corresponding bonds and sets _Anchor property
    """
    exits = get_exits(frag)
    if len(exits) > 1:
        raise Exception('Found more than one exits in fragment')
    exit = exits[0]

    bonds = exit.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]

    exit_idx = exit.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    set_anchor_flags(frag, anchor_idx)

    efragment = Chem.EditableMol(frag)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(exit_idx)

    return efragment.GetMol()


def update_linker(linker):
    """
    Removes exit atoms with corresponding bonds
    """
    exits = get_exits(linker)
    if len(exits) > 2:
        raise Exception('Found more than two exits in linker')

    # Sort exit atoms by id for further correct deletion
    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)
    elinker = Chem.EditableMol(linker)

    # Remove exit bonds
    for exit in exits:
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        elinker.RemoveBond(source_idx, target_idx)

    # Remove exit atoms
    for exit in exits:
        elinker.RemoveAtom(exit.GetIdx())

    return elinker.GetMol()


def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer


def transfer_conformers(frag, mol):
    """
    Computes coordinates from molecule to fragment (for all matchings)
    """
    matches = mol.GetSubstructMatches(frag)
    if len(matches) < 1:
        raise Exception('Could not find fragment or linker matches')

    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf[match] = frag_conformer

    return match2conf


def find_non_intersecting_matches(matches1, matches2, matches3):
    """
    Checks all possible triplets and selects only non-intersecting matches
    """
    triplets = list(itertools.product(matches1, matches2, matches3))
    non_intersecting_matches = set()
    for m1, m2, m3 in triplets:
        m1m2 = set(m1) & set(m2)
        m1m3 = set(m1) & set(m3)
        m2m3 = set(m2) & set(m3)
        if len(m1m2 | m1m3 | m2m3) == 0:
            non_intersecting_matches.add((m1, m2, m3))
    return list(non_intersecting_matches)


def find_matches_with_linker_in_the_middle(non_intersecting_matches, mol):
    """
    Selects only matches where linker is between fragments
    I.e. each fragment should have one atom that is included in the set of neighbors of all linker atoms
    """
    matches_with_linker_in_the_middle = []
    for m1, m2, lm in non_intersecting_matches:
        neighbors = set()
        for atom_idx in lm:
            atom_neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            for neighbor in atom_neighbors:
                neighbors.add(neighbor.GetIdx())

        conn1 = set(m1) & neighbors
        conn2 = set(m2) & neighbors
        if len(conn1) == 1 and len(conn2) == 1:
            matches_with_linker_in_the_middle.append((m1, m2, lm))

    return matches_with_linker_in_the_middle


def find_correct_matches(matches_frag1, matches_frag2, matches_linker, mol):
    """
    Finds all correct fragments and linker matches
    """
    non_intersecting_matches = find_non_intersecting_matches(matches_frag1, matches_frag2, matches_linker)
    if len(non_intersecting_matches) == 1:
        return non_intersecting_matches

    return find_matches_with_linker_in_the_middle(non_intersecting_matches, mol)


def prepare_fragments_and_linker(frags_smi, linker_smi, mol):
    """
    Given a molecule and SMILES string of fragments from DeLinker data,
    creates fragment and linker conformers according to the molecule conformer,
    removes exit atoms and sets _Anchor property to all fragment atoms
    """

    frag1_smi, frag2_smi = frags_smi.split('.')
    frag1 = Chem.MolFromSmiles(frag1_smi)
    frag2 = Chem.MolFromSmiles(frag2_smi)
    linker = Chem.MolFromSmiles(linker_smi)

    newfrag1 = update_fragment(frag1)
    newfrag2 = update_fragment(frag2)
    newlinker = update_linker(linker)

    match2conf_frag1 = transfer_conformers(newfrag1, mol)
    match2conf_frag2 = transfer_conformers(newfrag2, mol)
    match2conf_linker = transfer_conformers(newlinker, mol)

    correct_matches = find_correct_matches(
        match2conf_frag1.keys(),
        match2conf_frag2.keys(),
        match2conf_linker.keys(),
        mol,
    )

    # Expecting to see either a single correct match or two correct matches
    # If the molecule is symmetric then both matches are equivalent, and we can use any
    if len(correct_matches) > 2:
        raise Exception('Found more than two fragment matches')

    conf_frag1 = match2conf_frag1[correct_matches[0][0]]
    conf_frag2 = match2conf_frag2[correct_matches[0][1]]
    conf_linker = match2conf_linker[correct_matches[0][2]]
    newfrag1.AddConformer(conf_frag1)
    newfrag2.AddConformer(conf_frag2)
    newlinker.AddConformer(conf_linker)

    return newfrag1, newfrag2, newlinker


def process_sdf(sdf_path, table, progress=True):
    supplier = Chem.SDMolSupplier(sdf_path)
    molecules = []
    fragments = []
    linkers = []
    out_table = []
    uuid = 0

    supplier = tqdm(supplier, total=len(supplier)) if progress else supplier
    for mol in supplier:
        mol_name = mol.GetProp('_Name')
        mol_smi = Chem.MolToSmiles(mol)
        mol.SetProp('_Name', mol_smi)
        for linker_smi, frags_smi in table[table.molecule == mol_name][['linker', 'fragments']].values:
            try:
                frag1, frag2, linker = prepare_fragments_and_linker(frags_smi, linker_smi, mol)
            except Exception as e:
                print(f'{mol_smi} | {linker_smi} | {frags_smi} : {e}')
                continue

            frags = Chem.CombineMols(frag1, frag2)
            anchors_idx = get_anchors_idx(frags)
            if len(anchors_idx) != 2:
                print(f'{mol_smi} | {linker_smi} | {frags_smi} : found {len(anchors_idx)} anchors')
                continue

            molecules.append(mol)
            fragments.append(frags)
            linkers.append(linker)

            out_table.append({
                'uuid': uuid,
                'molecule': mol_smi,
                'fragments': Chem.MolToSmiles(frags),
                'linker': Chem.MolToSmiles(linker),
                'anchor_1': anchors_idx[0],
                'anchor_2': anchors_idx[1],
                'energy': mol.GetProp('_Energy'),
            })
            uuid += 1

    return molecules, fragments, linkers, pd.DataFrame(out_table)


def run(table_path, sdf_path, out_mol_path, out_frag_path, out_link_path, out_table_path, progress=True):
    table = pd.read_csv(table_path, sep=' ', names=['molecule', 'linker', 'fragments', 'distance', 'angle'])
    molecules, fragments, linkers, out_table = process_sdf(sdf_path, table, progress)

    out_table.to_csv(out_table_path, index=False)
    with Chem.SDWriter(open(out_mol_path, 'w')) as writer:
        for mol in molecules:
            writer.write(mol)
    with Chem.SDWriter(open(out_frag_path, 'w')) as writer:
        writer.SetKekulize(False)
        for frags in fragments:
            writer.write(frags)
    with Chem.SDWriter(open(out_link_path, 'w')) as writer:
        writer.SetKekulize(False)
        for linker in linkers:
            writer.write(linker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', action='store', type=str, required=True)
    parser.add_argument('--sdf', action='store', type=str, required=True)
    parser.add_argument('--out-mol-sdf', action='store', type=str, required=True)
    parser.add_argument('--out-frag-sdf', action='store', type=str, required=True)
    parser.add_argument('--out-link-sdf', action='store', type=str, required=True)
    parser.add_argument('--out-table', action='store', type=str, required=True)
    args = parser.parse_args()

    run(
        table_path=args.table,
        sdf_path=args.sdf,
        out_mol_path=args.out_mol_sdf,
        out_frag_path=args.out_frag_sdf,
        out_link_path=args.out_link_sdf,
        out_table_path=args.out_table,
    )
