import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from src import const
from src.molecule_builder import get_bond_order
from scipy.stats import wasserstein_distance

from pdb import set_trace


def is_valid(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False
    return True


def is_connected(mol):
    try:
        mol_frags = Chem.GetMolFrags(mol, asMols=True)
    except Chem.rdchem.AtomValenceException:
        return False
    if len(mol_frags) != 1:
        return False
    return True


def get_valid_molecules(molecules):
    valid = []
    for mol in molecules:
        if is_valid(mol):
            valid.append(mol)
    return valid


def get_connected_molecules(molecules):
    connected = []
    for mol in molecules:
        if is_connected(mol):
            connected.append(mol)
    return connected


def get_unique_smiles(valid_molecules):
    unique = set()
    for mol in valid_molecules:
        unique.add(Chem.MolToSmiles(mol))
    return list(unique)


def get_novel_smiles(unique_true_smiles, unique_pred_smiles):
    return list(set(unique_pred_smiles).difference(set(unique_true_smiles)))


def compute_energy(mol):
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    energy = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=0).CalcEnergy()
    return energy


def wasserstein_distance_between_energies(true_molecules, pred_molecules):
    true_energy_dist = []
    for mol in true_molecules:
        try:
            energy = compute_energy(mol)
            true_energy_dist.append(energy)
        except:
            continue

    pred_energy_dist = []
    for mol in pred_molecules:
        try:
            energy = compute_energy(mol)
            pred_energy_dist.append(energy)
        except:
            continue

    if len(true_energy_dist) > 0 and len(pred_energy_dist) > 0:
        return wasserstein_distance(true_energy_dist, pred_energy_dist)
    else:
        return 0


def compute_metrics(pred_molecules, true_molecules):
    if len(pred_molecules) == 0:
        return {
            'validity': 0,
            'validity_and_connectivity': 0,
            'validity_as_in_delinker': 0,
            'uniqueness': 0,
            'novelty': 0,
            'energies': 0,
        }

    # Passing rdkit.Chem.Sanitize filter
    true_valid = get_valid_molecules(true_molecules)
    pred_valid = get_valid_molecules(pred_molecules)
    validity = len(pred_valid) / len(pred_molecules)

    # Checking if molecule consists of a single connected part
    true_valid_and_connected = get_connected_molecules(true_valid)
    pred_valid_and_connected = get_connected_molecules(pred_valid)
    validity_and_connectivity = len(pred_valid_and_connected) / len(pred_molecules)

    # Unique molecules
    true_unique = get_unique_smiles(true_valid_and_connected)
    pred_unique = get_unique_smiles(pred_valid_and_connected)
    uniqueness = len(pred_unique) / len(pred_valid_and_connected) if len(pred_valid_and_connected) > 0 else 0

    # Novel molecules
    pred_novel = get_novel_smiles(true_unique, pred_unique)
    novelty = len(pred_novel) / len(pred_unique) if len(pred_unique) > 0 else 0

    # Difference between Energy distributions
    energies = wasserstein_distance_between_energies(true_valid_and_connected, pred_valid_and_connected)

    return {
        'validity': validity,
        'validity_and_connectivity': validity_and_connectivity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'energies': energies,
    }


# def check_stability(positions, atom_types):
#     assert len(positions.shape) == 2
#     assert positions.shape[1] == 3
#     x = positions[:, 0]
#     y = positions[:, 1]
#     z = positions[:, 2]
#
#     nr_bonds = np.zeros(len(x), dtype='int')
#     for i in range(len(x)):
#         for j in range(i + 1, len(x)):
#             p1 = np.array([x[i], y[i], z[i]])
#             p2 = np.array([x[j], y[j], z[j]])
#             dist = np.sqrt(np.sum((p1 - p2) ** 2))
#             atom1, atom2 = const.IDX2ATOM[atom_types[i].item()], const.IDX2ATOM[atom_types[j].item()]
#             order = get_bond_order(atom1, atom2, dist)
#             nr_bonds[i] += order
#             nr_bonds[j] += order
#     nr_stable_bonds = 0
#     for atom_type_i, nr_bonds_i in zip(atom_types, nr_bonds):
#         possible_bonds = const.ALLOWED_BONDS[const.IDX2ATOM[atom_type_i.item()]]
#         if type(possible_bonds) == int:
#             is_stable = possible_bonds == nr_bonds_i
#         else:
#             is_stable = nr_bonds_i in possible_bonds
#         nr_stable_bonds += int(is_stable)
#
#     molecule_stable = nr_stable_bonds == len(x)
#     return molecule_stable, nr_stable_bonds, len(x)
#
#
# def count_stable_molecules(one_hot, x, node_mask):
#     stable_molecules = 0
#     for i in range(len(one_hot)):
#         mol_size = node_mask[i].sum()
#         atom_types = one_hot[i][:mol_size, :].argmax(dim=1).detach().cpu()
#         positions = x[i][:mol_size, :].detach().cpu()
#         stable, _, _ = check_stability(positions, atom_types)
#         stable_molecules += int(stable)
#
#     return stable_molecules
