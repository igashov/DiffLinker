import csv
import numpy as np

from rdkit import Chem
from rdkit.Chem import MolStandardize
from src import metrics
from src.delinker_utils import sascorer, calc_SC_RDKit
from tqdm import tqdm

from pdb import set_trace


def get_valid_as_in_delinker(data, progress=False):
    valid = []
    generator = tqdm(enumerate(data), total=len(data)) if progress else enumerate(data)
    for i, m in generator:
        pred_mol = Chem.MolFromSmiles(m['pred_mol_smi'], sanitize=False)
        true_mol = Chem.MolFromSmiles(m['true_mol_smi'], sanitize=False)
        frag = Chem.MolFromSmiles(m['frag_smi'], sanitize=False)

        pred_mol_frags = Chem.GetMolFrags(pred_mol, asMols=True, sanitizeFrags=False)
        pred_mol_filtered = max(pred_mol_frags, default=pred_mol, key=lambda mol: mol.GetNumAtoms())

        try:
            Chem.SanitizeMol(pred_mol_filtered)
            Chem.SanitizeMol(true_mol)
            Chem.SanitizeMol(frag)
        except:
            continue

        if len(pred_mol_filtered.GetSubstructMatch(frag)) > 0:
            valid.append({
                'pred_mol': m['pred_mol'],
                'true_mol': m['true_mol'],
                'pred_mol_smi': Chem.MolToSmiles(pred_mol_filtered),
                'true_mol_smi': Chem.MolToSmiles(true_mol),
                'frag_smi': Chem.MolToSmiles(frag)
            })

    return valid


def extract_linker_smiles(molecule, fragments):
    match = molecule.GetSubstructMatch(fragments)
    elinker = Chem.EditableMol(molecule)
    for atom_id in sorted(match, reverse=True):
        elinker.RemoveAtom(atom_id)
    linker = elinker.GetMol()
    Chem.RemoveStereochemistry(linker)
    try:
        linker = MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(linker))
    except:
        linker = Chem.MolToSmiles(linker)
    return linker


def compute_and_add_linker_smiles(data, progress=False):
    data_with_linkers = []
    generator = tqdm(data) if progress else data
    for m in generator:
        pred_mol = Chem.MolFromSmiles(m['pred_mol_smi'], sanitize=True)
        true_mol = Chem.MolFromSmiles(m['true_mol_smi'], sanitize=True)
        frag = Chem.MolFromSmiles(m['frag_smi'], sanitize=True)

        pred_linker = extract_linker_smiles(pred_mol, frag)
        true_linker = extract_linker_smiles(true_mol, frag)
        data_with_linkers.append({
            **m,
            'pred_linker': pred_linker,
            'true_linker': true_linker,
        })

    return data_with_linkers


def compute_uniqueness(data, progress=False):
    mol_dictionary = {}
    generator = tqdm(data) if progress else data
    for m in generator:
        frag = m['frag_smi']
        pred_mol = m['pred_mol_smi']
        true_mol = m['true_mol_smi']

        key = f'{true_mol}.{frag}'
        mol_dictionary.setdefault(key, []).append(pred_mol)

    total_mol = 0
    unique_mol = 0
    for molecules in mol_dictionary.values():
        total_mol += len(molecules)
        unique_mol += len(set(molecules))

    return unique_mol / total_mol


def compute_novelty(data, progress=False):
    novel = 0
    true_linkers = set([m['true_linker'] for m in data])
    generator = tqdm(data) if progress else data
    for m in generator:
        pred_linker = m['pred_linker']
        if pred_linker in true_linkers:
            continue
        else:
            novel += 1

    return novel / len(data)


def compute_recovery_rate(data, progress=False):
    total = set()
    recovered = set()
    generator = tqdm(data) if progress else data
    for m in generator:
        pred_mol = Chem.MolFromSmiles(m['pred_mol_smi'], sanitize=True)
        Chem.RemoveStereochemistry(pred_mol)
        pred_mol = Chem.MolToSmiles(Chem.RemoveHs(pred_mol))

        true_mol = Chem.MolFromSmiles(m['true_mol_smi'], sanitize=True)
        Chem.RemoveStereochemistry(true_mol)
        true_mol = Chem.MolToSmiles(Chem.RemoveHs(true_mol))

        true_link = m['true_linker']
        total.add(f'{true_mol}.{true_link}')
        if pred_mol == true_mol:
            recovered.add(f'{true_mol}.{true_link}')

    return len(recovered) / len(total)


def calc_sa_score_mol(mol):
    if mol is None:
        return None
    return sascorer.calculateScore(mol)


def check_ring_filter(linker):
    check = True
    # Get linker rings
    ssr = Chem.GetSymmSSSR(linker)
    # Check rings
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring:
                    check = False
    return check


def check_pains(mol, pains_smarts):
    for pain in pains_smarts:
        if mol.HasSubstructMatch(pain):
            return False
    return True


def calc_2d_filters(toks, pains_smarts):
    pred_mol = Chem.MolFromSmiles(toks['pred_mol_smi'])
    frag = Chem.MolFromSmiles(toks['frag_smi'])
    linker = Chem.MolFromSmiles(toks['pred_linker'])

    result = [False, False, False]
    if len(pred_mol.GetSubstructMatch(frag)) > 0:
        sa_score = False
        ra_score = False
        pains_score = False

        try:
            sa_score = calc_sa_score_mol(pred_mol) < calc_sa_score_mol(frag)
        except Exception as e:
            print(f'Could not compute SA score: {e}')
        try:
            ra_score = check_ring_filter(linker)
        except Exception as e:
            print(f'Could not compute RA score: {e}')
        try:
            pains_score = check_pains(pred_mol, pains_smarts)
        except Exception as e:
            print(f'Could not compute PAINS score: {e}')

        result = [sa_score, ra_score, pains_score]

    return result


def calc_filters_2d_dataset(data):
    with open('resources/wehi_pains.csv', 'r') as f:
        pains_smarts = [Chem.MolFromSmarts(line[0], mergeHs=True) for line in csv.reader(f)]

    pass_all = pass_SA = pass_RA = pass_PAINS = 0
    for m in data:
        filters_2d = calc_2d_filters(m, pains_smarts)
        pass_all += filters_2d[0] & filters_2d[1] & filters_2d[2]
        pass_SA += filters_2d[0]
        pass_RA += filters_2d[1]
        pass_PAINS += filters_2d[2]

    return pass_all / len(data), pass_SA / len(data), pass_RA / len(data), pass_PAINS / len(data)


def calc_sc_rdkit_full_mol(gen_mol, ref_mol):
    try:
        score = calc_SC_RDKit.calc_SC_RDKit_score(gen_mol, ref_mol)
        return score
    except:
        return -0.5


def sc_rdkit_score(data):
    scores = []
    for m in data:
        score = calc_sc_rdkit_full_mol(m['pred_mol'], m['true_mol'])
        scores.append(score)

    return np.mean(scores)


def get_delinker_metrics(pred_molecules, true_molecules, true_fragments):
    default_values = {
        'DeLinker/validity': 0,
        'DeLinker/uniqueness': 0,
        'DeLinker/novelty': 0,
        'DeLinker/recovery': 0,
        'DeLinker/2D_filters': 0,
        'DeLinker/2D_filters_SA': 0,
        'DeLinker/2D_filters_RA': 0,
        'DeLinker/2D_filters_PAINS': 0,
        'DeLinker/SC_RDKit': 0,
    }
    if len(pred_molecules) == 0:
        return default_values

    data = []
    for pred_mol, true_mol, true_frag in zip(pred_molecules, true_molecules, true_fragments):
        data.append({
            'pred_mol': pred_mol,
            'true_mol': true_mol,
            'pred_mol_smi': Chem.MolToSmiles(pred_mol),
            'true_mol_smi': Chem.MolToSmiles(true_mol),
            'frag_smi': Chem.MolToSmiles(true_frag)
        })

    # Validity according to DeLinker paper:
    # Passing rdkit.Chem.Sanitize and the biggest fragment contains both fragments
    valid_data = get_valid_as_in_delinker(data)
    validity_as_in_delinker = len(valid_data) / len(data)
    if len(valid_data) == 0:
        return default_values

    # Compute linkers and add to results
    valid_data = compute_and_add_linker_smiles(valid_data)

    # Compute uniqueness
    uniqueness = compute_uniqueness(valid_data)

    # Compute novelty
    novelty = compute_novelty(valid_data)

    # Compute recovered molecules
    recovery_rate = compute_recovery_rate(valid_data)

    # 2D filters
    pass_all, pass_SA, pass_RA, pass_PAINS = calc_filters_2d_dataset(valid_data)

    # 3D Filters
    sc_rdkit = sc_rdkit_score(valid_data)

    return {
        'DeLinker/validity': validity_as_in_delinker,
        'DeLinker/uniqueness': uniqueness,
        'DeLinker/novelty': novelty,
        'DeLinker/recovery': recovery_rate,
        'DeLinker/2D_filters': pass_all,
        'DeLinker/2D_filters_SA': pass_SA,
        'DeLinker/2D_filters_RA': pass_RA,
        'DeLinker/2D_filters_PAINS': pass_PAINS,
        'DeLinker/SC_RDKit': sc_rdkit,
    }
