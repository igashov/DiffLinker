import argparse
import os
import numpy as np

import torch
import subprocess
from rdkit import Chem
from Bio.PDB import PDBParser

from src import const
from src.datasets import (
    collate_with_fragment_without_pocket_edges, get_dataloader, get_one_hot, parse_molecule, MOADDataset
)
from src.lightning import DDPM
from src.visualizer import save_xyz_file
from src.utils import FoundNaNException, set_deterministic
from tqdm import tqdm

from src.linker_size_lightning import SizeClassifier


from pdb import set_trace


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fragments', action='store', type=str, required=True,
    help='Path to the file with input fragments'
)
parser.add_argument(
    '--protein', action='store', type=str, required=True,
    help='Path to the file with the target protein'
)
parser.add_argument(
    '--backbone_atoms_only', action='store_true', required=False, default=False,
    help='Flag if to use only protein backbone atoms'
)
parser.add_argument(
    '--model', action='store', type=str, required=True,
    help='Path to the DiffLinker model'
)
parser.add_argument(
    '--linker_size', action='store', type=str, required=True,
    help='Linker size (int) or allowed size boundaries (comma-separated integers) or path to the size prediction model'
)
parser.add_argument(
    '--output', action='store', type=str, required=False, default='./',
    help='Directory where sampled molecules will be saved'
)
parser.add_argument(
    '--n_samples', action='store', type=int, required=False, default=5,
    help='Number of linkers to generate'
)
parser.add_argument(
    '--n_steps', action='store', type=int, required=False, default=None,
    help='Number of denoising steps'
)
parser.add_argument(
    '--anchors', action='store', type=str, required=False, default=None,
    help='Comma-separated indices of anchor atoms '
         '(according to the order of atoms in the input fragments file, enumeration starts with 1)'
)
parser.add_argument(
    '--max_batch_size', action='store', type=int, required=False, default=64,
    help='Max batch size'
)
parser.add_argument(
    '--random_seed', action='store', type=int, required=False, default=None,
    help='Random seed'
)


def read_molecule(path):
    if path.endswith('.pdb'):
        return Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol'):
        return Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol2'):
        return Chem.MolFromMol2File(path, sanitize=False, removeHs=True)
    elif path.endswith('.sdf'):
        return Chem.SDMolSupplier(path, sanitize=False, removeHs=True)[0]
    raise Exception('Unknown file extension')


def get_pocket(mol, pdb_path, backbone_atoms_only=False):
    struct = PDBParser().get_structure('', pdb_path)
    residue_ids = []
    atom_coords = []

    for residue in struct.get_residues():
        resid = residue.get_id()[1]
        for atom in residue.get_atoms():
            atom_coords.append(atom.get_coord())
            residue_ids.append(resid)

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)
    mol_atom_coords = mol.GetConformer().GetPositions()

    distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    contact_residues = np.unique(residue_ids[np.where(distances.min(1) <= 6)[0]])

    pocket_coords_full = []
    pocket_types_full = []

    pocket_coords_bb = []
    pocket_types_bb = []

    for residue in struct.get_residues():
        resid = residue.get_id()[1]
        if resid not in contact_residues:
            continue

        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            atom_type = atom.element.upper()
            atom_coord = atom.get_coord()

            pocket_coords_full.append(atom_coord.tolist())
            pocket_types_full.append(atom_type)

            if atom_name in {'N', 'CA', 'C', 'O'}:
                pocket_coords_bb.append(atom_coord.tolist())
                pocket_types_bb.append(atom_type)

    pocket_pos = []
    pocket_one_hot = []
    pocket_charges = []

    generator = (
        zip(pocket_coords_bb, pocket_types_bb)
        if backbone_atoms_only
        else zip(pocket_coords_full, pocket_types_full)
    )
    for coord, atom_type in generator:
        if atom_type not in const.GEOM_ATOM2IDX.keys():
            continue

        pocket_pos.append(coord)
        pocket_one_hot.append(get_one_hot(atom_type, const.GEOM_ATOM2IDX))
        pocket_charges.append(const.GEOM_CHARGES[atom_type])

    pocket_pos = np.array(pocket_pos)
    pocket_one_hot = np.array(pocket_one_hot)
    pocket_charges = np.array(pocket_charges)

    return pocket_pos, pocket_one_hot, pocket_charges


def main(input_path, protein_path, backbone_atoms_only, model,
         output_dir, n_samples, n_steps, linker_size, anchors, max_batch_size, random_seed):

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    if random_seed is not None:
        set_deterministic(random_seed)

    if linker_size.isdigit():
        print(f'Will generate linkers with {linker_size} atoms')
        linker_size = int(linker_size)

        def sample_fn(_data):
            return torch.ones(_data['positions'].shape[0], device=device, dtype=const.TORCH_INT) * linker_size

    else:
        boundaries = [x.strip() for x in linker_size.split(',')]
        if len(boundaries) == 2 and boundaries[0].isdigit() and boundaries[1].isdigit():
            left = int(boundaries[0])
            right = int(boundaries[1])
            print(f'Will generate linkers with numbers of atoms sampled from U({left}, {right})')

            def sample_fn(_data):
                shape = len(_data['positions']),
                return torch.randint(left, right + 1, shape, device=device, dtype=const.TORCH_INT)

        else:
            print(f'Will generate linkers with sampled numbers of atoms')
            size_nn = SizeClassifier.load_from_checkpoint(linker_size, map_location=device).eval().to(device)

            def sample_fn(_data):
                out, _ = size_nn.forward(_data, return_loss=False, with_pocket=True, adjust_shape=True)
                probabilities = torch.softmax(out, dim=1)
                distribution = torch.distributions.Categorical(probs=probabilities)
                samples = distribution.sample()
                sizes = []
                for label in samples.detach().cpu().numpy():
                    sizes.append(size_nn.linker_id2size[label])
                sizes = torch.tensor(sizes, device=samples.device, dtype=const.TORCH_INT)
                return sizes

    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)

    if n_steps is not None:
        ddpm.edm.T = n_steps

    if ddpm.center_of_mass == 'anchors' and anchors is None:
        print(
            'Please pass anchor atoms indices '
            'or use another DiffLinker model that does not require information about anchors'
        )
        return

    # Reading input fragments
    extension = input_path.split('.')[-1]
    if extension not in ['sdf', 'pdb', 'mol', 'mol2']:
        print('Please upload the fragments file in one of the following formats: .pdb, .sdf, .mol, .mol2')
        return

    protein_extension = protein_path.split('.')[-1]
    if protein_extension != 'pdb':
        print('Please upload the protein file in .pdb format')
        return

    try:
        molecule = read_molecule(input_path)
        molecule = Chem.RemoveAllHs(molecule)
        name = '.'.join(input_path.split('/')[-1].split('.')[:-1])
    except Exception as e:
        return f'Could not read the file with fragments: {e}'

    # Parsing fragments data
    frag_pos, frag_one_hot, frag_charges = parse_molecule(molecule, is_geom=ddpm.is_geom)

    # Parsing pocket data
    try:
        pocket_pos, pocket_one_hot, pocket_charges = get_pocket(molecule, protein_path, backbone_atoms_only)
    except Exception as e:
        return f'Could not read the file with pocket: {e}'

    positions = np.concatenate([frag_pos, pocket_pos], axis=0)
    one_hot = np.concatenate([frag_one_hot, pocket_one_hot], axis=0)
    charges = np.concatenate([frag_charges, pocket_charges], axis=0)
    anchor_flags = np.zeros_like(charges)
    if anchors is not None:
        for anchor in anchors.split(','):
            anchor_flags[int(anchor.strip()) - 1] = 1

    fragment_only_mask = np.concatenate([
        np.ones_like(frag_charges),
        np.zeros_like(pocket_charges),
    ])
    pocket_mask = np.concatenate([
        np.zeros_like(frag_charges),
        np.ones_like(pocket_charges),
    ])
    linker_mask = np.concatenate([
        np.zeros_like(frag_charges),
        np.zeros_like(pocket_charges),
    ])
    fragment_mask = np.concatenate([
        np.ones_like(frag_charges),
        np.ones_like(pocket_charges),
    ])

    dataset = [{
        'uuid': '0',
        'name': '0',
        'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
        'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
        'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
        'anchors': torch.tensor(anchor_flags, dtype=const.TORCH_FLOAT, device=device),
        'fragment_only_mask': torch.tensor(fragment_only_mask, dtype=const.TORCH_FLOAT, device=device),
        'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
        'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
        'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
        'num_atoms': len(positions),
    }] * n_samples
    dataset = MOADDataset(data=dataset)
    ddpm.val_dataset = dataset

    global_batch_size = min(n_samples, max_batch_size)
    dataloader = get_dataloader(
        dataset, batch_size=global_batch_size, collate_fn=collate_with_fragment_without_pocket_edges
    )

    # Sampling
    print('Sampling...')
    for batch_i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_size = len(data['positions'])

        chain = None
        for i in range(5):
            try:
                chain, node_mask = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=1)
                break
            except FoundNaNException:
                continue
        if chain is None:
            raise Exception('Could not generate in 5 attempts')

        x = chain[0][:, :, :ddpm.n_dims]
        h = chain[0][:, :, ddpm.n_dims:]

        # Put the molecule back to the initial orientation
        com_mask = data['fragment_only_mask'] if ddpm.center_of_mass == 'fragments' else data['anchors']
        pos_masked = data['positions'] * com_mask
        N = com_mask.sum(1, keepdims=True)
        mean = torch.sum(pos_masked, dim=1, keepdim=True) / N
        x = x + mean * node_mask

        offset_idx = batch_i * global_batch_size
        names = [f'output_{offset_idx+i}_{name}' for i in range(batch_size)]

        node_mask[torch.where(data['pocket_mask'])] = 0
        save_xyz_file(output_dir, h, x, node_mask, names=names, is_geom=ddpm.is_geom, suffix='')

        for i in range(batch_size):
            out_xyz = f'{output_dir}/output_{offset_idx+i}_{name}_.xyz'
            out_sdf = f'{output_dir}/output_{offset_idx+i}_{name}_.sdf'
            subprocess.run(f'obabel {out_xyz} -O {out_sdf} 2> /dev/null', shell=True)

    print(f'Saved generated molecules in .xyz and .sdf format in directory {output_dir}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        input_path=args.fragments,
        protein_path=args.protein,
        backbone_atoms_only=args.backbone_atoms_only,
        model=args.model,
        output_dir=args.output,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        linker_size=args.linker_size,
        anchors=args.anchors,
        max_batch_size=args.max_batch_size,
        random_seed=args.random_seed,
    )

