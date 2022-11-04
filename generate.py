import argparse
import os
import numpy as np

import torch
import subprocess
from rdkit import Chem

from src import const
from src.datasets import collate_with_fragment_edges, get_dataloader, parse_molecule
from src.lightning import DDPM
from src.linker_size_lightning import SizeClassifier
from src.visualizer import save_xyz_file
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fragments', action='store', type=str, required=True,
    help='Path to the file with input fragments'
)
parser.add_argument(
    '--model', action='store', type=str, required=True,
    help='Path to the DiffLinker model'
)
parser.add_argument(
    '--linker_size', action='store', type=str, required=True,
    help='Either linker size (int) or path to the GNN for size prediction'
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


def main(input_path, model, output_dir, n_samples, n_steps, linker_size):

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    if linker_size.isdigit():
        print(f'Will generate linkers with {linker_size} atoms')
        linker_size = int(linker_size)

        def sample_fn(_data):
            return torch.ones(_data['positions'].shape[0], device=device, dtype=const.TORCH_INT) * linker_size
    else:
        print(f'Will generate linkers with sampled numbers of atoms')
        size_nn = SizeClassifier.load_from_checkpoint(linker_size, map_location=device).eval().to(device)

        def sample_fn(_data):
            out, _ = size_nn.forward(_data, return_loss=False)
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

    if ddpm.center_of_mass != 'fragments':
        print('Please use another DiffLinker model that does not require information about anchors')
        return

    # Reading input fragments
    extension = input_path.split('.')[-1]
    if extension not in ['sdf', 'pdb', 'mol', 'mol2']:
        print('Please upload the file in one of the following formats: .pdb, .sdf, .mol, .mol2')
        return

    try:
        molecule = read_molecule(input_path)
        molecule = Chem.RemoveAllHs(molecule)
        name = '.'.join(input_path.split('/')[-1].split('.')[:-1])
    except Exception as e:
        return f'Could not read the molecule: {e}'

    positions, one_hot, charges = parse_molecule(molecule, is_geom=ddpm.is_geom)
    anchors = np.zeros_like(charges)
    fragment_mask = np.ones_like(charges)
    linker_mask = np.zeros_like(charges)

    dataset = [{
        'uuid': '0',
        'name': '0',
        'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
        'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
        'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
        'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
        'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
        'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
        'num_atoms': len(positions),
    }] * n_samples
    batch_size = min(n_samples, 64)
    dataloader = get_dataloader(dataset, batch_size=batch_size, collate_fn=collate_with_fragment_edges)

    # Sampling
    print('Sampling...')
    for batch_i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        chain, node_mask = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=1)
        x = chain[0][:, :, :ddpm.n_dims]
        h = chain[0][:, :, ddpm.n_dims:]

        # Put the molecule back to the initial orientation
        pos_masked = data['positions'] * data['fragment_mask']
        N = data['fragment_mask'].sum(1, keepdims=True)
        mean = torch.sum(pos_masked, dim=1, keepdim=True) / N
        x = x + mean * node_mask

        offset_idx = batch_i * batch_size
        names = [f'output_{offset_idx+i}_{name}' for i in range(batch_size)]
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
        model=args.model,
        output_dir=args.output,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        linker_size=args.linker_size,
    )

