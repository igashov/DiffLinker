import argparse
import os

import torch

from src import utils
from src.lightning import DDPM
from src.linker_size_lightning import SizeClassifier
from src.visualizer import save_xyz_file
from src.datasets import collate, collate_with_fragment_edges, MOADDataset
from tqdm import tqdm

from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store', type=str, required=True)
parser.add_argument('--samples', action='store', type=str, required=True)
parser.add_argument('--data', action='store', type=str, required=False, default=None)
parser.add_argument('--prefix', action='store', type=str, required=True)
parser.add_argument('--n_samples', action='store', type=int, required=True)
parser.add_argument('--n_steps', action='store', type=int, required=False, default=None)
parser.add_argument('--linker_size_model', action='store', type=str, required=False, default=None)
parser.add_argument('--device', action='store', type=str, required=True)
args = parser.parse_args()

experiment_name = args.checkpoint.split('/')[-1].replace('.ckpt', '')

if args.linker_size_model is None:
    output_dir = os.path.join(args.samples, args.prefix, experiment_name)
else:
    linker_size_name = args.linker_size_model.split('/')[-1].replace('.ckpt', '')
    output_dir = os.path.join(args.samples, args.prefix, 'sampled_size', linker_size_name, experiment_name)

os.makedirs(output_dir, exist_ok=True)


def check_if_generated(_output_dir, _uuids, n_samples):
    generated = True
    starting_points = []
    for _uuid in _uuids:
        uuid_dir = os.path.join(_output_dir, _uuid)
        numbers = []
        for fname in os.listdir(uuid_dir):
            try:
                num = int(fname.split('_')[0])
                numbers.append(num)
            except:
                continue
        if len(numbers) == 0 or max(numbers) != n_samples - 1:
            generated = False
            if len(numbers) == 0:
                starting_points.append(0)
            else:
                starting_points.append(max(numbers) - 1)

    if len(starting_points) > 0:
        starting = min(starting_points)
    else:
        starting = None

    return generated, starting


collate_fn = collate
sample_fn = None
if args.linker_size_model is not None:
    size_nn = SizeClassifier.load_from_checkpoint(args.linker_size_model, map_location=args.device)
    size_nn = size_nn.eval().to(args.device)

    collate_fn = collate_with_fragment_edges

    def sample_fn(_data):
        output, _ = size_nn.forward(_data)
        probabilities = torch.softmax(output, dim=1)
        distribution = torch.distributions.Categorical(probs=probabilities)
        samples = distribution.sample()
        sizes = []
        for label in samples.detach().cpu().numpy():
            sizes.append(size_nn.linker_id2size[label])
        sizes = torch.tensor(sizes, device=samples.device, dtype=torch.long)
        return sizes

# Loading model form checkpoint (all hparams will be automatically set)
model = DDPM.load_from_checkpoint(args.checkpoint, map_location=args.device)

# Possibility to evaluate on different datasets (e.g., on CASF instead of ZINC)
model.val_data_prefix = args.prefix

# In case <Anonymous> will run my model or vice versa
if args.data is not None:
    model.data_path = args.data

# Less sampling steps
if args.n_steps is not None:
    model.edm.T = args.n_steps

# Setting up the model
model = model.eval().to(args.device)
model.torch_device = args.device
model.setup(stage='val')

# Getting the dataloader
dataloader = model.val_dataloader(collate_fn=collate_fn)
print(f'Dataloader contains {len(dataloader)} batches')

for batch_idx, data in enumerate(dataloader):
    uuids = []
    true_names = []
    frag_names = []
    pock_names = []
    for uuid in data['uuid']:
        uuid = str(uuid)
        uuids.append(uuid)
        true_names.append(f'{uuid}/true')
        frag_names.append(f'{uuid}/frag')
        pock_names.append(f'{uuid}/pock')
        os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)

    generated, starting_point = check_if_generated(output_dir, uuids, args.n_samples)
    if generated:
        print(f'Already generated batch={batch_idx}, max_uuid={max(uuids)}')
        continue
    if starting_point > 0:
        print(f'Generating {args.n_samples - starting_point} for batch={batch_idx}')

    # Removing COM of fragment from the atom coordinates
    h, x, node_mask, frag_mask = data['one_hot'], data['positions'], data['atom_mask'], data['fragment_mask']
    if model.inpainting:
        center_of_mass_mask = node_mask
    if isinstance(model.val_dataset, MOADDataset) and model.center_of_mass == 'fragments':
        center_of_mass_mask = data['fragment_only_mask']
    elif model.center_of_mass == 'fragments':
        center_of_mass_mask = data['fragment_mask']
    elif model.center_of_mass == 'anchors':
        center_of_mass_mask = data['anchors']
    else:
        raise NotImplementedError(model.center_of_mass)
    x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
    utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

    # Saving pocket if applicable
    if isinstance(model.val_dataset, MOADDataset):
        node_mask = data['atom_mask'] - data['pocket_mask']
        frag_mask = data['fragment_only_mask']
        pock_mask = data['pocket_mask']
        save_xyz_file(output_dir, h, x, pock_mask, pock_names, is_geom=model.is_geom)

    # Saving ground-truth molecules
    save_xyz_file(output_dir, h, x, node_mask, true_names, is_geom=model.is_geom)

    # Saving fragments
    save_xyz_file(output_dir, h, x, frag_mask, frag_names, is_geom=model.is_geom)

    # Sampling and saving generated molecules
    for i in tqdm(range(starting_point, args.n_samples), desc=str(batch_idx)):
        chain, node_mask = model.sample_chain(data, sample_fn=sample_fn, keep_frames=1)
        x = chain[0][:, :, :model.n_dims]
        h = chain[0][:, :, model.n_dims:]

        if isinstance(model.val_dataset, MOADDataset):
            node_mask = node_mask - data['pocket_mask']

        pred_names = [f'{uuid}/{i}' for uuid in uuids]
        save_xyz_file(output_dir, h, x, node_mask, pred_names, is_geom=model.is_geom)
