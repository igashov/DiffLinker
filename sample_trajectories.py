import argparse
import os
import torch

from src.datasets import get_dataloader
from src.lightning import DDPM
from src.visualizer import save_xyz_file, visualize_chain
from tqdm import tqdm
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store', type=str, required=True)
parser.add_argument('--chains', action='store', type=str, required=True)
parser.add_argument('--prefix', action='store', type=str, required=True)
parser.add_argument('--data', action='store', type=str, required=False, default=None)
parser.add_argument('--keep_frames', action='store', type=int, required=True)
parser.add_argument('--device', action='store', type=str, required=True)
args = parser.parse_args()

experiment_name = args.checkpoint.split('/')[-1].replace('.ckpt', '')
chains_output_dir = os.path.join(args.chains, experiment_name, args.prefix, 'chains')
final_states_output_dir = os.path.join(args.chains, experiment_name, args.prefix, 'final_states')
os.makedirs(chains_output_dir, exist_ok=True)
os.makedirs(final_states_output_dir, exist_ok=True)

# Loading model form checkpoint (all hparams will be automatically set)
model = DDPM.load_from_checkpoint(args.checkpoint, map_location=args.device)

# Possibility to evaluate on different datasets (e.g., on CASF instead of ZINC)
model.val_data_prefix = args.prefix

# In case <Anonymous> will run my model or vice versa
if args.data is not None:
    model.data_path = args.data

model = model.eval().to(args.device)
model.setup(stage='val')
dataloader = get_dataloader(
    model.val_dataset,
    batch_size=32,
    # batch_size=len(model.val_dataset)
)

start = 0
for data in tqdm(dataloader):
    chain_batch, node_mask = model.sample_chain(data, keep_frames=args.keep_frames)
    for i in tqdm(range(len(data['positions']))):
        chain = chain_batch[:, i, :, :]
        assert chain.shape[0] == args.keep_frames
        assert chain.shape[1] == data['positions'].shape[1]
        assert chain.shape[2] == data['positions'].shape[2] + data['one_hot'].shape[2] + model.include_charges

        # Saving chains
        name = str(i + start)
        chain_output = os.path.join(chains_output_dir, name)
        os.makedirs(chain_output, exist_ok=True)

        one_hot = chain[:, :, 3:-1]
        positions = chain[:, :, :3]
        chain_node_mask = torch.cat([node_mask[i].unsqueeze(0) for _ in range(args.keep_frames)], dim=0)
        names = [f'{name}_{j}' for j in range(args.keep_frames)]

        save_xyz_file(chain_output, one_hot, positions, chain_node_mask, names=names, is_geom=model.is_geom)
        visualize_chain(
            chain_output,
            spheres_3d=True,
            alpha=0.7,
            bg='white',
            is_geom=model.is_geom,
            fragment_mask=data['fragment_mask'][i].squeeze()
        )

        # Saving final prediction and ground truth separately
        true_one_hot = data['one_hot'][i].unsqueeze(0)
        true_positions = data['positions'][i].unsqueeze(0)
        true_node_mask = data['atom_mask'][i].unsqueeze(0)
        save_xyz_file(
            final_states_output_dir,
            true_one_hot,
            true_positions,
            true_node_mask,
            names=[f'{name}_true'],
            is_geom=model.is_geom,
        )

        pred_one_hot = chain[0, :, 3:-1].unsqueeze(0)
        pred_positions = chain[0, :, :3].unsqueeze(0)
        pred_node_mask = chain_node_mask[0].unsqueeze(0)
        save_xyz_file(
            final_states_output_dir,
            pred_one_hot,
            pred_positions,
            pred_node_mask,
            names=[f'{name}_pred'],
            is_geom=model.is_geom
        )

    start += len(data['positions'])
