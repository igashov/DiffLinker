# DiffLinker: Equivariant 3D-Conditional Diffusion Model for Molecular Linker Design


## Environment Setup

The code was tested in the following environment:

| Software          | Version   |
|-------------------|-----------|
| Python            | 3.10.5    |
| CUDA              | 10.2.89   |
| PyTorch           | 1.11.0    |
| PyTorch Lightning | 1.6.3     |
| OpenBabel         | 3.0.0     |

You can create a new conda environment using provided `yaml` file:
```shell
conda env create -f environment.yml
```
or installing all necessary packages manually:
```shell
conda create -c conda-forge -n difflinker rdkit
conda activate difflinker
conda install -c conda-forge imageio
conda install numpy
conda install -c anaconda scipy
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge tqdm
conda install -c conda-forge wandb
conda install -c conda-forge pytorch-lightning=1.6.3
conda install -c anaconda scikit-learn
conda install -c anaconda jupyter
conda install -c conda-forge timeout-decorator
conda install -c anaconda networkx
conda install -c conda-forge biopython
```

## Datasets

Please find the datasets with additional files here:

- [ZINC](https://doi.org/10.5281/zenodo.7121271)
- [CASF](https://doi.org/10.5281/zenodo.7121264)
- [GEOM](https://doi.org/10.5281/zenodo.7121278)
- [Pockets](https://doi.org/10.5281/zenodo.7121280)

or use direct download links:

| Dataset                       | Download Link                                                              |
|-------------------------------|----------------------------------------------------------------------------|
| ZINC train                    | https://zenodo.org/record/7121271/files/zinc_final_train.pt?download=1     |
| ZINC val                      | https://zenodo.org/record/7121271/files/zinc_final_val.pt?download=1       |
| ZINC test                     | https://zenodo.org/record/7121271/files/zinc_final_test.pt?download=1      |
| CASF test                     | https://zenodo.org/record/7121264/files/casf_final_test.pt?download=1      |
| GEOM train                    | https://zenodo.org/record/7121278/files/geom_multifrag_train.pt?download=1 |
| GEOM val                      | https://zenodo.org/record/7121278/files/geom_multifrag_val.pt?download=1   |
| GEOM test                     | https://zenodo.org/record/7121278/files/geom_multifrag_test.pt?download=1  |
| Pockets Full train            | https://zenodo.org/record/7121280/files/MOAD_train_full.pt?download=1      |
| Pockets Full val              | https://zenodo.org/record/7121280/files/MOAD_val_full.pt?download=1        |
| Pockets Full test             | https://zenodo.org/record/7121280/files/MOAD_test_full.pt?download=1       |
| Pockets Backbone train        | https://zenodo.org/record/7121280/files/MOAD_train_bb.pt?download=1        |
| Pockets Backbone val          | https://zenodo.org/record/7121280/files/MOAD_val_bb.pt?download=1          |
| Pockets Backbone test         | https://zenodo.org/record/7121280/files/MOAD_test_bb.pt?download=1         |
| Pockets Unconditioned train   | https://zenodo.org/record/7121280/files/MOAD_train.pt?download=1           |
| Pockets Unconditioned val     | https://zenodo.org/record/7121280/files/MOAD_val.pt?download=1             |
| Pockets Unconditioned test    | https://zenodo.org/record/7121280/files/MOAD_test.pt?download=1            |


## Models

Please find the models [here](https://doi.org/10.5281/zenodo.7121300) or use direct download links:

| Model                            | Download Link                                                                            |
|----------------------------------|------------------------------------------------------------------------------------------|
| ZINC DiffLinker                  | https://zenodo.org/record/7121300/files/zinc_difflinker.ckpt?download=1                  |
| ZINC DiffLinker (given anchors)  | https://zenodo.org/record/7121300/files/zinc_difflinker_given_anchors.ckpt?download=1    |
| ZINC Size GNN                    | https://zenodo.org/record/7121300/files/zinc_size_gnn.ckpt?download=1                    |
| GEOM DiffLinker                  | https://zenodo.org/record/7121300/files/geom_difflinker.ckpt?download=1                  |
| GEOM DiffLinker (given anchors)  | https://zenodo.org/record/7121300/files/geom_difflinker_given_anchors.ckpt?download=1    |
| GEOM Size GNN                    | https://zenodo.org/record/7121300/files/geom_size_gnn.ckpt?download=1                    |
| GEOM DiffLinker (full pocket)    | https://zenodo.org/record/7121300/files/pockets_difflinker_full.ckpt?download=1          |
| GEOM DiffLinker (backbone atoms) | https://zenodo.org/record/7121300/files/pockets_difflinker_backbone.ckpt?download=1      |
| GEOM DiffLinker (unconditioned)  | https://zenodo.org/record/7121300/files/pockets_difflinker_unconditioned.ckpt?download=1 |


## Usage

In this tutorial, we will consider training and testing process on ZINC dataset.

### Training DiffLinker

First, download datasets:
```shell
mkdir -p datasets
wget https://zenodo.org/record/7121271/files/zinc_final_train.pt?download=1 -O datasets/zinc_final_train.pt
wget https://zenodo.org/record/7121271/files/zinc_final_val.pt?download=1 -O datasets/zinc_final_val.pt
```

Next, create necessary directories:
```shell
mkdir -p models
mkdir -p logs
```

Run trainig:
```shell
python -W ignore train_difflinker.py --config configs/zinc_difflinker.yml
```

### Training Size GNN

```shell
python -W ignore train_size_gnn.py \
                 --experiment zinc_size_gnn \
                 --data datasets \
                 --train_data_prefix zinc_final_val \
                 --val_data_prefix zinc_final_val \
                 --hidden_nf 256 \
                 --n_layers 5 \
                 --batch_size 256 \
                 --normalization batch_norm \
                 --lr 1e-3 \
                 --task classification \
                 --loss_weights \
                 --device gpu \
                 --checkpoints models \
                 --logs logs
```


### Sampling

First, download test dataset:
```shell
mkdir -p datasets
wget https://zenodo.org/record/7121271/files/zinc_final_test.pt?download=1 -O datasets/zinc_final_test.pt
```

Download the necessary models:
```shell
mkdir -p models
wget https://zenodo.org/record/7121300/files/zinc_difflinker.ckpt?download=1 -O models/zinc_difflinker.ckpt
wget https://zenodo.org/record/7121300/files/zinc_size_gnn.ckpt?download=1 -O models/zinc_size_gnn.ckpt
```

Next, create necessary directories:
```shell
mkdir -p samples
mkdir -p trajectories
```

If you want to sample 250 linkers for each input set of fragments, run the following:
```shell
python -W ignore sample.py \
                 --checkpoint models/zinc_difflinker.ckpt \
                 --linker_size_model models/zinc_size_gnn.ckpt \
                 --samples samples \
                 --data datasets \
                 --prefix zinc_final_test \
                 --n_samples 2 \
                 --device cuda:0
```
You will be able to see `.xyz` files of the generated molecules in the directory `./samples`.

If you want to sample linkers and save trajectories, run the following:
```shell
python -W ignore sample_trajectories.py \
                 --checkpoint models/zinc_difflinker.ckpt \
                 --chains trajectories \
                 --data datasets \
                 --prefix zinc_final_test \
                 --keep_frames 10 \
                 --device cuda:0
```
You will be able to see trajectories as `.xyz`, `.png` and `.gif` files in the directory `./trajectories`. 

### Evaluation

First, you need to download ground-truth SMILES and SDF files of molecules, 
fragments and linkers from the relevant test sets (recomputed with OpenBabel) + SMILES of the training linkers.
Check [this resource](https://doi.org/10.5281/zenodo.7121448) for finding the right ones.
Here, we will download files for ZINC:
```shell
mkdir -p datasets
wget https://zenodo.org/record/7121448/files/zinc_final_test_smiles.smi?download=1 -O datasets/zinc_final_test_smiles.smi
wget https://zenodo.org/record/7121448/files/zinc_final_test_molecules.sdf?download=1 -O datasets/zinc_final_test_molecules.sdf
wget https://zenodo.org/record/7121448/files/zinc_final_train_linkers.smi?download=1 -O datasets/zinc_final_train_linkers.smi 
```

Next, you need to run OpenBabel to reformat the data:
```shell
mkdir -p formatted
python -W ignore reformat_data_obabel.py \
                 --samples samples \
                 --dataset zinc_final_test \
                 --true_smiles_path datasets/zinc_final_test_smiles.smi \
                 --checkpoint zinc_difflinker \
                 --formatted formatted \
                 --linker_size_model_name zinc_size_gnn
```

Then you can run evaluation scripts:
```shell
python -W ignore compute_metrics.py \
                 ZINC \
                 formatted/zinc_difflinker/sampled_size/zinc_size_gnn/zinc_final_test.smi \
                 datasets/zinc_final_train_linkers.smi \
                 5 1 None \
                 resources/wehi_pains.csv \
                 diffusion
```
All the metrics will be saved in the directory `./formatted`.