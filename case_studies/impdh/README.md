# Design of IMPDH inhibitors

```shell
mkdir -p models
wget https://zenodo.org/records/10988017/files/pockets_difflinker_full_fc_pdb_excluded.ckpt?download=1 -O models/pockets_difflinker_full_given_anchors.ckpt

python ../../generate_with_protein.py \
    --fragments 5ou2_fragments_input.sdf \
    --protein 5ou2_protein.pdb \
    --output samples_size_5 \
    --model models/pockets_difflinker_full_given_anchors.ckpt \
    --linker_size 5 \
    --anchors 2,21 \
    --n_samples 1000 \
    --max_batch_size 64

python ../../generate_with_protein.py \
    --fragments 5ou2_fragments_input.sdf \
    --protein 5ou2_protein.pdb \
    --output samples_size_6 \
    --model models/pockets_difflinker_full_given_anchors.ckpt \
    --linker_size 6 \
    --anchors 2,21 \
    --n_samples 1000 \
    --max_batch_size 64
```