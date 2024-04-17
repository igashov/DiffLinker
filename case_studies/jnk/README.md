# Improving selectivity of JNK inhibitors

```shell
mkdir -p models
wget https://zenodo.org/records/10988017/files/pockets_difflinker_full_fc_pdb_excluded.ckpt?download=1 -O models/pockets_difflinker_full_given_anchors.ckpt

python ../../generate_with_protein.py \
    --fragments 3fi3_fragments.sdf \
    --protein 3fi3_protein.pdb \
    --output samples_size_8 \
    --model models/pockets_difflinker_full_given_anchors.ckpt \
    --linker_size 8 \
    --anchors 16,24 \
    --n_samples 1000 \
    --max_batch_size 64 \
    --random_seed 1

python ../../generate_with_protein.py \
    --fragments 3fi3_fragments.sdf \
    --protein 3fi3_protein.pdb \
    --output samples_size_9 \
    --model models/pockets_difflinker_full_given_anchors.ckpt \
    --linker_size 9 \
    --anchors 16,24 \
    --n_samples 1000 \
    --max_batch_size 64 \
    --random_seed 1
```
