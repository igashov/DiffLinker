# Design of Hsp90 inhibitors

```shell
mkdir -p models
wget https://zenodo.org/records/10988017/files/pockets_difflinker_full_fc_pdb_excluded.ckpt?download=1 -O models/pockets_difflinker_full_given_anchors.ckpt
wget https://zenodo.org/record/7121300/files/zinc_size_gnn.ckpt?download=1 -O models/zinc_size_gnn.ckpt

python ../../generate_with_protein.py \
    --fragments 3hz1_modified_fragments_obabel.sdf \
    --protein 3hz1_protein.pdb \
    --output samples \
    --model models/pockets_difflinker_full_given_anchors.ckpt \
    --linker_size models/zinc_size_gnn.ckpt \
    --anchors 12,22 \
    --n_samples 1000 \
    --max_batch_size 16 \
    --random_seed 1
```
