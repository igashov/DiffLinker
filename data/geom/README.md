# How to build GEOM dataset

Download GEOM-DRUGS dataset:  
```
mkdir $GEOMDRUGS
wget https://dataverse.harvard.edu/api/access/datafile/4327252 -O $GEOMDRUGS/rdkit_folder.tar
tar -xvf $GEOMDRUGS/rdkit_folder.tar
```

Generate all possible linkers-fragments splits using MMPA and BRICS algorithms:
```
python -W ignore generate_geom_multifrag.py \ 
                 --geom-json $GEOMDRUGS/summary_drugs.json \
                 --output $GEOMDRUGS/generated_splits.csv \
                 --min-size 3 \
                 --mmpa \
                 --brics
```

Prepare SDF-file with conformers:
```
python -W ignore generate_geom_conformers.py \
                 --geom-json $GEOMDRUGS/summary_drugs.json \
                 --geom-root $GEOMDRUGS \
                 --generated-splits $GEOMDRUGS/generated_splits.csv \ 
                 --output $CONFORMERS_SDF_DIR/geom_multifrag.sdf
```

Train/val/test splits:
```
python -W ignore train_test_split.py \
                 --generated-splits $GEOMDRUGS/generated_splits.csv \
                 --train $GEOMDRUGS/generated_splits_train.csv \ 
                 --val $GEOMDRUGS/generated_splits_val.csv \ 
                 --test $GEOMDRUGS/generated_splits_test.csv \ 
                 --plot plots/geom_multifrag_distributions.pdf
```

Prepare final datasets:
```
python -W ignore prepare_geom_dataset.py \
                 --table $GEOMDRUGS/generated_splits_test.csv \
                 --sdf $CONFORMERS_SDF_DIR/geom_multifrag.sdf \
                 --out-mol-sdf $DATASETS/geom_multifrag_test_mol.sdf \
                 --out-frag-sdf $DATASETS/geom_multifrag_test_frag.sdf \
                 --out-link-sdf $DATASETS/geom_multifrag_test_link.sdf \
                 --out-table $DATASETS/geom_multifrag_test_table.csv
```