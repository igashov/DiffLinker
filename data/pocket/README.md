# How to prepare Pockets dataset 

Download Binding MOAD:
```
wget http://www.bindingmoad.org/files/biou/every_part_a.zip
wget http://www.bindingmoad.org/files/biou/every_part_b.zip
unzip every_part_a.zip
unzip every_part_b.zip
```

Clean and split raw PL-complexes:
```
python -W ignore clean_and_split.py 
                 --in-dir $RAW_MOAD \ 
                 --proteins-dir $PROCESSED_MOAD/proteins \ 
                 --ligands-dir $PROCESSED_MOAD/ligands
```

Create fragments and conformers:
```
python -W ignore generate_fragmentation_and_conformers.py \
                 --in-ligands $PROCESSED_MOAD/ligands \
                 --out-fragmentations $PROCESSED_MOAD/generated_splits.csv \
                 --out-conformers $PROCESSED_MOAD/generated_conformers.sdf
```

Prepare dataset:
```
python -W ignore prepare_dataset.py \
                 --table $PROCESSED_MOAD/generated_splits.csv \
                 --sdf $PROCESSED_MOAD/generated_conformers.sdf \
                 --proteins $PROCESSED_MOAD/proteins \
                 --out-mol-sdf $PROCESSED_MOAD/MOAD_mol.sdf \
                 --out-frag-sdf $PROCESSED_MOAD/MOAD_frag.sdf \
                 --out-link-sdf $PROCESSED_MOAD/MOAD_link.sdf \
                 --out-pockets-pkl $PROCESSED_MOAD/MOAD_pockets.pkl \
                 --out-table $PROCESSED_MOAD/MOAD_table.csv
```

Final filtering and train/val/test split:
```
python -W ignore filter_and_train_test_split.py \
                 --mol-sdf $PROCESSED_MOAD/MOAD_mol.sdf \
                 --frag-sdf $PROCESSED_MOAD/MOAD_frag.sdf \
                 --link-sdf $PROCESSED_MOAD/MOAD_link.sdf \
                 --pockets-pkl $PROCESSED_MOAD/MOAD_pockets.pkl \
                 --table $PROCESSED_MOAD/MOAD_table.csv
```