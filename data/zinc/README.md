# How to build ZINC train dataset

SMILES of molecules, fragments and linkers used for training, validating and testing (provided by DeLinker) can be downloaded here:
* [train set](https://github.com/oxpig/DeLinker/blob/master/data/data_zinc_final_train.txt)
* [validation set](https://github.com/oxpig/DeLinker/blob/master/data/data_zinc_final_valid.txt)
* [test set](https://github.com/oxpig/DeLinker/blob/master/data/data_zinc_final_test.txt)

To generate conformers for the train set, first create an auxiliary file with unique SMILES of molecules:
```
python -W ignore create_unique_smiles.py $INPUT_DATA $OUTPUT_SMILES
```

Next, generate conformers:
```
python -W ignore rdkit_conf_parallel.py $INPUT_SMILES $OUTPUT_TEMPLATE --cores $CORES
```

It will create `$CORES` SDF-files containing all training conformers (up to 20 for each molecule).
For example, `$CORES` is set to 5 and `$OUTPUT_TEMPLATE` is set to `/data/data_zinc_final_train` then the resulting files will look as follows:
```
data/
     data_zinc_final_train_0.sdf
     data_zinc_final_train_1.sdf
     data_zinc_final_train_2.sdf
     data_zinc_final_train_3.sdf
     data_zinc_final_train_4.sdf
```

Conformers for validation and test sets can be downloaded from [DeLinker repository](https://github.com/oxpig/DeLinker/blob/master/analysis/zinc_250k_valid_test_only.sdf).

Once conformers are ready, prepare fragments and linkers with 3D-coordinates and marked anchor atoms:
```
python -W ignore prepare_dataset.py \ 
                 --table $INPUT_TABLE \
                 --sdf $CONFORMERS_SDF \ 
                 --out-mol-sdf $OUT_MOL_SDF \ 
                 --out-frag-sdf $OUT_FRAG_SDF \ 
                 --out-link-sdf $OUT_LINK_SDF \
                 --out-table $OUTPUT_TABLE
```


Molecules will be written to `$OUT_MOL_SDF`, fragments and linkers will be written to `$OUT_FRAG_SDF` and `$OUT_LINK_SDF` correspondingly.
Order of molecules is the same in all three resulting files.
Molecules are written with properties `_Name` (SMILES string of the molecule) and `_Energy` (conformer energy).
Fragments and linkers are written with properties `_Name` (SMILES string of the molecule), `_SMILES` (SMILES string of fragments/linker) and `_Energy` (conformer energy).
The resulting table written to `$OUTPUT_TABLE` contains 5 columns: `molecule`, `fragments`, `linker`, `anchor_1`, `anchor_2`, `energy`. 

Example:

| molecule                                       | fragments                             | linker      | anchor_1   | anchor_2   | energy            |
|------------------------------------------------|---------------------------------------|-------------|------------|------------|-------------------|
| `CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1`      | `CC(C)(C)c1ccc2occc2c1.Fc1ccccc1`     | `CC(N)=O`   | 10         | 19         | 80.06926642582071 |
| `C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1` | `C[C@@H]1CC(N)C[C@@H](C)C1.Cn1cnnc1`  | `c1ccncc1`  | 4          | 14         | 57.11087739612921 |

For parallel processing of all SDF-files with conformers:
```
python -W ignore prepare_dataset_parallel.py \ 
                 --table $INPUT_TABLE \
                 --sdf-dir $CONFORMERS_SDF_DIR \ 
                 --out-dir $OUT_DIR \ 
                 --template $TEMPLATE \ 
                 --cores $CORES
```

For sequential processing of all SDF-files with conformers:
```
python -W ignore prepare_dataset_sequential.py \
                 --table $INPUT_TABLE \
                 --sdf-dir $INPUT_SDF_DIR \
                 --out-dir $OUTPUT_DIR \
                 --template $TEMPLATE \
```

It will process SDF-files from `$CONFORMERS_SDF_DIR` that have names `$TEMPLATE_X.sdf` with `0 <= X < $CORES`.
Files `$TEMPLATE_X_mol.sdf`, `$TEMPLATE_X_frag.sdf`, `$TEMPLATE_X_link.sdf` and `$TEMPLATE_X_data.csv` with `0 <= X < $CORES` will be saved to `$OUT_DIR`.

Once all chunks are processed, merge the resulting SDF and CSV files into single ones. 
Note that we do not include all the conformers in the resulting data, only one conformer with the lowest energy for each molecule:

```
python -W ignore filter_and_merge.py \
                 --in-dir $INPUT_DIR \
                 --out-dir $OUTPUT_DIR \
                 --template $TEMPLATE \
                 --number-of-files $CORES
```

# How to build ZINC test and val datasets

- SDF-file can be downloaded from [DeLinker repository](https://github.com/oxpig/DeLinker/blob/master/analysis/zinc_250k_valid_test_only.sdf).
- Val SMILES and fragmentations can be downloaded [here](https://github.com/oxpig/DeLinker/blob/master/data/data_zinc_final_valid.txt).
- Test SMILES and fragmentations can be downloaded [here](https://github.com/oxpig/DeLinker/blob/master/data/data_zinc_final_test.txt).

Download these files and save to the directory `$ZINC` and run script `prepare_val_test.py`:
```
ZINC=<define your directory>
mkdir -p $ZINC
wget https://raw.githubusercontent.com/oxpig/DeLinker/master/analysis/zinc_250k_valid_test_only.sdf -O $ZINC/zinc_250k_valid_test_only.sdf
wget https://raw.githubusercontent.com/oxpig/DeLinker/master/data/data_zinc_final_valid.txt -O $ZINC/data_zinc_final_valid.txt
wget https://raw.githubusercontent.com/oxpig/DeLinker/master/data/data_zinc_final_test.txt -O $ZINC/data_zinc_final_test.txt
python -W ignore prepare_val_test.py --zinc-dir $ZINC
```