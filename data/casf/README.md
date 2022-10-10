# How to build CASF test dataset

- SDF-file can be downloaded from [DeLinker repository](https://github.com/oxpig/DeLinker/blob/master/analysis/casf_structures.sdf).
- SMILES and fragmentations can be downloaded [here](https://github.com/oxpig/DeLinker/blob/master/data/data_casf_final.txt).

Download these files and save to the directory `$CASF` and run script `prepare.py`:
```
CASF=<define your directory>
mkdir -p $CASF
wget https://raw.githubusercontent.com/oxpig/DeLinker/master/analysis/casf_structures.sdf -O $CASF/casf_structures.sdf
wget https://raw.githubusercontent.com/oxpig/DeLinker/master/data/data_casf_final.txt -O $CASF/data_casf_final.txt
python -W ignore prepare.py --casf-dir $CASF
```