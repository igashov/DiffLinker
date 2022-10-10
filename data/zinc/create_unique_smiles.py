import sys
import pandas as pd

input_path = sys.argv[1]
output_path = sys.argv[2]

table = pd.read_csv(input_path, sep=' ', names=['molecule', 'linker', 'fragments', 'distance', 'angle'])
smiles = table.molecule.unique()
with open(output_path, 'w') as f:
    for smi in smiles:
        f.write(f'{smi}\n')
