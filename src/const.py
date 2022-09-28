import torch

from rdkit import Chem


TORCH_FLOAT = torch.float32
TORCH_INT = torch.int8

# #################################################################################### #
# ####################################### ZINC ####################################### #
# #################################################################################### #

# Atom idx for one-hot encoding
ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7}
IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I'}

# Atomic numbers (Z)
CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}

# One-hot atom types
NUMBER_OF_ATOM_TYPES = len(ATOM2IDX)


# #################################################################################### #
# ####################################### GEOM ####################################### #
# #################################################################################### #

# Atom idx for one-hot encoding
GEOM_ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
GEOM_IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P'}

# Atomic numbers (Z)
GEOM_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}

# One-hot atom types
GEOM_NUMBER_OF_ATOM_TYPES = len(GEOM_ATOM2IDX)

# Dataset keys
DATA_LIST_ATTRS = {
    'uuid', 'name', 'fragments_smi', 'linker_smi', 'num_atoms'
}
DATA_ATTRS_TO_PAD = {
    'positions', 'one_hot', 'charges', 'anchors', 'fragment_mask', 'linker_mask', 'pocket_mask', 'fragment_only_mask'
}
DATA_ATTRS_TO_ADD_LAST_DIM = {
    'charges', 'anchors', 'fragment_mask', 'linker_mask', 'pocket_mask', 'fragment_only_mask'
}

# Distribution of linker size in train data
LINKER_SIZE_DIST = {
    4: 85540,
    3: 113928,
    6: 70946,
    7: 30408,
    5: 77671,
    9: 5177,
    10: 1214,
    8: 12712,
    11: 158,
    12: 7,
}


# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
BONDS_1 = {
    'H': {
        'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
        'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
        'Cl': 127, 'Br': 141, 'I': 161
    },
    'C': {
        'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
        'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
        'I': 214
    },
    'N': {
        'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
        'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177
    },
    'O': {
        'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
        'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
        'I': 194
    },
    'F': {
        'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
        'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
        'I': 187
    },
    'B': {
        'H':  119, 'Cl': 175
    },
    'Si': {
        'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
        'F': 160, 'Cl': 202, 'Br': 215, 'I': 243,
    },
    'Cl': {
        'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
        'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
        'Br': 214
    },
    'S': {
        'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
        'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
        'I': 234
    },
    'Br': {
        'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
        'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222
    },
    'P': {
        'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
        'S': 210, 'F': 156, 'N': 177, 'Br': 222
    },
    'I': {
        'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
        'S': 234, 'F': 187, 'I': 266
    },
    'As': {
        'H': 152
    }
}

BONDS_2 = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186}
}

BONDS_3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113}
}

BOND_DICT = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND2IDX = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

ALLOWED_BONDS = {
    'H': 1,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'B': 3,
    'Al': 3,
    'Si': 4,
    'P': [3, 5],
    'S': 4,
    'Cl': 1,
    'As': 3,
    'Br': 1,
    'I': 1,
    'Hg': [1, 2],
    'Bi': [3, 5]
}

MARGINS_EDM = [10, 5, 2]

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
# RADII = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
RADII = [0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77]

ZINC_TRAIN_LINKER_ID2SIZE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ZINC_TRAIN_LINKER_SIZE2ID = {
    size: idx
    for idx, size in enumerate(ZINC_TRAIN_LINKER_ID2SIZE)
}
ZINC_TRAIN_LINKER_SIZE_WEIGHTS = [
    3.47347831e-01,
    4.63079100e-01,
    5.12370917e-01,
    5.62392614e-01,
    1.30294388e+00,
    3.24247801e+00,
    8.12391184e+00,
    3.45634358e+01,
    2.72428571e+02,
    6.26585714e+03
]


GEOM_TRAIN_LINKER_ID2SIZE = [
    3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 38, 41
]
GEOM_TRAIN_LINKER_SIZE2ID = {
    size: idx
    for idx, size in enumerate(GEOM_TRAIN_LINKER_ID2SIZE)
}
GEOM_TRAIN_LINKER_SIZE_WEIGHTS = [
    1.07790681e+00, 4.54693604e-01, 3.62575713e-01, 3.75199484e-01,
    3.67812588e-01, 3.92388528e-01, 3.83421054e-01, 4.26924670e-01,
    4.92768040e-01, 4.99761944e-01, 4.92342726e-01, 5.71456905e-01,
    7.30631393e-01, 8.45412928e-01, 9.97252243e-01, 1.25423985e+00,
    1.57316129e+00, 2.19902962e+00, 3.22640431e+00, 4.25481066e+00,
    6.34749573e+00, 9.00676236e+00, 1.43084017e+01, 2.25763173e+01,
    3.36867096e+01, 9.50713805e+01, 2.08693274e+02, 2.51659537e+02,
    7.77856749e+02, 8.55642424e+03, 8.55642424e+03, 4.27821212e+03,
    4.27821212e+03
]
