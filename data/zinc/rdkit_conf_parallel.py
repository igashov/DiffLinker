#!/usr/bin/env python

import sys

from multiprocessing import Process
from optparse import OptionParser
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from tqdm import tqdm

'''
Given a list of SMILES strings, generate 3D conformers in sdf format using RDKit.  
Energy minimizes and filters conformers to meet energy window and rms constraints.

Script modified from: https://github.com/dkoes/rdkit-scripts/blob/master/rdconf.py
'''


# convert smiles to sdf
def getRMS(mol, c1, c2):
    (rms, trans) = AllChem.GetAlignmentTransform(mol, mol, c1, c2)
    return rms


def compute_confs_worker(
        smifile,
        sdffile,
        pid,
        maxconfs=20,
        sample_mult=1,
        seed=42,
        rms_threshold=0.7,
        energy=10,
        verbose=False,
        mmff=False,
        nomin=False,
        etkdg=False,
        smi_frags=[],
        jpsettings=False,
):
    print(f'Worker {pid} started')
    outf = open(sdffile, 'w+')
    sdwriter = Chem.SDWriter(outf)
    if sdwriter is None:
        print("Could not open ", sdffile)
        sys.exit(-1)

    if verbose:
        print("Generating a maximum of", maxconfs, "per mol")

    if etkdg and not AllChem.ETKDG:
        print("ETKDB does not appear to be implemented. Please upgrade RDKit.")
        sys.exit(1)

    if smi_frags != []:
        if len(smifile) != len(smi_frags):
            print("smifile and smi_frags not equal in length")
            return None

    # Set clustering and sampling as per https://pubs.acs.org/doi/10.1021/ci2004658
    if jpsettings == True:
        rms_threshold = 0.35
        sample_mult = 1

    generator = tqdm(enumerate(smifile), total=len(smifile)) if pid == 0 else enumerate(smifile)
    for count, smi in generator:
        name = smi
        pieces = smi.split('.')
        if len(pieces) > 1:
            smi = max(pieces, key=len)  # take largest component by length
            print("Taking largest component: %s" % (smi))

        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            if verbose:
                print(smi)
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
                mol.SetProp("_Name", name)
                if jpsettings == True:
                    rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                    if rot_bonds <= 7:
                        maxconfs = 50
                    elif rot_bonds >= 8 and rot_bonds <= 12:
                        maxconfs = 200
                    else:
                        maxconfs = 300
                if smi_frags != []:
                    mol.SetProp("_StartingPoint", smi_frags[count])

                if etkdg:
                    cids = AllChem.EmbedMultipleConfs(mol, numConfs=int(sample_mult * maxconfs),
                                                      useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
                                                      randomSeed=seed, numThreads=1)
                else:
                    cids = AllChem.EmbedMultipleConfs(mol, int(sample_mult * maxconfs), randomSeed=seed, numThreads=1)
                if verbose:
                    print(len(cids), "conformers found")
                cenergy = []
                if mmff:
                    converged_res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
                    cenergy = [i[1] for i in converged_res]
                elif not nomin and not mmff:
                    converged_res = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1)
                    cenergy = [i[1] for i in converged_res]
                else:
                    for conf in cids:
                        # not passing confID only minimizes the first conformer
                        if nomin:
                            cenergy.append(conf)
                        elif mmff:
                            converged = AllChem.MMFFOptimizeMolecule(mol, confId=conf)
                            mp = AllChem.MMFFGetMoleculeProperties(mol)
                            cenergy.append(AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf).CalcEnergy())
                        else:
                            converged = not AllChem.UFFOptimizeMolecule(mol, confId=conf)
                            cenergy.append(AllChem.UFFGetMoleculeForceField(mol, confId=conf).CalcEnergy())
                        if verbose:
                            print("Convergence of conformer", conf, converged)

                mol = Chem.RemoveHs(mol)
                sortedcids = sorted(cids, key=lambda cid: cenergy[cid])
                if len(sortedcids) > 0:
                    mine = cenergy[sortedcids[0]]
                else:
                    mine = 0
                if (rms_threshold == 0):
                    cnt = 0
                    for conf_num, conf in enumerate(sortedcids):
                        if (cnt >= maxconfs):
                            break
                        if (energy < 0) or cenergy[conf] - mine <= energy:
                            mol.SetProp("_Model", str(conf_num))
                            mol.SetProp("_Energy", str(cenergy[conf]))
                            sdwriter.write(mol, conf)
                            cnt += 1
                else:
                    written = {}
                    for conf_num, conf in enumerate(sortedcids):
                        if len(written) >= maxconfs:
                            break
                        # check rmsd
                        passed = True
                        for seenconf in written:
                            rms = getRMS(mol, seenconf, conf)
                            if (rms < rms_threshold) or (energy > 0 and cenergy[conf] - mine > energy):
                                passed = False
                                break
                        if passed:
                            written[conf] = True
                            mol.SetProp("_Model", str(conf_num))
                            mol.SetProp("_Energy", str(cenergy[conf]))
                            sdwriter.write(mol, conf)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print("Exception", e)
        else:
            print("ERROR:", smi)

    sdwriter.close()
    return None


if __name__ == "__main__":
    parser = OptionParser(usage="Usage: %prog [options] <input>.smi <output>.sdf")
    parser.add_option("--maxconfs", dest="maxconfs", action="store",
                      help="maximum number of conformers to generate per a molecule (default 20)", default="20",
                      type="int", metavar="CNT")
    parser.add_option("--sample_multiplier", dest="sample", action="store",
                      help="sample N*maxconfs conformers and choose the maxconformers with lowest energy (default 1)",
                      default="1", type="float", metavar="N")
    parser.add_option("--seed", dest="seed", action="store",
                      help="random seed (default 9162006)", default="9162006", type="int", metavar="s")
    parser.add_option("--rms_threshold", dest="rms", action="store",
                      help="filter based on rms (default 0.7)", default="0.7", type="float", metavar="R")
    parser.add_option("--energy_window", dest="energy", action="store",
                      help="filter based on energy difference with lowest energy conformer", default="10", type="float",
                      metavar="E")
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
                      help="verbose output")
    parser.add_option("--mmff", dest="mmff", action="store_true", default=False,
                      help="use MMFF forcefield instead of UFF")
    parser.add_option("--nomin", dest="nomin", action="store_true", default=False,
                      help="don't perform energy minimization (bad idea)")
    parser.add_option("--etkdg", dest="etkdg", action="store_true", default=False,
                      help="use new ETKDG knowledge-based method instead of distance geometry")
    parser.add_option("--cores", dest="cores", action="store",
                      help="number of CPU cores to use", default=1, type="int")
    parser.add_option("--jpoptions", dest="jp", action="store_true", default=False,
                      help="use sampling options from JPs paper")

    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.error("Need input and output")
        sys.exit(-1)

    input_file = args[0]
    smiles = []
    with open(input_file, 'r') as f:
        for line in f:
            smiles.append(line.strip())
    output_template = args[1]

    assert options.cores < len(smiles)
    chunk_size = len(smiles) // options.cores

    processes = []
    for pid in range(options.cores):
        start = pid * chunk_size
        end = start + chunk_size
        kwargs = {
            'smifile': smiles[start:end],
            'sdffile': f'{output_template}_{pid}.sdf',
            'pid': pid,
            'maxconfs': options.maxconfs,
            'sample_mult': options.sample,
            'seed': options.seed,
            'rms_threshold': options.rms,
            'energy': options.energy,
            'verbose': False,
            'mmff': options.mmff,
            'nomin': options.nomin,
            'etkdg': options.etkdg,
            'smi_frags': [],
            'jpsettings': options.jp
        }
        process = Process(target=compute_confs_worker, kwargs=kwargs)
        process.start()
        processes.append(process)

    for pid in range(options.cores):
        processes[pid].join()
