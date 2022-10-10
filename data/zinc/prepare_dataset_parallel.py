import argparse
import os
import prepare_dataset as prep
import multiprocessing as mp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', action='store', type=str, required=True)
    parser.add_argument('--sdf-dir', action='store', type=str, required=True)
    parser.add_argument('--out-dir', action='store', type=str, required=True)
    parser.add_argument('--template', action='store', type=str, required=True)
    parser.add_argument('--cores', action='store', type=int, required=True)
    args = parser.parse_args()

    processes = []
    for pid in range(args.cores):
        sdf_path = os.path.join(args.sdf_dir, f'{args.template}_{pid}.sdf')
        out_mol_path = os.path.join(args.out_dir, f'{args.template}_mol_{pid}.sdf')
        out_frag_path = os.path.join(args.out_dir, f'{args.template}_frag_{pid}.sdf')
        out_link_path = os.path.join(args.out_dir, f'{args.template}_link_{pid}.sdf')
        out_table_path = os.path.join(args.out_dir, f'{args.template}_table_{pid}.csv')
        kwargs = {
            'table_path': args.table,
            'sdf_path': sdf_path,
            'out_mol_path': out_mol_path,
            'out_frag_path': out_frag_path,
            'out_link_path': out_link_path,
            'out_table_path': out_table_path,
            'progress': pid == 0,
        }
        process = mp.Process(target=prep.run, kwargs=kwargs)
        process.start()
        processes.append(process)

    for pid in range(args.cores):
        processes[pid].join()
