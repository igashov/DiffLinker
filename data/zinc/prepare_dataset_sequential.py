import argparse
import os
import prepare_dataset as prep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', action='store', type=str, required=True)
    parser.add_argument('--sdf-dir', action='store', type=str, required=True)
    parser.add_argument('--out-dir', action='store', type=str, required=True)
    parser.add_argument('--template', action='store', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fname in sorted(os.listdir(args.sdf_dir)):
        if fname.startswith(args.template) and fname.endswith('.sdf'):
            idx = fname.replace('.sdf', '').replace(args.template, '').replace('_', '')
            sdf_path = os.path.join(args.sdf_dir, fname)
            out_mol_path = os.path.join(args.out_dir, f'{args.template}_mol_{idx}.sdf')
            out_frag_path = os.path.join(args.out_dir, f'{args.template}_frag_{idx}.sdf')
            out_link_path = os.path.join(args.out_dir, f'{args.template}_link_{idx}.sdf')
            out_table_path = os.path.join(args.out_dir, f'{args.template}_table_{idx}.csv')

            print(f'Processing {idx}')
            kwargs = {
                'table_path': args.table,
                'sdf_path': sdf_path,
                'out_mol_path': out_mol_path,
                'out_frag_path': out_frag_path,
                'out_link_path': out_link_path,
                'out_table_path': out_table_path,
                'progress': True,
            }
            prep.run(**kwargs)
