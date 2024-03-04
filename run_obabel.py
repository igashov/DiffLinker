import os
import sys
import subprocess


def convert(in_path, out_path):
    subprocess.run(f'obabel {in_path} -O {out_path}', shell=True)


input_dir = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

for in_fname in os.listdir(input_dir):
    if in_fname.endswith('.xyz'):
        out_fname = in_fname.replace('.xyz', '.sdf')
        input_path = os.path.join(input_dir, in_fname)
        output_path = os.path.join(output_dir, out_fname)
        convert(input_path, output_path)
