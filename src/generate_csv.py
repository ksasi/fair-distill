import torch
import sys
import os
import argparse
import pickle
import PIL.Image
import csv
import glob
import subprocess
import pandas as pd


parser = argparse.ArgumentParser(description='This provides attributes for the generated images to verify fairness w.r.t these attributes')

parser.add_argument("--imgdir", default="/workspace/fairDL/data/stylegan2", type=str, help='path of the directory containing images of generative algorithms')
parser.add_argument("--outdir", default="/workspace/fairDL/results", type=str, help='path to save the results')


def main():
    global args
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    try:
        sys.path.index('/workspace/FairFace')
    except:
        sys.path.append('/workspace/FairFace')

    #print(sys.path)
    os.makedirs(args.outdir, exist_ok=True)

    with open(f'{args.outdir}/test_imgs.csv', 'w') as f:
        writer = csv.writer(f)
        a = glob.glob(f'{args.imgdir}/*')
        writer.writerows(zip(['img_path'] + a))


if __name__ == "__main__":
    main()

    

