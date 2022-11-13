import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Generates plot of the face attributes from the CSV file')

parser.add_argument("--src", default="/scratch/data/kotti1/fair-distill/results/img_outputs.csv", type=str, help='path of the directory containing images of generative algorithms')
parser.add_argument("--outdir", default="/scratch/data/kotti1/fair-distill/results", type=str, help='path to save the results')

def plot_attributes(src, outdir, figsize = (8,8)):
    df = pd.read_csv(src)
    plt.figure(figsize=figsize)

    plt.title("Race")
    plt.ylim(0.0, 0.7)
    df['race'].value_counts(normalize=True).sort_index(ascending=False).plot(kind='bar', rot=-45, legend=True)
    plt.savefig(outdir + '/plot_race.pdf', format = 'pdf')
    #plt.show()
    plt.close()


    plt.title("Race4")
    plt.ylim(0.0, 0.8)
    df['race4'].value_counts(normalize=True).sort_index(ascending=False).plot(kind='bar', rot=-45, legend=True)
    plt.savefig(outdir + '/plot_race4.pdf', format = 'pdf')
    #plt.show()
    plt.close()


    plt.title("Gender")
    df['gender'].value_counts(normalize=True).sort_index(ascending=False).plot(kind='bar', rot=-45, legend=True)
    plt.savefig(outdir + '/plot_gender.pdf', format = 'pdf')
    #plt.show()
    plt.close()

    plt.title("Age")
    plt.ylim(0.00, 0.40)
    df['age'].value_counts(normalize=True).sort_index(ascending=False).plot(kind='bar', rot=-45, legend=True)
    plt.savefig(outdir + '/plot_age.pdf', format = 'pdf')
    #plt.show()
    plt.close()


    plt.title("Race4-Age")
    df[['race4','age']].value_counts(normalize=True).sort_index(ascending=False).plot(kind='bar', rot=-90, legend=True)
    plt.savefig(outdir + '/plot_race4_age.pdf', format = 'pdf')
    #plt.show()
    plt.close()

    plt.title("Race-Age")
    df[['race','age']].value_counts(normalize=True).sort_index(ascending=False).plot(kind='bar', rot=-90, legend=True)
    plt.savefig(outdir + '/plot_race_age.pdf', format = 'pdf')
    #plt.show()
    plt.close()

def main():
    global args
    args = parser.parse_args()
    plot_attributes(src = args.src, outdir = args.outdir, figsize = (10,10))

if __name__ == "__main__":
    main()

