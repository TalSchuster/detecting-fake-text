import torch
import argparse
import pickle
from tqdm import tqdm

from server import *

'''
compute the sentence perplexities with gpt2 lm.
assumes sentence per row
'''

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='gpt-2-small')
parser.add_argument("-i", "--input_file", required=True)

OUTPUT_FILE = '%s.perplexity.pickle'


if __name__ == '__main__':
    args = parser.parse_args()

    proj = projects[args.model]

    #sample = 'The cat was playing in the garden.'
    #sample = "The following is a transcript from The Guardian's interview with the British ambassador to the UN, John Baird."

    perps = []
    with open(args.input_file, 'r') as f:
        for line in tqdm(f):
            sample = line.strip()
            out = proj.lm.check_probabilities(sample)

            yhat = out['yhat'].cpu().detach().numpy()
            y = out['y'].cpu().numpy()

            probs = []
            perp = 1
            for i, tok in enumerate(yhat):
                p = tok[y[i]]
                probs.append(p)
                perp = perp * (1/p)

            perp = pow(perp, 1/float(len(probs)))
            perps.append(perp)

    print(perps)
    out_file = OUTPUT_FILE % args.input_file
    pickle.dump(perps, open(out_file, 'wb'))
