from collections import defaultdict
import numpy as np
import pdb
import argparse

def cal_entropy(generated):
    etp_score = [0.0,0.0,0.0,0.0]
    div_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for gg in generated:
        g = gg.strip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) +1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) /total
    return etp_score, div_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default="pred.txt",
                        help='prediction file')
    args = parser.parse_args()

    with open(args.p) as f:
        ours = f.readlines()


    print(cal_entropy(ours))


if __name__ == "__main__":
    main()


