#! /usr/bin/env python
# coding: utf-8
# --------------------------------------------------------------------------------
#     File Name           :     models/pgnet/__main__.py
#     Created By          :     hao
#     Description         :
# --------------------------------------------------------------------------------

import os
import torch
import json
from types import SimpleNamespace
from .model import DrQA


def test_dataset():
    print("!!!!!!!!")
    pass


def debug_main():
    homedir = os.path.expanduser("~")
    vocab = torch.load(os.path.join(homedir, "workspace/ml/nlp/clta/out/clta/clta1/vocab.pt"))
    json_config = json.load(open(os.path.join(homedir, "workspace/ml/nlp/clta/out/clta/clta1/hparams.json")))
    args = SimpleNamespace(**json_config)
    model = DrQA(vocab, args)
    print(len(vocab))
    print(args)


if __name__ == "__main__":
    debug_main()
