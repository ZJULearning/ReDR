#! /usr/bin/env python
# coding: utf-8
# --------------------------------------------------------------------------------
#     File Name           :     src/models/seq2seq/model_handler.py
#     Created By          :     hao
#     Description         :
# --------------------------------------------------------------------------------
from .coqa_dataset import create_dataset
from .preprocessor import preprocess as prepro
from .drqa_model import DrQA
from .config import update_model_specific_args

_dataset = None


def create_model(dataset, args):
    vocab = dataset.get_vocab()
    return DrQA(vocab, args)


def get_dataset(trainset_file=None, devset_file=None, args=None):
    global _dataset
    if _dataset is None:
        if trainset_file is None or devset_file is None or args is None:
            raise Exception("Not supposed to be here.")
        _dataset = create_dataset(trainset_file, devset_file,
                                  vocab_size=args.vocab_size,
                                  device=args.device,
                                  embed_type=args.embed_type,
                                  embed_dir=args.embed_dir)
        args.sos_id = _dataset.sos_id
    return _dataset


def preprocess(args):
    prepro(args)


def update_model_args(args):
    update_model_specific_args(args)
