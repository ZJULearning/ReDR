#! /usr/bin/env python
# coding: utf-8
# --------------------------------------------------------------------------------
#     File Name           :     embedding.py
#     Created By          :     hao
#     Description         :
# --------------------------------------------------------------------------------

import torchtext


def get_embedding_vectors(directory, embed_type, vocab_size):
    if embed_type == 'glove':
        vectors = torchtext.vocab.GloVe(name='840B', cache=directory, dim=300, max_vectors=vocab_size)
        dim = 300
        return vectors, dim
    else:
        raise NotImplementedError("Embed type not support. one of [glove] is required")
