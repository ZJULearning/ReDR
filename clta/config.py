#! /usr/bin/env python
# coding: utf-8
# --------------------------------------------------------------------------------
#     File Name           :     src/models/drqa/config.py
#     Created By          :     hao
#     Description         :
# --------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse


def update_model_specific_args(normal_args):
    parser = argparse.ArgumentParser()

    # parser.add_argument('--n_history', type=int, default=0)
    # parser.add_argument('--cased', type=bool, default=True, help='Cased or uncased version.')
    # parser.add_argument('--min_freq', type=int, default=20)
    # parser.add_argument('--top_vocab', type=int, default=100000)

    group = parser.add_argument_group('model_spec')
    group.add_argument('--rnn_padding', type=bool, default=False, help='Whether to use RNN padding.')
    group.add_argument('--hidden_size', type=int, default=300, help='Set hidden size.')
    group.add_argument('--n_history', type=int, default=2, help='...')
    group.add_argument('--num_layers', type=int, default=3, help='Number of layers for document/question encoding.')
    group.add_argument('--rnn_type', type=str, choices=['lstm', 'gru', 'rnn'], default='lstm', help='RNN type.')
    group.add_argument('--concat_rnn_layers', type=bool, default=True, help='Whether to concat RNN layers.')
    group.add_argument('--question_merge', type=str, choices=['avg', 'self_attn'],
                       default='self_attn', help='The way of question encoding.')
    group.add_argument('--use_cove', type=bool, default=False, help='CoVe')
    group.add_argument('--fix_cove', type=bool, default=True, help='CoVe')
    group.add_argument('--use_qemb', type=bool, default=True, help='Whether to add question aligned embedding.')
    group.add_argument('--f_qem', type=bool, default=True, help='Add exact match question feature to embedding.')
    group.add_argument('--f_pos', type=bool, default=False, help='Add POS feature to embedding.')
    group.add_argument('--f_ner', type=bool, default=False, help='Add NER feature to embedding.')
    group.add_argument('--sum_loss', type=bool, default=False, help="Set the type of loss.")
    group.add_argument('--doc_self_attn', type=bool, default=False,
                       help="Set whether to use self attention on the document.")
    group.add_argument('--resize_rnn_input', type=bool, default=False,
                       help='Reshape input layer to hidden size dimension.')
    group.add_argument('--span_dependency', type=bool, default=True,
                       help='Toggles dependency between the start and end span predictions for DrQA.')
    group.add_argument('--fix_embeddings', type=bool, default=False, help='Whether to fix embeddings.')
    group.add_argument('--dropout_rnn', type=float, default=0.3, help='Set RNN dropout in reader.')
    group.add_argument('--dropout_emb', type=float, default=0.5, help='Set embedding dropout.')
    group.add_argument('--dropout_ff', type=float, default=0.5, help='Set dropout for all feedforward layers.')
    group.add_argument('--dropout_rnn_output', type=bool, default=True, help='Whether to dropout last layer.')
    group.add_argument('--variational_dropout', type=bool, default=True, help='Set variational dropout on/off.')
    group.add_argument('--word_dropout', type=bool, default=False, help='Whether to dropout word.')

    # Optimizer
    group = parser.add_argument_group('training_spec')
    group.add_argument('--grad_clipping', type=float, default=10.0, help='Whether to use grad clipping.')
    group.add_argument('--weight_decay', type=float, default=0.0, help='Set weight decay.')
    group.add_argument('--momentum', type=float, default=0.0, help='Set momentum.')
    group.add_argument('--max_answer_len', type=int, default=15, help='Set max answer length for decoding.')
    group.add_argument('--predict_train', type=bool, default=True, help='Whether to predict on training set.')
    group.add_argument('--out_predictions', type=bool, default=True, help='Whether to output predictions.')
    group.add_argument('--predict_raw_text', type=bool, default=True,
                       help='Whether to use raw text and offsets for prediction.')
    args = parser.parse_known_args()[0]

    vars(normal_args).update(vars(args))

