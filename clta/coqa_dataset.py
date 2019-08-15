#! /usr/bin/env python
# coding: utf-8
# --------------------------------------------------------------------------------
#     File Name           :     coqa_dataset.py
#     Created By          :     hao
#     Description         :
# --------------------------------------------------------------------------------
import re
import torch
import logging
import torchtext
import CONSTANTS as CONST


def null_tokenizer(text):
    return text


def char_tokenizer(text_list):
    """
    @text: list of string
    """
    text_list = [re.sub(r'\s', ' ', text) for text in text_list]
    return list('\t'.join(text_list))


def filter_pred(example):
    answers = example.answers
    if 'unknown' in answers or 'yes' in answers or 'no' in answers:
        return False
    return True


class RawDataField(torchtext.data.Field):
    """
    RawDataField
    """
    def __init__(self, **kwargs):
        if kwargs.get('tokenize') is None:
            kwargs['tokenize'] = null_tokenizer
        if 'unk_token' not in kwargs:
            kwargs['unk_token'] = ' UNK '
        super(RawDataField, self).__init__(**kwargs)

    def process(self, batch, *args, **kwargs):
        return batch


class SimpleField(torchtext.data.Field):
    """
    SimpleFiled, similary to ReverseableField in torchtext,
    but more suitable for our project

    Used for denumericalize on minibatch:
        [1,2,3] -> [this is sample]
    """
    def __init__(self, **kwargs):
        if kwargs.get('tokenize') is None:
            kwargs['tokenize'] = 'revtok'
        if 'unk_token' not in kwargs:
            kwargs['unk_token'] = ' UNK '
        super(SimpleField, self).__init__(**kwargs)

    def reverse(self, batch):
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        # denumericalize
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        # trim past frst eos
        batch = [trim(ex, self.eos_token) for ex in batch]

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        return [' '.join(ex) for ex in batch]


class CoQADataset(object):
    """CoQA DataSet dataset."""

    def __init__(self, trainset_file, devset_file, device, vocab_size=None, embed_dir=None, embed_type=None):
        """
        CoQA dataset
        :param trainset_file: trainset file, tsv format
        :param devset_file: deveset file, tsv format
        :param device:
        """
        logging.info("Building dataset...")
        self._device = device
        self._context = SimpleField(
            sequential=True,
            include_lengths=True,
            batch_first=True,
            unk_token=CONST.UNK_TOKEN,
            pad_token=CONST.PAD_TOKEN,
            fix_length=None,
            use_vocab=True,
            tokenize=null_tokenizer,
            pad_first=False,
            dtype=torch.long,
            lower=True
        )
        self._question = SimpleField(
            sequential=True,
            include_lengths=True,
            batch_first=True,
            unk_token=CONST.UNK_TOKEN,
            pad_token=CONST.PAD_TOKEN,
            fix_length=None,
            use_vocab=True,
            tokenize=null_tokenizer,
            pad_first=False,
            is_target=True,
            dtype=torch.long,
            lower=True
        )
        self._answers = RawDataField(
            sequential=True,
            include_lengths=False,
            batch_first=True,
            fix_length=None,
            use_vocab=False,
            tokenize=null_tokenizer,
            pad_first=False,
            is_target=False,
            dtype=torch.long,
            lower=True
        )
        self._context_offsets = RawDataField(
            sequential=True,
            include_lengths=False,
            batch_first=True,
            fix_length=None,
            use_vocab=False,
            tokenize=null_tokenizer,
            pad_first=False,
            is_target=False,
            dtype=torch.long,
            lower=False
        )
        self._target = torchtext.data.Field(use_vocab=False, batch_first=True, dtype=torch.int64)

        trainset, devset = torchtext.data.TabularDataset.splits(
            path='/',
            train=trainset_file,
            validation=devset_file,
            filter_pred=filter_pred,
            format='json',
            fields={
                'span_word': ('context', self._context),
                'span_offsets': ('context_offsets', self._context_offsets),
                'question_word': ('question', self._question),
                'answers': ('answers', self._answers),
                'targets_in_span': ('targets', self._target)})
        self._trainset = trainset
        self._devset = devset

        # issue of torchtext 0.3.1
        # We want share vocab between src and trg,
        # but torchtext only supports same field to share different column.
        # So we build vocab only on one field and assign this vocab to another.
        # However, trg contains some tokens which not appear in src(EOS_TOKEN, SOS_TOKEN),
        # These tokens are prepend in vocab dict in function build_vocab.
        # So we use Field of trg to build vocab to make sure these tokens are included in our vocab.
        if embed_type is None:
            embed_vectors = None
            embed_dim = 0
        else:
            from lib.utils.embedding import get_embedding_vectors
            embed_vectors, embed_dim = get_embedding_vectors(embed_dir, embed_type, vocab_size)
        self._embed_dim = embed_dim
        self._context.build_vocab(
            trainset.context, devset.context, trainset.question, devset.question,
            max_size=vocab_size - 2,  # <unk>, <pad>
            vectors=embed_vectors)
        vocab = self._context.vocab
        self._vocab = vocab
        self._vectors = vocab.vectors

        self._question.vocab = vocab
        """Inputs:
        xq = question word indices             (batch, max_q_len)
        xd = document word indices             (batch, max_d_len)
        targets = span targets                 (batch,)
        """

        self.sos_token = CONST.SOS_TOKEN
        self.eos_token = CONST.EOS_TOKEN
        self.unk_token = CONST.UNK_TOKEN
        self.pad_token = CONST.PAD_TOKEN
        self.sos_id = self._vocab.stoi[CONST.SOS_TOKEN]
        self.eos_id = self._vocab.stoi[CONST.EOS_TOKEN]
        self.unk_id = self._vocab.stoi[CONST.UNK_TOKEN]
        self.pad_id = self._vocab.stoi[CONST.PAD_TOKEN]
        logging.info("Dataset is built.")
        logging.info("Vocab size: {}".format(len(vocab.itos)))
        logging.info("Top 10 words: {}".format(vocab.itos[:10]))

    def get_vectors(self):
        return self._vectors

    def get_num_train(self):
        return len(self._trainset)

    def get_vocab_size(self):
        return len(self._vocab.itos)

    def get_embed_dim(self):
        return self._embed_dim

    def get_num_dev(self):
        return len(self._devset)

    def get_train_iterator(self, batch_size, shuffle=None):
        train_iter = torchtext.data.BucketIterator(
            dataset=self._trainset, batch_size=batch_size,
            shuffle=shuffle, device=self._device,
            sort_within_batch=True,
            sort_key=lambda x: len(x.context[0]))
        # sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg)))
        return train_iter

    def get_dev_iterator(self, batch_size, shuffle=None):
        dev_iter = torchtext.data.BucketIterator(
            dataset=self._devset, batch_size=batch_size,
            shuffle=shuffle, device=self._device,
            sort_within_batch=True,
            sort_key=lambda x: len(x.context[0]))
        # sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg)))
        return dev_iter

    def reverse(self, data):
        vocab = self._vocab
        rep_str = []
        rep_list = []
        for ex in data:
            l_ex = [vocab.itos[idx] for idx in ex if vocab.itos[idx] not in (
                CONST.PAD_TOKEN, CONST.SOS_TOKEN, CONST.EOS_TOKEN)]
            s_ex = ' '.join(l_ex)
            rep_list.append(l_ex)
            rep_str.append(s_ex)
        return rep_list, rep_str

    def get_vocab(self):
        return self._vocab


def create_dataset(trainset_file, devset_file, device, vocab_size=None, embed_type=None, embed_dir=None):
    return CoQADataset(trainset_file, devset_file,
                       vocab_size=vocab_size, device=device, embed_type=embed_type, embed_dir=embed_dir)


if __name__ == '__main__':
    import os
    homedir = os.path.expanduser("~")
    trainset_file = os.path.join(homedir, "data/nlp/coqa/coqa-dev.json.txt")
    devset_file = os.path.join(homedir, "data/nlp/coqa/coqa-dev.json.txt")

    dataset = CoQADataset(trainset_file, devset_file, device='cpu', vocab_size=10000)

    for batch in dataset.get_train_iterator(batch_size=2):
        batch_input, batch_lengths = batch.context
        answers = batch.answers
        # print(batch_input.size())
        # print(batch_input)
        # print(batch_target)
        print(answers)
        # rep_list, rep_str = dataset.reverse(batch_input)
        # print(rep_str)
        # print('-------------------------')
        # print(rep_str[0])
        # print('-------------------------')
        # print(rep_str[1])
        # break
