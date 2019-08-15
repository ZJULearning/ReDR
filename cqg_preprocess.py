#! /usr/bin/env python
# coding: utf-8
# --------------------------------------------------------------------------------
#     File Name           :     preprocessor.py
#     Created By          :     hao
#     Description         :
# --------------------------------------------------------------------------------
import re
import os
import argparse
import unicodedata
import json
from tqdm import tqdm


MAX_LENGTH = 100
parser = argparse.ArgumentParser(description='Learn to ask')


parser.add_argument('--data_root_dir',
                    type=str,
                    default='data',
                    metavar='N',
                    help='')
parser.add_argument('--raw_trainset_file',
                    type=str,
                    default='coqa-train-v1.0.json',
                    metavar='N',
                    help='')
parser.add_argument('--raw_devset_file',
                    type=str,
                    default='coqa-dev-v1.0.json',
                    metavar='N',
                    help='')
parser.add_argument('--out_prefix',
                    type=str,
                    default='coqa-cqg',
                    metavar='N',
                    help='')
parser.add_argument('--n_history', type=int, default=5)


def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode2ascii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def remove_space(s):
    s = re.sub(r'\s', r' ', s)
    return s


def filter_pair(p):
    return len(p[0]) < MAX_LENGTH and \
           len(p[1]) < MAX_LENGTH


def history_to_string(history, n_history):
    flag = 0
    if len(history) > n_history:
        text_list = []
    else:
        text_list = ["<sos>"]
    for i, (que, ans) in enumerate(history[-n_history + flag:]):
        record = ["q{}".format(i)]
        record.extend(que)
        record.append("a{}".format(i))
        record.extend(ans)

        text_list.extend(record)
    return text_list


def process_file(tokenizer, data_type, json_file, out_root_dir, out_prefix, n_history):
    """
    doc = nlp(u'An example sentence. Another sentence.')
    assert (doc[0].text, doc[0].head.tag_) == ('An', 'NN')
    """
    data = json.load(open(json_file, "r"))["data"]
    history_file = os.path.join(out_root_dir, "{}-history.{}.txt".format(out_prefix, data_type))
    ref_file = os.path.join(out_root_dir, "{}-src.{}.txt".format(out_prefix, data_type))
    target_file = os.path.join(out_root_dir, "{}-tgt.{}.txt".format(out_prefix, data_type))
    ans_file = os.path.join(out_root_dir, "{}-ans.{}.txt".format(out_prefix, data_type))
    id_file = os.path.join(out_root_dir, "{}-id.{}.txt".format(out_prefix, data_type))
    with open(history_file, "w") as fhis, \
            open(ref_file, "w") as fref, \
            open(target_file, "w") as ftgt, \
            open(ans_file, 'w') as fans, \
            open(id_file, 'w') as fid:
        for entry in tqdm(data):
            # story = entry["story"]
            questions = entry["questions"]
            answers = entry["answers"]
            history = []
            for que, ans in zip(questions, answers):
                que_raw_text = remove_space(que["input_text"].lower())
                ans_raw_text = remove_space(ans["input_text"].lower())
                ref_raw_text = remove_space(ans['span_text'].lower())
                identify = "{},{}".format(entry['id'], que['turn_id'])
                que_tokens = tokenizer(que_raw_text)
                que_tokens = que_tokens[:MAX_LENGTH]
                ans_tokens = tokenizer(ans_raw_text)
                ans_tokens = ans_tokens[:MAX_LENGTH]
                ref_tokens = tokenizer(ref_raw_text)
                ref_tokens = ref_tokens[:MAX_LENGTH]
                ref_text = [token.text for token in ref_tokens]
                que_text = [token.text for token in que_tokens]
                ans_text = [token.text for token in ans_tokens]
                history_text = history_to_string(history, n_history)
                # we append this ans and que to history after generate history_text
                history.append((que_text, ans_text))
                fhis.write("{}\n".format(" ".join(history_text).strip()))
                fref.write("{}\n".format(" ".join(ref_text).strip()))
                ftgt.write("{}\n".format(" ".join(que_text).strip()))
                fans.write("{}\n".format(" ".join(ans_text).strip()))
                fid.write("{}\n".format(identify))


def preprocess(args):

    from spacy.lang.en import English
    nlp = English()
    tokenizer = English().Defaults.create_tokenizer(nlp)
    process_file(
        tokenizer,
        "train",
        args.raw_trainset_file,
        args.data_root_dir,
        args.out_prefix,
        args.n_history)

    process_file(
        tokenizer,
        "dev",
        args.raw_devset_file,
        args.data_root_dir,
        args.out_prefix,
        args.n_history)


if __name__ == "__main__":
    args = parser.parse_known_args()[0]
    args.raw_trainset_file = os.path.join(args.data_root_dir, args.raw_trainset_file)
    args.raw_devset_file = os.path.join(args.data_root_dir, args.raw_devset_file)
    preprocess(args)
