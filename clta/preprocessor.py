"""
    This file takes a CoQA data file as input and generates the input files for training the DrQA (extractive QA) model.
"""


import argparse
import json
import re
import time
import string
from collections import Counter

UNK = 'unknown'
_NLP = None


def trip_space(text):
    start = 0
    end = 0
    while len(text) > 0 and text[0] in string.whitespace:
        text = text[1:]
        start += 1
    while len(text) > 0 and text[-1] in string.whitespace:
        text = text[:-1]
        end -= 1
    return text, start, end


def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s


def process(text, require_sents=False):
    output = {'word': [],
              # 'lemma': [],
              # 'pos': [],
              # 'ner': [],
              'offsets': []}
    global _NLP
    doc = _NLP(text)
    if require_sents:
        output['sent_offsets'] = []
        for sent in doc.sents:
            pre = 0
            post = 0
            try:
                while text[sent.start_char + pre].strip() == "":
                    pre += 1
                while text[sent.end_char - 1 - post].strip() == "":
                    post += 1
            except Exception as e:
                print(e)
                print(text, len(text), sent.start_char, sent.end_char, pre, post)
                exit()
            output['sent_offsets'].append((sent.start_char + pre, sent.end_char - post))
    for token in doc:
        # context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
        output['word'].append(token.text)
        output['offsets'].append((token.idx, token.idx + len(token.text)))
    return output


def normalize_answer(s):
    """Lower text and remove punctuation, storys and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def find_ref_span(sent_offsets, target):
    start, end = target
    ref_start = -1
    ref_end = -1
    # ref_start_idx = -1
    # ref_end_idx = -1
    for i, (sent_start, sent_end) in enumerate(sent_offsets):
        # print(target, sent_start, sent_end)
        if start >= sent_start and start <= sent_end:
            ref_start = sent_start
            # ref_start_idx = i
        if end >= sent_start and end <= sent_end:
            ref_end = sent_end
            # ref_end_idx = i
    assert ref_end >= ref_start and ref_end >= 0 and ref_start >= 0, "ref span is wrong {}".format((ref_start, ref_end))
    return ref_start, ref_end


def find_span_with_gt(context, offsets, ground_truth):
    best_f1 = 0.0
    best_span = (len(offsets) - 1, len(offsets) - 1)
    gt = normalize_answer(ground_truth).split()

    ls = [i for i in range(len(offsets))
          if context[offsets[i][0]:offsets[i][1]].lower() in gt]

    best_pred = None
    for i in range(len(ls)):
        for j in range(i, len(ls)):
            pred = normalize_answer(context[offsets[ls[i]][0]: offsets[ls[j]][1]]).split()
            common = Counter(pred) & Counter(gt)
            num_same = sum(common.values())
            if num_same > 0:
                precision = 1.0 * num_same / len(pred)
                recall = 1.0 * num_same / len(gt)
                f1 = (2 * precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_pred = pred
                    best_f1 = f1
                    best_span = (ls[i], ls[j])
    return best_span, best_pred


def find_span(offsets, start, end):
    start_index = end_index = -1
    for i, offset in enumerate(offsets):
        if (start_index < 0) or (start >= offset[0]):
            start_index = i
        if (end_index < 0) and (end <= offset[1]):
            end_index = i
    return (start_index, end_index)


def format_data(dataset, max_history, outfile):
    paragraph_lens = []
    question_lens = []
    # paragraphs = []
    examples = []
    with open(outfile, 'w') as fout:
        for paragraph in dataset['data']:
            history = []
            for qas in paragraph['qas']:
                temp = []
                n_history = len(history) if max_history < 0 else min(max_history, len(history))
                if n_history > 0:
                    for i, (q, a) in enumerate(history[-n_history:]):
                        d = n_history - i
                        temp.append('<Q{}>'.format(d))
                        temp.extend(q)
                        temp.append('<A{}>'.format(d))
                        temp.extend(a)
                temp.append('<Q>')
                temp.extend(qas['annotated_question']['word'])
                history.append((qas['annotated_question']['word'], qas['annotated_answer']['word']))
                qas['annotated_question']['word'] = temp
                examples.append(qas)
                question_lens.append(len(qas['annotated_question']['word']))
                paragraph_lens.append(len(paragraph['annotated_context']['word']))
                ref_span = qas['ref_span']
                span_targets = qas['span_targets']
                targets_in_span = (qas['answer_span'][0] - span_targets[0],
                                   qas['answer_span'][1] - span_targets[0])

                question = qas['annotated_question']
                answers = [qas['answer']]
                spans = [qas['span']]
                if 'additional_answers' in qas:
                    answers = answers + qas['additional_answers']
                    spans = spans + qas['additional_spans']
                # print(targets_in_span, span_targets)
                assert (targets_in_span[0] >= 0
                        and targets_in_span[1] >= 0
                        and targets_in_span[0] <= len(qas['annotated_span']['word'])
                        and targets_in_span[1] <= len(qas['annotated_span']['word']))

                assert (span_targets[0] >= 0
                        and span_targets[1] >= 0
                        and span_targets[0] <= len(paragraph['annotated_context']['word'])
                        and span_targets[1] <= len(paragraph['annotated_context']['word']))

                sample = {'id': (paragraph['id'], qas['turn_id']),
                          'question_word': question['word'],
                          # 'question_offsets': [x for sublist in question['offsets'],
                          'question_offsets': question['offsets'],
                          'answers': answers,
                          'spans': spans,
                          'context_word': paragraph['annotated_context']['word'],
                          # 'context_offsets': [x for sublist in paragraph['annotated_context']['offsets'] for x in sublist],
                          'context_offsets': paragraph['annotated_context']['offsets'],
                          'span_word': qas['annotated_span']['word'],
                          'span_offsets': qas['annotated_span']['offsets'],
                          'raw_context': paragraph['context'],
                          'new_ans': None if 'new_ans' not in qas else qas['new_ans'],
                          'ref_span': ref_span,
                          'span_targets': span_targets,
                          'targets_in_span': targets_in_span,
                          'targets': qas['answer_span']}
                fout.write('{}\n'.format(json.dumps(sample)))


def process_file(data_file, output_file, n_history):

    with open(data_file, 'r') as f:
        dataset = json.load(f)

    data = []
    start_time = time.time()
    for i, datum in enumerate(dataset['data']):
        if i % 10 == 0:
            print('processing %d / %d (used_time = %.2fs)...' %
                  (i, len(dataset['data']), time.time() - start_time))
        context_str = datum['story']
        _datum = {'context': context_str,
                  'source': datum['source'],
                  'id': datum['id'],
                  'filename': datum['filename']}
        _datum['annotated_context'] = process(context_str, require_sents=True)
        _datum['qas'] = []
        _datum['context'] += UNK
        _datum['annotated_context']['word'].append(UNK)
        _datum['annotated_context']['offsets'].append(
            (len(_datum['context']) - len(UNK), len(_datum['context'])))
        _datum['annotated_context']['sent_offsets'].append((len(_datum['context']) - len(UNK), len(_datum['context'])))

        assert len(datum['questions']) == len(datum['answers'])

        additional_answers = {}
        additional_spans = {}
        if 'additional_answers' in datum:
            for k, answer in datum['additional_answers'].items():
                if len(answer) == len(datum['answers']):
                    for ex in answer:
                        idx = ex['turn_id']
                        if idx not in additional_answers:
                            additional_answers[idx] = []
                            additional_spans[idx] = []
                        additional_answers[idx].append(ex['input_text'])
                        additional_spans[idx].append(ex['span_text'].strip())

        for question, answer in zip(datum['questions'], datum['answers']):
            # remove space in span text
            text, offset_s, offset_e = trip_space(answer['span_text'])
            answer['span_text'] = text
            answer['span_start'] += offset_s
            answer['span_end'] += offset_e

            assert question['turn_id'] == answer['turn_id']
            idx = question['turn_id']
            _qas = {'turn_id': idx,
                    'question': question['input_text'],
                    'span': answer['span_text'],
                    'answer': answer['input_text']}
            if idx in additional_answers:
                _qas['additional_answers'] = additional_answers[idx]
                _qas['additional_spans'] = additional_spans[idx]

            _qas['annotated_question'] = process(question['input_text'])
            _qas['annotated_answer'] = process(answer['input_text'])

            _qas['answer_span_start'] = answer['span_start']
            _qas['answer_span_end'] = answer['span_end']

            _qas['annotated_span'] = process(answer['span_text'])
            _qas['span'] += UNK
            _qas['annotated_span']['word'].append(UNK)
            _qas['annotated_span']['offsets'].append(
                (len(_qas['span']) - len(UNK), len(_qas['span'])))

            start = answer['span_start']
            end = answer['span_end']
            chosen_text = _datum['context'][start: end].lower()

            while len(chosen_text) > 0 and chosen_text[0] in string.whitespace:
                chosen_text = chosen_text[1:]
                start += 1
            while len(chosen_text) > 0 and chosen_text[-1] in string.whitespace:
                chosen_text = chosen_text[:-1]
                end -= 1
            input_text = _qas['answer'].strip().lower()
            if input_text in chosen_text:
                i = chosen_text.find(input_text)
                _qas['answer_span'] = find_span(_datum['annotated_context']['offsets'],
                                                start + i, start + i + len(input_text))
                span_targets = find_span(_datum['annotated_context']['offsets'],
                                         start, start + len(chosen_text))
                # char index
                _qas['ref_span'] = (_qas['answer_span_start'], _qas['answer_span_end'])
                # token index
                _qas['span_targets'] = span_targets
            else:
                _qas['answer_span'], _qas['new_ans'] = find_span_with_gt(_datum['context'],
                                                                         _datum['annotated_context']['offsets'],
                                                                         input_text)
                sent_offsets = _datum['annotated_context']['sent_offsets']
                answer_ch_span = (_datum['annotated_context']['offsets'][_qas['answer_span'][0]][0],
                                  _datum['annotated_context']['offsets'][_qas['answer_span'][1]][1])
                # new_answer_text = _datum['annotated_context']['word'][_qas['answer_span'][0]:_qas['answer_span'][1]+1]
                ref_span = find_ref_span(sent_offsets, answer_ch_span)
                span_targets = find_span(_datum['annotated_context']['offsets'],
                                         ref_span[0], ref_span[1])
                new_span = _datum['context'][ref_span[0]:ref_span[1]]
                assert new_span[0].strip() != '', "{} {}".format(repr(new_span), ref_span)
                if new_span.strip().lower() == 'unknown':
                    continue
                _qas['annotated_span'] = process(new_span)
                _qas['span'] = new_span
                # _qas['span'] += UNK
                # _qas['annotated_span']['word'].append(UNK)
                # _qas['annotated_span']['offsets'].append(
                #     (len(_qas['span']) - len(UNK), len(_qas['span'])))
                # char index
                _qas['ref_span'] = ref_span
                # token index
                _qas['span_targets'] = span_targets
            assert (_qas['answer_span'][0] >= _qas['span_targets'][0]
                    and _qas['answer_span'][1] >= _qas['span_targets'][0]), "{} {}".format(
                            _qas['answer_span'], _qas['span_targets'])
            _datum['qas'].append(_qas)
        data.append(_datum)

    dataset['data'] = data
    with open(output_file + '.ori', 'w') as fout:
        json.dump(dataset, fout, sort_keys=True, indent=4)
    format_data(dataset, n_history, output_file)
    return dataset


def preprocess(args):
    import spacy
    global _NLP
    _NLP = spacy.load('en', parser=False)
    process_file(
        args.raw_devset_file,
        args.devset_file,
        args.n_history)

    process_file(
        args.raw_trainset_file,
        args.trainset_file,
        args.n_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_trainset_file', '-rt', type=str, required=True)
    parser.add_argument('--raw_devset_file', '-rd', type=str, required=True)
    parser.add_argument('--trainset_file', '-t', type=str, required=True)
    parser.add_argument('--devset_file', '-d', type=str, required=True)
    parser.add_argument('--n_history', '-n', type=int, required=True)
    args = parser.parse_args()
    print(args)
    dataset = preprocess(args)
