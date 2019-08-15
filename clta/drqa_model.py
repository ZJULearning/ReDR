import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from tqdm import tqdm
from .layers import SeqAttnMatch, StackedBRNN, LinearSeqAttn, BilinearSeqAttn
from .layers import weighted_avg, uniform_weights, dropout
from .eval_utils import compute_eval_metric
from .cove import MTLSTM
import CONSTANTS as CONST
_NLP = None
try:
    import spacy
    _NLP = spacy.load('en', parser=False)
except Exception as e:
    logging.info("import error {}".format(str(e)))


def reverse(vocab, data):
    rep_str = []
    rep_list = []
    for ex in data:
        l_ex = [vocab.itos[idx] for idx in ex if vocab.itos[idx] not in (
            CONST.PAD_TOKEN, CONST.SOS_TOKEN, CONST.EOS_TOKEN)]
        s_ex = ' '.join(l_ex)
        rep_list.append(l_ex)
        rep_str.append(s_ex)
    return rep_list, rep_str


class DrQA(nn.Module):
    """Network for the Document Reader module of DrQA."""
    _RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, vocab, args):
        """Configuration, word embeddings"""
        super(DrQA, self).__init__()
        # print(vocab.keys())
        # Store config
        self.args = args
        self.best_metrics = None
        self.metrics_history = {'f1': [], 'em': []}
        self.gpu = False
        self.vocab = vocab
        self.vocab_size = len(vocab)
        if self.vocab_size != args.vocab_size:
            logging.warn(
                    "required vocab_size is not equal to real vocab_size({} vs {})".format(
                        args.vocab_size, self.vocab_size))
        self.w_embedding = nn.Embedding(self.vocab_size, vocab.vectors.size(1))
        if vocab.vectors is not None:
            self.w_embedding.weight.data.copy_(vocab.vectors)
        input_w_dim = self.w_embedding.embedding_dim
        if args.fix_embeddings:
            for p in self.w_embedding.parameters():
                p.requires_grad = False

        self.mt_cove = None
        if args.use_cove:
            self.mt_cove = MTLSTM()
            input_w_dim = input_w_dim + 600
            for p in self.mt_cove.parameters():
                p.requires_grad = True
        q_input_size = input_w_dim

        # Projection for attention weighted question
        if self.args.use_qemb:
            self.qemb_match = SeqAttnMatch(input_w_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = input_w_dim
        if self.args.use_qemb:
            doc_input_size += input_w_dim

        # Project document and question to the same size as their encoders
        if self.args.resize_rnn_input:
            self.doc_linear = nn.Linear(doc_input_size, args.hidden_size, bias=True)
            self.q_linear = nn.Linear(input_w_dim, args.hidden_size, bias=True)
            doc_input_size = q_input_size = args.hidden_size

        # RNN document encoder
        self.doc_rnn = StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            variational_dropout=args.variational_dropout,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self._RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
            bidirectional=True,
        )

        # RNN question encoder
        self.question_rnn = StackedBRNN(
            input_size=q_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            variational_dropout=args.variational_dropout,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self._RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
            bidirectional=True,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.num_layers
            question_hidden_size *= args.num_layers

        if args.doc_self_attn:
            self.doc_self_attn = SeqAttnMatch(doc_hidden_size)
            doc_hidden_size = doc_hidden_size + question_hidden_size

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % args.question_merge)
        if args.question_merge == 'self_attn':
            self.self_attn = LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )
        q_rep_size = question_hidden_size + doc_hidden_size if args.span_dependency else question_hidden_size
        self.end_attn = BilinearSeqAttn(
            doc_hidden_size,
            q_rep_size,
        )

        if args.fix_cove and self.mt_cove is not None:
            for p in self.mt_cove.parameters():
                p.requires_grad = False

    def load(self, path):
        try:
            self.load_state_dict(torch.load(path))
        except Exception as e:
            print(e)

    def forward(self, **kwargs):
        if 'batch' in kwargs:
            batch = kwargs['batch']
            batch_que, que_lengths = batch.question
            batch_doc, doc_lengths = batch.context
            batch_offsets = batch.context_offsets
            batch_tgt = batch.targets
            raw_doc = None
        else:
            batch_que, que_lengths = kwargs['question']
            batch_doc, doc_lengths, raw_doc = kwargs['context']
            batch_offsets = kwargs['context_offsets']
            batch_tgt = kwargs.get('targets')

        # Embed both document and question
        xq_emb = self.w_embedding(batch_que)                         # (batch, max_q_len, word_embed)
        xd_emb = self.w_embedding(batch_doc)                         # (batch, max_d_len, word_embed)

        shared_axes = [2] if self.args.word_dropout else []
        xq_emb = dropout(xq_emb, self.args.dropout_emb, shared_axes=shared_axes, training=self.training)
        xd_emb = dropout(xd_emb, self.args.dropout_emb, shared_axes=shared_axes, training=self.training)

        xd_mask = torch.ones_like(batch_doc, dtype=torch.uint8)
        for i, q in enumerate(doc_lengths):
            xd_mask[i, :q].fill_(0)

        xq_mask = torch.ones_like(batch_que, dtype=torch.uint8)
        for i, q in enumerate(que_lengths):
            xq_mask[i, :q].fill_(0)

        # -------------------------
        if self.mt_cove is not None:
            xq_emb_c = xq_emb
            xd_emb_c = xd_emb

            xq_emb_cove = self.mt_cove(xq_emb_c, mask=xq_mask)
            xq_emb = torch.cat([xq_emb, xq_emb_cove], -1)
            xd_emb_cove = self.mt_cove(xd_emb_c, mask=xd_mask)
            xd_emb = torch.cat([xd_emb_c, xd_emb_cove], -1)
        # ----------------------------

        # Add attention-weighted question representation
        if self.args.use_qemb:
            xq_weighted_emb = self.qemb_match(xd_emb, xq_emb, xq_mask)
            drnn_input = torch.cat([xd_emb, xq_weighted_emb], 2)
        else:
            drnn_input = xd_emb

        # Project document and question to the same size as their encoders
        if self.args.resize_rnn_input:
            drnn_input = F.relu(self.doc_linear(drnn_input))
            xq_emb = F.relu(self.q_linear(xq_emb))
            if self.args.dropout_ff > 0:
                drnn_input = F.dropout(drnn_input, training=self.training)
                xq_emb = F.dropout(xq_emb, training=self.training)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, xd_mask)       # (batch, max_d_len, hidden_size)

        # Document self attention
        if self.args.doc_self_attn:
            xd_weighted_emb = self.doc_self_attn(doc_hiddens, doc_hiddens, xd_mask)
            doc_hiddens = torch.cat([doc_hiddens, xd_weighted_emb], 2)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(xq_emb, xq_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = uniform_weights(question_hiddens, xq_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens.contiguous(), xq_mask)
        question_hidden = weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        # doc_hiddens: batch, len, hidden_size * x
        # question_hidden: batch, hidden_size * x
        start_scores = self.start_attn(doc_hiddens, question_hidden, xd_mask)
        if self.args.span_dependency:
            question_hidden = torch.cat([question_hidden, (doc_hiddens * start_scores.exp().unsqueeze(2)).sum(1)], 1)
        end_scores = self.end_attn(doc_hiddens, question_hidden, xd_mask)
        # score: batch, length
        predictions, spans = self.extract_predictions(batch_doc, batch_offsets, start_scores, end_scores, raw_doc)

        return {'score_s': start_scores,
                'score_e': end_scores,
                'predictions': predictions,
                'targets': batch_tgt}

    def evaluate(self, devset, out_predictions=False):
        # Train/Eval mode
        self.train(False)
        # Run forward
        metrics = {
            'f1': 0.0,
            'em': 0.0,
            'loss': 0.0
        }
        is_best = False
        f1 = []
        em = []
        total_predictions = []
        for i, batch in tqdm(enumerate(devset), total=len(devset)):
            if self.args.debug and i >= 10:
                break
            res = self.forward(batch=batch)
            batch_que, que_lengths = batch.question
            batch_doc, doc_lengths = batch.context
            batch_offsets = batch.context_offsets
            batch_answers = batch.answers

            context_str = reverse(self.vocab, batch_doc)[1]
            question_str = reverse(self.vocab, batch_que)[1]

            score_s, score_e = res['score_s'], res['score_e']
            predictions = res['predictions']
            for i, (context, question, answers, pred) in enumerate(zip(context_str, question_str,
                                                                       batch_answers, predictions)):
                total_predictions.append({
                    'passage': context,
                    'question': question,
                    'answers': answers,
                    'prediction': pred})

            _pred, _spans = self.extract_predictions(batch_doc, batch_offsets, score_s, score_e)
            _f1, _em = self.evaluate_predictions(predictions, batch_answers)
            f1.append(_f1)
            em.append(_em)
        metrics['f1'] = sum(f1) / len(f1)
        metrics['em'] = sum(em) / len(em)
        self.metrics_history['f1'].append(metrics['f1'])
        self.metrics_history['em'].append(metrics['em'])
        self.train(True)
        if self.best_metrics is None or metrics['f1'] > self.best_metrics['f1']:
            self.best_metrics = metrics
            is_best = True
        return {'predictions': total_predictions}, metrics, is_best, self.metrics_history

    def extract_predictions(self, context, context_offsets, score_s, score_e, raw_doc=None):
        # Transfer to CPU/normal tensors for numpy ops (and convert log probabilities to probabilities)
        batch_size = score_s.size(0)
        score_s = score_s.exp().view(batch_size, -1)
        score_e = score_e.exp().view(batch_size, -1)

        predictions = []
        spans = []
        if raw_doc is None:
            _, context_text = reverse(self.vocab, context)
        else:
            context_text = raw_doc
        for i, (_s, _e) in enumerate(zip(score_s, score_e)):
            if self.args.predict_raw_text:
                prediction, span = self._scores_to_raw_text(context_text[i],
                                                            context_offsets[i], _s, _e)
            else:
                prediction, span = self._scores_to_text(context_text[i], _s, _e)
            predictions.append(prediction)
            spans.append(span)
        return predictions, spans

    def _scores_to_text(self, text, score_s, score_e):
        max_len = self.args.max_answer_len or score_s.size(1)
        scores = torch.ger(score_s.squeeze(), score_e.squeeze())
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return ' '.join(text[s_idx: e_idx + 1]), (int(s_idx), int(e_idx))

    def _scores_to_raw_text(self, raw_text, offsets, score_s, score_e):
        max_len = self.args.max_answer_len or score_s.size(1)
        scores = torch.ger(score_s, score_e)
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return raw_text[offsets[s_idx][0]: offsets[e_idx][1]], (offsets[s_idx][0], offsets[e_idx][1])

    def evaluate_predictions(self, predictions, answers):
        f1_score = compute_eval_metric('f1', predictions, answers)
        em_score = compute_eval_metric('em', predictions, answers)
        return f1_score, em_score

    # @staticmethod
    def calc_loss(self, results, batch):
        targets = batch.targets
        score_s = results['score_s']
        score_e = results['score_e']
        assert targets.size(0) == score_s.size(0) == score_e.size(0)
        loss = F.nll_loss(score_s, targets[:, 0]) + F.nll_loss(score_e, targets[:, 1])
        return loss

    def predict(self, doc, que, target=None):
        global _NLP
        assert _NLP is not None, "_NLP is None, whether spacy is available?"
        tokenized_doc = {'word': [], 'offsets': []}
        tokenized_que = {'word': [], 'offsets': []}
        for token in _NLP(doc):
            tokenized_doc['word'].append(token.text)
            tokenized_doc['offsets'].append((token.idx, token.idx + len(token.text)))
        for token in _NLP(que):
            tokenized_que['word'].append(token.text)
            tokenized_que['offsets'].append((token.idx, token.idx + len(token.text)))

        doc_ids = [self.vocab.stoi[w] for w in tokenized_doc['word']]
        offsets = tokenized_doc['offsets']
        que_ids = [self.vocab.stoi[w] for w in tokenized_que['word']]

        doc_tensor = torch.tensor([doc_ids], dtype=torch.long)
        doc_length = torch.tensor([len(doc_ids)], dtype=torch.long)
        que_tensor = torch.tensor([que_ids], dtype=torch.long)
        que_length = torch.tensor([len(que_ids)], dtype=torch.long)
        offsets_tensor = torch.tensor([offsets], dtype=torch.long)
        if self.gpu:
            doc_tensor = doc_tensor.cuda()
            doc_length = doc_length.cuda()
            que_tensor = que_tensor.cuda()
            que_length = que_length.cuda()
            offsets_tensor = offsets_tensor.cuda()

        results = self.forward(
                question=(que_tensor, que_length),
                context=(doc_tensor, doc_length, [doc]),
                context_offsets=offsets_tensor)
        if target is not None:
            results['targets'] = target
            f1_score = compute_eval_metric('f1', results["predictions"], [[target]])
            results['f1'] = f1_score

        return results
