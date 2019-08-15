""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, dcn_encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = dcn_encoder
        self.decoder = decoder
        self.translator = None

    def set_translator(self, translator):
        self.translator = translator

    def set_examples(self, trainset_examples, devset_examples):
        self.trainset_examples = trainset_examples
        self.devset_examples = devset_examples

    def set_vocabs(self, trainset_vocabs, devset_vocabs):
        self.trainset_vocabs = trainset_vocabs
        self.devset_vocabs = devset_vocabs

    def set_drqa_model(self, drqa_model):
        self.drqa_model = drqa_model

    def reverse(self, translation_batch):
        return self.translator.reverse(self.trainset_vocabs, self.trainset_examples, translation_batch)

    def drqa_predict(self, doc, que, target):
        results = self.drqa_model.predict(doc=doc, que=que, target=target)
        return results

    def forward(self, batch, src, history, tgt, src_lengths, history_lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, src_lengths = self.encoder(src, history, src_lengths, history_lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)

        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=src_lengths)
        vocabs = self.trainset_vocabs if self.train() else self.devset_vocabs
        results = self.translator.translate_batch(batch, vocabs, False, training=self.train())
        return dec_out, attns, results
