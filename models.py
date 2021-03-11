import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def all_equal(*args):

    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(
        arg == first for arg in arguments
    ), "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):

    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )


class Cast(nn.Module):
    def __init__(self, dtype):
        super(Cast, self).__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)


class Elementwise(nn.ModuleList):
    def __init__(self, *args):
        super(Elementwise, self).__init__(*args)

    def forward(self, inputs):
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        assert len(self) == len(inputs_)
        outputs = [f(x) for f, x in zip(self, inputs_)]
        return torch.cat(outputs, 2)


class Embeddings(nn.Module):
    def __init__(
        self,
        word_vec_size,
        word_vocab_size,
        word_padding_idx,
        feat_vec_exponent=0.7,
        feat_vec_size=-1,
        feat_padding_idx=[],
        feat_vocab_sizes=[],
        dropout=0,
        sparse=False,
        fix_word_vecs=False,
    ):

        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        feat_dims = [int(vocab ** feat_vec_exponent) for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [
            nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
            for vocab, dim, pad in emb_params
        ]
        emb_lookups = Elementwise(embeddings)

        self.embedding_size = sum(emb_dims)

        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module("emb_lookups", emb_lookups)

    @classmethod
    def from_opt(cls, opt, text_field):

        pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
        word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

        num_embs = [len(f.vocab) for _, f in text_field]
        num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

        return cls(
            word_vec_size=opt.tgt_word_vec_size,
            feat_vec_exponent=opt.feat_vec_exponent,
            feat_vec_size=opt.feat_vec_size,
            dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            word_padding_idx=word_padding_idx,
            feat_padding_idx=feat_pad_indices,
            word_vocab_size=num_word_embeddings,
            feat_vocab_sizes=num_feat_embeddings,
            sparse=opt.optim == "sparseadam",
        )

    @property
    def word_lookup(self):

        return self.make_embedding[0][0]

    @property
    def emb_lookups(self):

        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file):

        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lookup.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lookup.weight.data.copy_(pretrained[:, : self.word_vec_size])
            else:
                self.word_lookup.weight.data.copy_(pretrained)

    def forward(self, source, step=None):

        source = self.make_embedding(source)

        return source


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()

        self.dim = dim

        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)

    def score(self, h_t, h_s):

        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        all_equal(src_batch, tgt_batch)
        all_equal(src_dim, tgt_dim)
        all_equal(self.dim, src_dim)

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)

        return torch.bmm(h_t, h_s_)

    def forward(self, source, memory_bank, memory_lengths=None):

        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        all_equal(batch, batch_)
        all_equal(dim, dim_)
        all_equal(self.dim, dim)

        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)
            align.masked_fill_(~mask, -float("inf"))

        align_vectors = F.softmax(align.view(batch * target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        c = torch.bmm(align_vectors, memory_bank)

        concat_c = torch.cat([c, source], 2).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            batch_, dim_ = attn_h.size()
            all_equal(batch, batch_)
            all_equal(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            all_equal(batch, batch_)
            all_equal(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            target_l_, batch_, dim_ = attn_h.size()
            all_equal(target_l, target_l_)
            all_equal(batch, batch_)
            all_equal(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            all_equal(target_l, target_l_)
            all_equal(batch, batch_)
            all_equal(source_l, source_l_)

        return attn_h, align_vectors


class ImageEncoder(nn.Module):
    def __init__(
        self, num_layers, bidirectional, rnn_size, dropout, image_chanel_size=3
    ):
        super(ImageEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        self.layer1 = nn.Conv2d(
            image_chanel_size, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
        )
        self.layer2 = nn.Conv2d(
            64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
        )
        self.layer3 = nn.Conv2d(
            128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
        )
        self.layer4 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
        )
        self.layer5 = nn.Conv2d(
            256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
        )
        self.layer6 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
        )

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)

        src_size = 512
        dropout = dropout[0] if type(dropout) is list else dropout
        self.rnn = nn.LSTM(
            src_size,
            int(rnn_size / self.num_directions),
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.pos_lookup = nn.Embedding(1000, src_size)

    def forward(self, src, lengths=None):

        batch_size = src.size(0)

        src = F.relu(self.layer1(src[:, :, :, :] - 0.5), True)

        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        src = F.relu(self.layer2(src), True)

        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        src = F.relu(self.batch_norm1(self.layer3(src)), True)

        src = F.relu(self.layer4(src), True)

        src = F.max_pool2d(src, kernel_size=(1, 2), stride=(1, 2))

        src = F.relu(self.batch_norm2(self.layer5(src)), True)

        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))

        src = F.relu(self.batch_norm3(self.layer6(src)), True)

        all_outputs = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2).transpose(1, 2)
            row_vec = torch.Tensor(batch_size).type_as(inp.data).long().fill_(row)
            pos_emb = self.pos_lookup(row_vec)
            with_pos = torch.cat(
                (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0
            )
            outputs, hidden_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)

        return hidden_t, out, lengths

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)


class InputFeedRNNDecoder(nn.Module):
    def __init__(
        self,
        rnn_type,
        bidirectional_encoder,
        num_layers,
        hidden_size,
        dropout=0.0,
        embeddings=None,
    ):
        super(InputFeedRNNDecoder, self).__init__()

        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        self.state = {}

        self.rnn = self._build_rnn(
            rnn_type,
            input_size=self._input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.attn = GlobalAttention(hidden_size)

    def init_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(hidden):

            if self.bidirectional_encoder:
                hidden = torch.cat(
                    [hidden[0 : hidden.size(0) : 2], hidden[1 : hidden.size(0) : 2]], 2
                )
            return hidden

        if isinstance(encoder_final, tuple):
            self.state["hidden"] = tuple(
                _fix_enc_hidden(enc_hid) for enc_hid in encoder_final
            )
        else:
            self.state["hidden"] = (_fix_enc_hidden(encoder_final),)

        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = (
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        )

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank, memory_lengths=None, step=None, **kwargs):

        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths
        )

        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)

        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return dec_outs, attns

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):

        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        all_equal(tgt_batch, input_feed_batch)

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3

        dec_state = self.state["hidden"]

        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)

            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths,
            )
            attns["std"].append(p_attn)

            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        return StackedLSTM(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):

        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):

        dec_in = tgt[:-1]

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank, memory_lengths=lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)


def build_model(model_opt, opt, fields, checkpoint, logs=True):
    if logs:
        logger.info("Building model...")

    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    encoder = ImageEncoder(
        opt.enc_layers,
        opt.brnn,
        opt.enc_rnn_size,
        opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        opt.image_channel_size if isinstance(opt.image_channel_size, int) else 3,
    )

    tgt_field = fields["tgt"]
    tgt_emb = Embeddings.from_opt(model_opt, tgt_field)
    decoder = InputFeedRNNDecoder(
        opt.rnn_type,
        opt.brnn,
        opt.dec_layers,
        opt.dec_rnn_size,
        opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        tgt_emb,
    )

    model = NMTModel(encoder, decoder)

    generator = nn.Sequential(
        nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].base_field.vocab)),
        Cast(torch.float32),
        nn.LogSoftmax(dim=-1),
    )
    if model_opt.share_decoder_embeddings:
        generator[0].weight = decoder.embeddings.word_lookup.weight

    if checkpoint is not None:

        def fix_key(s):
            s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.b_2", r"\1.layer_norm\2.bias", s)
            s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.a_2", r"\1.layer_norm\2.weight", s)
            return s

        checkpoint["model"] = {fix_key(k): v for k, v in checkpoint["model"].items()}

        model.load_state_dict(checkpoint["model"], strict=False)
        generator.load_state_dict(checkpoint["generator"], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, "embeddings"):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc
            )
        if hasattr(model.decoder, "embeddings"):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec
            )

    model.generator = generator

    model.to(opt.device)

    if logs:
        logger.info(model)

    return model
