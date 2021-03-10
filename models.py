import torch
import torch.nn as nn
import torch.nn.functional as F


def all_equal(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(
        arg == first for arg in arguments
    ), "Not all arguments have the same value: " + str(args)


class Cast(nn.Module):
    """
    Basic layer that casts its input to a specific data type. The same tensor
    is returned if the data type is already correct.
    """

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Tensor whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, inputs):
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        assert len(self) == len(inputs_)
        outputs = [f(x) for f, x in zip(self, inputs_)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class Embeddings(nn.Module):
    """Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feat_padding_idx (List[int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes (List[int], optional): list of size of dictionary
            of embeddings for each feature.
        feat_merge (string): merge action for the features embeddings:
            concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
            embedding size is N^feat_dim_exponent, where N is the
            number of values the feature takes.
        feat_vec_size (int): embedding dimension for features when using
            `-feat_merge mlp`
        dropout (float): dropout probability.
    """

    def __init__(
        self,
        word_vec_size,
        word_vocab_size,
        word_padding_idx,
        feat_merge="concat",
        feat_vec_exponent=0.7,
        feat_vec_size=-1,
        feat_padding_idx=[],
        feat_vocab_sizes=[],
        dropout=0,
        sparse=False,
        fix_word_vecs=False,
    ):
    
        self._validate_args(
            feat_merge,
            feat_vocab_sizes,
            feat_vec_exponent,
            feat_vec_size,
            feat_padding_idx,
        )

        if feat_padding_idx is None:
            feat_padding_idx = []
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == "sum":
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent) for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [
            nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
            for vocab, dim, pad in emb_params
        ]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = sum(emb_dims) if feat_merge == "concat" else word_vec_size

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module("emb_luts", emb_luts)

        if feat_merge == "mlp" and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            mlp = nn.Sequential(nn.Linear(in_dim, word_vec_size), nn.ReLU())
            self.make_embedding.add_module("mlp", mlp)


        if fix_word_vecs:
            self.word_lut.weight.requires_grad = False


    @classmethod
    def from_opt(cls, opt, text_field):

        pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
        word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

        num_embs = [len(f.vocab) for _, f in text_field]
        num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

        return cls(
            word_vec_size=opt.tgt_word_vec_size,
            feat_merge=opt.feat_merge,
            feat_vec_exponent=opt.feat_vec_exponent,
            feat_vec_size=opt.feat_vec_size,
            dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            word_padding_idx=word_padding_idx,
            feat_padding_idx=feat_pad_indices,
            word_vocab_size=num_word_embeddings,
            feat_vocab_sizes=num_feat_embeddings,
            sparse=opt.optim == "sparseadam",
        )

    def _validate_args(
        self,
        feat_merge,
        feat_vocab_sizes,
        feat_vec_exponent,
        feat_vec_size,
        feat_padding_idx,
    ):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                warnings.warn(
                    "Merging with sum, but got non-default "
                    "feat_vec_exponent. It will be unused."
                )
            if feat_vec_size != -1:
                warnings.warn(
                    "Merging with sum, but got non-default "
                    "feat_vec_size. It will be unused."
                )
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                warnings.warn(
                    "Not merging with sum and positive "
                    "feat_vec_size, but got non-default "
                    "feat_vec_exponent. It will be unused."
                )
        else:
            if feat_vec_exponent <= 0:
                raise ValueError(
                    "Using feat_vec_exponent to determine "
                    "feature vec size, but got feat_vec_exponent "
                    "less than or equal to 0."
                )
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            raise ValueError(
                "Got unequal number of feat_vocab_sizes and "
                "feat_padding_idx ({:d} != {:d})".format(n_feats, len(feat_padding_idx))
            )

    @property
    def word_lut(self):
        """Word look-up table."""
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """Embedding look-up table."""
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """

        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data.copy_(pretrained[:, : self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)

    def forward(self, source, step=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """

        source = self.make_embedding(source)

        return source


class GlobalAttention(nn.Module):
    r"""
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
       attn_func (str): attention function to use, options [softmax,sparsemax]

    """

    def __init__(self, dim, coverage=False, attn_type="dot", attn_func="softmax"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in [
            "dot",
            "general",
            "mlp",
        ], "Please select a valid attention type (got {:s}).".format(attn_type)
        self.attn_type = attn_type
        assert attn_func in [
            "softmax",
            "sparsemax",
        ], "Please select a valid attention function."
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        all_equal(src_batch, tgt_batch)
        all_equal(src_dim, tgt_dim)
        all_equal(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        """

        Args:
          source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          memory_bank (FloatTensor): source vectors ``(batch, src_len, dim)``
          memory_lengths (LongTensor): the source context lengths ``(batch,)``
          coverage (FloatTensor): None (not supported yet)

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(tgt_len, batch, dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, src_len)``
        """

        # one step input
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
        if coverage is not None:
            batch_, source_l_ = coverage.size()
            all_equal(batch, batch_)
            all_equal(source_l, source_l_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = torch.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float("inf"))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch * target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch * target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            all_equal(batch, batch_)
            all_equal(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            all_equal(batch, batch_)
            all_equal(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
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
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :class:`Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    """A simple encoder CNN -> RNN for image src.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
    """

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
        self.pos_lut = nn.Embedding(1000, src_size)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
     
        image_channel_size = opt.image_channel_size if isinstance(opt.image_channel_size, int) else 3

        return cls(
            opt.enc_layers,
            opt.brnn,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            image_channel_size,
        )

    def load_pretrained_vectors(self, opt):
        """Pass in needed options only when modify function definition."""
        pass

    def forward(self, src, lengths=None):
        """See :func:`encoders.encoder.EncoderBase.forward()`"""

        batch_size = src.size(0)
        # (batch_size, 64, imgH, imgW)
        # layer 1
        src = F.relu(self.layer1(src[:, :, :, :] - 0.5), True)

        # (batch_size, 64, imgH/2, imgW/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        # (batch_size, 128, imgH/2, imgW/2)
        # layer 2
        src = F.relu(self.layer2(src), True)

        # (batch_size, 128, imgH/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        #  (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer 3
        # batch norm 1
        src = F.relu(self.batch_norm1(self.layer3(src)), True)

        # (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer4
        src = F.relu(self.layer4(src), True)

        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(1, 2), stride=(1, 2))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        # layer 5
        # batch norm 2
        src = F.relu(self.batch_norm2(self.layer5(src)), True)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.relu(self.batch_norm3(self.layer6(src)), True)

        # # (batch_size, 512, H, W)
        all_outputs = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2).transpose(1, 2)
            row_vec = torch.Tensor(batch_size).type_as(inp.data).long().fill_(row)
            pos_emb = self.pos_lut(row_vec)
            with_pos = torch.cat(
                (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0
            )
            outputs, hidden_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)

        return hidden_t, out, lengths

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout

    

    def _check_args(self, src, lengths=None, hidden=None):
        n_batch = src.size(1)
        if lengths is not None:
            (n_batch_,) = lengths.size()
            all_equal(n_batch, n_batch_)


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

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
    """Input feeding based decoder.

    See :class:`~decoders.decoder.RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[memory_bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """
    """Base recurrent attention-based decoder class.

    Specifies the interface used by different decoder types
    and required by :class:`~models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[memory_bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :class:`~modules.GlobalAttention`
       attn_func (str) : see :class:`~modules.GlobalAttention`
       coverage_attn (str): see :class:`~modules.GlobalAttention`
       context_gate (str): see :class:`~modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (modules.Embeddings): embedding module to use
       reuse_copy_attn (bool): reuse the attention for copying
       copy_attn_type (str): The copy attention style. See
        :class:`~modules.GlobalAttention`.
    """

    def __init__(
        self,
        rnn_type,
        bidirectional_encoder,
        num_layers,
        hidden_size,
        attn_type="general",
        attn_func="softmax",
        dropout=0.0,
        embeddings=None
    ):
        super(InputFeedRNNDecoder, self).__init__()


        self.attentional = attn_type != "none" and attn_type is not None
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(
            rnn_type,
            input_size=self._input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )     

        self.attn = GlobalAttention(
            hidden_size,
            attn_type=attn_type,
            attn_func=attn_func,
        )
     

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings
        )

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""

        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat(
                    [hidden[0 : hidden.size(0) : 2], hidden[1 : hidden.size(0) : 2]], 2
                )
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(
                _fix_enc_hidden(enc_hid) for enc_hid in encoder_final
            )
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final),)

        # Init the input feed.
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
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """

        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths
        )

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
     
        

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return dec_outs, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.embeddings.update_dropout(dropout)



    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        all_equal(tgt_batch, input_feed_batch)
        # END Additional args check.

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []
    

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
    
        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            if self.attentional:
                decoder_output, p_attn = self.attn(
                    rnn_output,
                    memory_bank.transpose(0, 1),
                    memory_lengths=memory_lengths,
                )
                attns["std"].append(p_attn)
            else:
                decoder_output = rnn_output
           
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]


        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        stacked_cell = StackedLSTM
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (encoders.EncoderBase): an encoder object
      decoder (decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(
            dec_in, memory_bank, memory_lengths=lengths
        )
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
