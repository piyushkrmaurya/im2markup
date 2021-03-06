import sys
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchtext
from torchvision import transforms

from PIL import Image

from beam import Beam, GNMTGlobalScorer
from embedding import Embeddings
from models import ImageEncoder, InputFeedRNNDecoder
from nmt import NMTModel
from preprocess import preprocess_image

warnings.filterwarnings("ignore")

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
UNK = 0
BOS_WORD = "<s>"
EOS_WORD = "</s>"

specials = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cpu"


def get_model(opt, vocab, checkpoint=None):

    image_channel_size = 3

    encoder = ImageEncoder(
        opt.enc_layers, opt.brnn, opt.rnn_size, opt.dropout, image_channel_size,
    )

    embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = vocab.stoi[PAD_WORD]
    num_word_embeddings = len(vocab)

    tgt_embeddings = Embeddings(
        word_vec_size=embedding_dim,
        word_vocab_size=num_word_embeddings,
        word_padding_idx=word_padding_idx,
        position_encoding=opt.position_encoding,
        dropout=opt.dropout,
        sparse=opt.optim == "sparseadam",
    )

    decoder = InputFeedRNNDecoder(
        opt.rnn_type,
        opt.brnn,
        opt.dec_layers,
        opt.rnn_size,
        attn_type=opt.global_attention,
        # opt.global_attention_function,
        attn_func="softmax",
        dropout=opt.dropout,
        embeddings=tgt_embeddings,
    )

    # Build NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)

    # Build Generator.
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(nn.Linear(opt.rnn_size, len(vocab)), gen_func)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)
        generator.load_state_dict(checkpoint["generator"], strict=False)
    else:
        if opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)

        model.decoder.embeddings.load_pretrained_vectors(
            opt.pre_word_vecs_dec, opt.fix_word_vecs_dec
        )

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator
    model.to(device)

    return model


def translate(img_path, model, vocab, n_best=3, beam_size=5, max_length=150):

    scorer = GNMTGlobalScorer()

    beam = [
        Beam(
            beam_size,
            n_best=n_best,
            pad=vocab.stoi[PAD_WORD],
            eos=vocab.stoi[EOS_WORD],
            bos=vocab.stoi[BOS_WORD],
            cuda=(str(device) == "cuda"),
            global_scorer=scorer,
        )
    ]

    img = preprocess_image(img_path)
    src = transforms.ToTensor()(img)

    src = src.unsqueeze(0)
    src = src.to(device)

    enc_states, memory_bank, _ = model.encoder(src)

    dec_states = model.decoder.init_decoder_state(src, memory_bank, enc_states)

    src_lengths = (
        torch.Tensor(1).type_as(memory_bank.data).long().fill_(memory_bank.size(0))
    )

    # (2) Repeat src objects `beam_size` times.
    memory_bank = torch.tensor(
        memory_bank.data.repeat(1, beam_size, 1), requires_grad=False
    )

    memory_lengths = src_lengths.repeat(beam_size)
    dec_states.repeat_beam_size_times(beam_size)

    # (3) run the decoder to generate sentences, using beam search.
    for i in range(max_length):
        if all((b.done() for b in beam)):
            break

        # Construct BATCH x beam_size nxt words.
        # Get all the pending current beam words and arrange for forward.
        inp = torch.tensor(
            torch.stack([b.get_current_state() for b in beam])
            .t()
            .contiguous()
            .view(1, -1),
            requires_grad=False,
        )

        # Temporary kludge solution to handle changed dim expectation
        # in the decoder
        inp = inp.unsqueeze(2)

        # Run one step.
        dec_out, dec_states, attn = model.decoder(
            inp, memory_bank, dec_states, memory_lengths=memory_lengths, step=i
        )

        # dec_out: beam x rnn_size
        dec_out = dec_out.squeeze(0)

        # (b) Compute a vector of batch x beam word scores.
        out = model.generator.forward(dec_out).data
        out = out.view(beam_size, 1, -1)

        # beam x tgt_vocab
        beam_attn = attn["std"].view(beam_size, 1, -1)

        # (c) Advance each beam.
        for j, b in enumerate(beam):
            b.advance(out[:, j], beam_attn.data[:, j, : memory_lengths[j]])
            dec_states.beam_update(j, b.get_current_origin(), beam_size)

    # (4) Extract sentences from beam.
    ret = {"predictions": [], "scores": [], "attention": []}
    for b in beam:
        n_best = n_best
        scores, ks = b.sort_finished(minimum=n_best)
        hyps, attn = [], []
        for i, (times, k) in enumerate(ks[:n_best]):
            hyp, att = b.get_hyp(times, k)
            hyps.append(hyp)
            attn.append(att)
        ret["predictions"].append(hyps)
        ret["scores"].append(scores)
        ret["attention"].append(attn)

    return ret


def generate_latex(img_path):
    # def load_test_model(opt, dummy_opt, model_path=None):
    model_path = "checkpoint.pt"
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    # del checkpoint["optim"]
    # del checkpoint["vocab"]
    # torch.save(checkpoint, "checkpoint.pt")

    vocab = dict(checkpoint["vocab"])["tgt"]
    vocab.stoi = defaultdict(lambda: 0, vocab.stoi)

    opt = checkpoint["opt"]

    model = get_model(opt, vocab, checkpoint=checkpoint)
    model.eval()
    model.generator.eval()

    # dict_keys(['predictions', 'scores', 'attention'])
    outputs = translate(img_path, model, vocab)

    if len(outputs["predictions"][0]) > 0:
        prediction = " ".join(
            [
                vocab.itos[x.item()]
                for x in outputs["predictions"][0][0]
                if vocab.itos[x.item()] not in specials
            ]
        )
        return prediction
    else:
        return "Ummm!"


if __name__ == "__main__":
    print(generate_latex(sys.argv[1]))
