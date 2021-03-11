import torch

from collections import defaultdict
from torchtext.data import Field, RawField
from data_reader import text_fields, image_fields
from copy import deepcopy

from loguru import logger


def _old_style_vocab(vocab):

    return isinstance(vocab, list) and any(isinstance(v[1], Vocab) for v in vocab)


def _old_style_field_list(vocab):

    return (
        (not _old_style_vocab(vocab))
        and _old_style_nesting(vocab)
        and (not isinstance(vocab["tgt"][0][1], TextMultiField))
    )


def _old_style_nesting(vocab):

    return isinstance(vocab, dict) and any(isinstance(v, list) for v in vocab.values())


def get_fields(
    src_data_type,
    n_src_feats,
    n_tgt_feats,
    pad="<blank>",
    bos="<s>",
    eos="</s>",
    dynamic_dict=False,
    with_align=False,
    src_truncate=None,
    tgt_truncate=None,
):

    assert src_data_type in ["text", "img", "audio", "vec"], "Data type not implemented"
    assert (
        not dynamic_dict or src_data_type == "text"
    ), "it is not possible to use dynamic_dict with non-text input"
    fields = {}

    fields_getters = {"text": text_fields, "img": image_fields}

    src_field_kwargs = {
        "n_feats": n_src_feats,
        "include_lengths": True,
        "pad": pad,
        "bos": None,
        "eos": None,
        "truncate": src_truncate,
        "base_name": "src",
    }
    fields["src"] = fields_getters[src_data_type](**src_field_kwargs)

    tgt_field_kwargs = {
        "n_feats": n_tgt_feats,
        "include_lengths": False,
        "pad": pad,
        "bos": bos,
        "eos": eos,
        "truncate": tgt_truncate,
        "base_name": "tgt",
    }
    fields["tgt"] = fields_getters["text"](**tgt_field_kwargs)

    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = indices

    corpus_ids = Field(use_vocab=True, sequential=False)
    fields["corpus_id"] = corpus_ids

    return fields


def old_style_vocab(vocab):

    return (
        _old_style_vocab(vocab)
        or _old_style_field_list(vocab)
        or _old_style_nesting(vocab)
    )


def filter_example(
    ex,
    use_src_len=True,
    use_tgt_len=True,
    min_src_len=1,
    max_src_len=float("inf"),
    min_tgt_len=1,
    max_tgt_len=float("inf"),
):

    src_len = len(ex.src[0])
    tgt_len = len(ex.tgt[0])
    return (not use_src_len or min_src_len <= src_len <= max_src_len) and (
        not use_tgt_len or min_tgt_len <= tgt_len <= max_tgt_len
    )


def load_old_vocab(vocab, data_type="text", dynamic_dict=False):

    if _old_style_vocab(vocab):

        vocab = dict(vocab)
        n_src_features = sum("src_feat_" in k for k in vocab)
        n_tgt_features = sum("tgt_feat_" in k for k in vocab)
        fields = get_fields(
            data_type, n_src_features, n_tgt_features, dynamic_dict=dynamic_dict
        )
        for n, f in fields.items():
            try:
                f_iter = iter(f)
            except TypeError:
                f_iter = [(n, f)]
            for sub_n, sub_f in f_iter:
                if sub_n in vocab:
                    sub_f.vocab = vocab[sub_n]
        return fields

    if _old_style_field_list(vocab):

        fields = vocab
        for base_name, vals in fields.items():
            if (base_name == "src" and data_type == "text") or base_name == "tgt":
                assert not isinstance(vals[0][1], TextMultiField)
                fields[base_name] = [
                    (base_name, TextMultiField(vals[0][0], vals[0][1], vals[1:]))
                ]

    if _old_style_nesting(vocab):

        fields = dict(list(chain.from_iterable(vocab.values())))

    return fields


def _load_vocab(vocab_path, name, counters, min_freq):

    vocab = _read_vocab_file(vocab_path, name)
    vocab_size = len(vocab)
    logger.info("Loaded %s vocab has %d tokens." % (name, vocab_size))
    for i, token in enumerate(vocab):

        counters[name][token] = vocab_size - i + min_freq
    return vocab, vocab_size


def _build_field_vocab(field, counter, size_multiple=1, **kwargs):

    all_specials = [field.unk_token, field.pad_token, field.init_token, field.eos_token]
    specials = [tok for tok in all_specials if tok is not None]
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
    if size_multiple > 1:
        _pad_vocab_to_multiple(field.vocab, size_multiple)


def _build_fv_from_multifield(multifield, counters, build_fv_args, size_multiple=1):
    for name, field in multifield:
        _build_field_vocab(
            field, counters[name], size_multiple=size_multiple, **build_fv_args[name]
        )
        logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))


def _build_fields_vocab(
    fields,
    counters,
    data_type,
    share_vocab,
    vocab_size_multiple,
    src_vocab_size,
    src_words_min_frequency,
    tgt_vocab_size,
    tgt_words_min_frequency,
    subword_prefix="▁",
    subword_prefix_is_joiner=False,
):
    build_fv_args = defaultdict(dict)
    build_fv_args["src"] = dict(
        max_size=src_vocab_size, min_freq=src_words_min_frequency
    )
    build_fv_args["tgt"] = dict(
        max_size=tgt_vocab_size, min_freq=tgt_words_min_frequency
    )
    tgt_multifield = fields["tgt"]
    _build_fv_from_multifield(
        tgt_multifield,
        counters,
        build_fv_args,
        size_multiple=vocab_size_multiple if not share_vocab else 1,
    )

    if fields.get("corpus_id", False):
        fields["corpus_id"].vocab = fields["corpus_id"].vocab_cls(counters["corpus_id"])

    if data_type == "text":
        src_multifield = fields["src"]
        _build_fv_from_multifield(
            src_multifield,
            counters,
            build_fv_args,
            size_multiple=vocab_size_multiple if not share_vocab else 1,
        )

        if share_vocab:

            logger.info(" * merging src and tgt vocab...")
            src_field = src_multifield.base_field
            tgt_field = tgt_multifield.base_field
            _merge_field_vocabs(
                src_field,
                tgt_field,
                vocab_size=src_vocab_size,
                min_freq=src_words_min_frequency,
                vocab_size_multiple=vocab_size_multiple,
            )
            logger.info(" * merged vocab size: %d." % len(src_field.vocab))

        build_noise_field(
            src_multifield.base_field,
            subword_prefix=subword_prefix,
            is_joiner=subword_prefix_is_joiner,
        )
    return fields


def build_vocab(
    train_dataset_files,
    fields,
    data_type,
    share_vocab,
    src_vocab_path,
    src_vocab_size,
    src_words_min_frequency,
    tgt_vocab_path,
    tgt_vocab_size,
    tgt_words_min_frequency,
    vocab_size_multiple=1,
):

    counters = defaultdict(Counter)

    if src_vocab_path:
        try:
            logger.info("Using existing vocabulary...")
            vocab = torch.load(src_vocab_path)

            return vocab
        except torch.serialization.pickle.UnpicklingError:
            logger.info("Building vocab from text file...")

            train_dataset_files = []

    if src_vocab_path:
        src_vocab, src_vocab_size = _load_vocab(
            src_vocab_path, "src", counters, src_words_min_frequency
        )
    else:
        src_vocab = None

    if tgt_vocab_path:
        tgt_vocab, tgt_vocab_size = _load_vocab(
            tgt_vocab_path, "tgt", counters, tgt_words_min_frequency
        )
    else:
        tgt_vocab = None

    for i, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for name, field in fields.items():
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(f_iter, all_data):
                    has_vocab = (sub_n == "src" and src_vocab) or (
                        sub_n == "tgt" and tgt_vocab
                    )
                    if sub_f.sequential and not has_vocab:
                        val = fd
                        counters[sub_n].update(val)

        if i < len(train_dataset_files) - 1:
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    fields = _build_fields_vocab(
        fields,
        counters,
        data_type,
        share_vocab,
        vocab_size_multiple,
        src_vocab_size,
        src_words_min_frequency,
        tgt_vocab_size,
        tgt_words_min_frequency,
    )

    return fields
