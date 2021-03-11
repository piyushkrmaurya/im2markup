import os
from functools import partial

import cv2
import six
import torch
from PIL import Image
from torchtext.data import Field, RawField
from torchvision import transforms

from itertools import islice, repeat


def text_sort_key(ex):

    if hasattr(ex, "tgt"):
        return len(ex.src[0]), len(ex.tgt[0])
    return len(ex.src[0])


def img_sort_key(ex):

    return ex.src.size(2), ex.src.size(1)


def _read_file(path):
    with open(path, "rb") as f:
        for line in f:
            yield line


def _split_corpus(path, shard_size):

    with open(path, "rb") as f:
        if shard_size <= 0:
            yield f.readlines()
        else:
            while True:
                shard = list(islice(f, shard_size))
                if not shard:
                    break
                yield shard


def split_corpus(path, shard_size, default=None):

    if path is not None:
        return _split_corpus(path, shard_size)
    else:
        return repeat(default)


class ImageDataReader:
    def __init__(self, truncate=None, channel_size=3):
        self._check_deps()
        self.truncate = truncate
        self.channel_size = channel_size

    @classmethod
    def from_opt(cls, opt):
        return cls(channel_size=opt.image_channel_size)

    @classmethod
    def _check_deps(cls):
        if any([Image is None, transforms is None, cv2 is None]):
            cls._raise_missing_dep("PIL", "torchvision", "cv2")

    def read(self, images, side, img_dir=None):

        if isinstance(images, str):
            images = _read_file(images)

        for i, filename in enumerate(images):
            filename = filename.decode("utf-8").strip()
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                img_path = filename

            assert os.path.exists(img_path), "img path %s not found" % filename

            if self.channel_size == 1:
                img = transforms.ToTensor()(Image.fromarray(cv2.imread(img_path, 0)))
            else:
                img = Image.open(img_path).convert("RGB")
                img = transforms.ToTensor()(img)
            if self.truncate and self.truncate != (0, 0):
                if not (
                    img.size(1) <= self.truncate[0] and img.size(2) <= self.truncate[1]
                ):
                    continue
            yield {side: img, side + "_path": filename, "indices": i}


class TextMultiField(RawField):
    def __init__(self, base_name, base_field, feats_fields):
        super(TextMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    def process(self, batch, device=None):

        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:

            base_data, lengths = base_data

        feats = [
            ff.process(batch_by_feat[i], device=device)
            for i, (_, ff) in enumerate(self.fields[1:], 1)
        ]
        levels = [base_data] + feats

        data = torch.stack(levels, 2)
        if self.base_field.include_lengths:
            return data, lengths
        else:
            return data

    def preprocess(self, x):

        return [f.preprocess(x) for _, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


def _feature_tokenize(string, layer=0, tok_delim=None, feat_delim=None, truncate=None):

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


def text_fields(**kwargs):

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize, layer=i, truncate=truncate, feat_delim=feat_delim
        )
        use_len = i == 0 and include_lengths
        feat = Field(
            init_token=bos,
            eos_token=eos,
            pad_token=pad,
            tokenize=tokenize,
            include_lengths=use_len,
        )
        fields_.append((name, feat))
    assert fields_[0][0] == base_name
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field


def batch_img(data, vocab):

    c = data[0].size(0)
    h = max([t.size(1) for t in data])
    w = max([t.size(2) for t in data])
    imgs = torch.zeros(len(data), c, h, w).fill_(1)
    for i, img in enumerate(data):
        imgs[i, :, 0 : img.size(1), 0 : img.size(2)] = img
    return imgs


def image_fields(**kwargs):
    img = Field(
        use_vocab=False, dtype=torch.float, postprocessing=batch_img, sequential=False
    )
    return img


class TextDataReader:
    def read(self, sequences, side, _dir=None):
        assert _dir is None or _dir == "", "Cannot use _dir with TextDataReader."
        if isinstance(sequences, str):
            sequences = _read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}

    @classmethod
    def from_opt(cls, opt):
        return cls()
