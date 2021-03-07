import codecs
import io
import os
import sys
from collections import Counter
from itertools import chain

import cv2
import torch
import torchtext

from constants import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD


class DatasetBase(torchtext.data.Dataset):
    """
    A dataset basically supports iteration over all the examples
    it contains. We currently have 3 datasets inheriting this base
    for 3 types of corpus respectively: "text", "img", "audio".
    Internally it initializes an `torchtext.data.Dataset` object with
    the following attributes:
     `examples`: a sequence of `torchtext.data.Example` objects.
     `fields`: a dictionary associating str keys with `torchtext.data.Field`
        objects, and not necessarily having the same keys as the input fields.
    """

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(DatasetBase, self).__reduce_ex__(proto)

    def load_fields(self, vocab_dict):
        """ Load fields from vocab.pt, and set the `fields` attribute.
        Args:
            vocab_dict (dict): a dict of loaded vocab from vocab.pt file.
        """
        fields = onmt.inputters.inputter.load_fields_from_vocab(
            vocab_dict.items(), self.data_type
        )
        self.fields = dict(
            [(k, f) for (k, f) in fields.items() if k in self.examples[0].__dict__]
        )

    @staticmethod
    def extract_text_features(tokens):
        """
        Args:
            tokens: A list of tokens, where each token consists of a word,
                optionally followed by u"￨"-delimited features.
        Returns:
            A sequence of words, a sequence of features, and num of features.
        """
        if not tokens:
            return [], [], -1

        specials = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
        words = []
        features = []
        n_feats = None
        for token in tokens:
            split_token = token.split(u"￨")
            assert all(
                [special != split_token[0] for special in specials]
            ), "Dataset cannot contain Special Tokens"

            if split_token[0]:
                words += [split_token[0]]
                features += [split_token[1:]]

                if n_feats is None:
                    n_feats = len(split_token)
                else:
                    assert (
                        len(split_token) == n_feats
                    ), "all words must have the same number of features"
        features = list(zip(*features))
        return tuple(words), features, n_feats - 1

    # Below are helper functions for intra-class use only.

    def _join_dicts(self, *args):
        """
        Args:
            dictionaries with disjoint keys.
        Returns:
            a single dictionary that has the union of these keys.
        """
        return dict(chain(*[d.items() for d in args]))

    def _peek(self, seq):
        """
        Args:
            seq: an iterator.
        Returns:
            the first thing returned by calling next() on the iterator
            and an iterator created by re-chaining that value to the beginning
            of the iterator.
        """
        first = next(seq)
        return first, chain([first], seq)

    def _construct_example_fromlist(self, data, fields):
        """
        Args:
            data: the data to be set as the value of the attributes of
                the to-be-created `Example`, associating with respective
                `Field` objects with same key.
            fields: a dict of `torchtext.data.Field` objects. The keys
                are attributes of the to-be-created `Example`.
        Returns:
            the created `Example` object.
        """
        ex = torchtext.data.Example()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
            else:
                setattr(ex, name, val)
        return ex


class ImageDataset(DatasetBase):
    """ Dataset for data_type=='img'
        Build `Example` objects, `Field` objects, and filter_pred function
        from image corpus.
        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            tgt_seq_length (int): maximum target sequence length.
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def __init__(
        self,
        fields,
        src_examples_iter,
        tgt_examples_iter,
        num_src_feats=0,
        num_tgt_feats=0,
        tgt_seq_length=0,
        use_filter_pred=True,
        image_channel_size=3,
    ):
        self.data_type = "img"

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        self.image_channel_size = image_channel_size
        if tgt_examples_iter is not None:
            examples_iter = (
                self._join_dicts(src, tgt)
                for src, tgt in zip(src_examples_iter, tgt_examples_iter)
            )
        else:
            examples_iter = src_examples_iter

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None) for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (
            self._construct_example_fromlist(ex_values, out_fields)
            for ex_values in example_values
        )
        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        out_examples = list(out_examples)

        def filter_pred(example):
            """ ? """
            if tgt_examples_iter is not None:
                return 0 < len(example.tgt) <= tgt_seq_length
            else:
                return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(ImageDataset, self).__init__(out_examples, out_fields, filter_pred)

    def sort_key(self, ex):
        """ Sort using the size of the image: (width, height)."""
        return (ex.src.size(2), ex.src.size(1))

    @staticmethod
    def make_image_examples_nfeats_tpl(
        img_iter, img_path, img_dir, image_channel_size=3
    ):
        """
        Note: one of img_iter and img_path must be not None
        Args:
            img_iter(iterator): an iterator that yields pairs (img, filename)
                (or None)
            img_path(str): location of a src file containing image paths
                (or None)
            src_dir (str): location of source images
        Returns:
            (example_dict iterator, num_feats) tuple
        """
        if img_iter is None:
            if img_path is not None:
                img_iter = ImageDataset.make_img_iterator_from_file(
                    img_path, img_dir, image_channel_size
                )
            else:
                raise ValueError(
                    """One of 'img_iter' and 'img_path'
                                    must be not None"""
                )
        examples_iter = ImageDataset.make_examples(img_iter, img_dir, "src")
        num_feats = 0  # Source side(img) has no features.

        return (examples_iter, num_feats)

    @staticmethod
    def make_examples(img_iter, src_dir, side, truncate=None):
        """
        Args:
            path (str): location of a src file containing image paths
            src_dir (str): location of source images
            side (str): 'src' or 'tgt'
            truncate: maximum img size ((0,0) or None for unlimited)
        Yields:
            a dictionary containing image data, path and index for each line.
        """
        assert (src_dir is not None) and os.path.exists(
            src_dir
        ), "src_dir must be a valid directory if data_type is img"

        for index, (img, filename) in enumerate(img_iter):
            if truncate and truncate != (0, 0):
                if not (img.size(1) <= truncate[0] and img.size(2) <= truncate[1]):
                    continue

            example_dict = {side: img, side + "_path": filename, "indices": index}
            yield example_dict

    @staticmethod
    def make_img_iterator_from_file(path, src_dir, image_channel_size=3):
        """
        Args:
            path(str):
            src_dir(str):
        Yields:
            img: and image tensor
            filename(str): the image filename
        """
        from PIL import Image
        from torchvision import transforms

        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                filename = line.strip()
                img_path = os.path.join(src_dir, filename)
                if not os.path.exists(img_path):
                    img_path = line

                assert os.path.exists(img_path), "img path %s not found" % (
                    line.strip()
                )

                if image_channel_size == 1:
                    img = transforms.ToTensor()(
                        Image.fromarray(cv2.imread(img_path, 0))
                    )
                else:
                    img = transforms.ToTensor()(Image.open(img_path))

                yield img, filename

    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features: the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features: the number of target features to
                create `torchtext.data.Field` for.
        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        def make_img(data, vocab):
            """ ? """
            c = data[0].size(0)
            h = max([t.size(1) for t in data])
            w = max([t.size(2) for t in data])
            imgs = torch.zeros(len(data), c, h, w).fill_(1)
            for i, img in enumerate(data):
                imgs[i, :, 0 : img.size(1), 0 : img.size(2)] = img
            return imgs

        fields["src"] = torchtext.data.Field(
            use_vocab=False,
            dtype=torch.float,
            postprocessing=make_img,
            sequential=False,
        )

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD
        )

        for j in range(n_tgt_features):
            fields["tgt_feat_" + str(j)] = torchtext.data.Field(
                init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD
            )

        def make_src(data, vocab):
            """ ? """
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False,
            dtype=torch.float,
            postprocessing=make_src,
            sequential=False,
        )

        def make_tgt(data, vocab):
            """ ? """
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[: sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long, postprocessing=make_tgt, sequential=False
        )

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long, sequential=False
        )

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        For image corpus, source side is in form of image, thus
        no feature; while target side is in form of text, thus
        we can extract its text features.
        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.
        Returns:
            number of features on `side`.
        """
        if side == "src":
            num_feats = 0
        else:
            with codecs.open(corpus_file, "r", "utf-8") as cf:
                f_line = cf.readline().strip().split()
                _, _, num_feats = ImageDataset.extract_text_features(f_line)

        return num_feats


class TextDataset(DatasetBase):
    """ Dataset for data_type=='text'
        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.
        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def __init__(
        self,
        fields,
        src_examples_iter,
        tgt_examples_iter,
        num_src_feats=0,
        num_tgt_feats=0,
        src_seq_length=0,
        tgt_seq_length=0,
        dynamic_dict=True,
        use_filter_pred=True,
    ):
        self.data_type = "text"

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (
                self._join_dicts(src, tgt)
                for src, tgt in zip(src_examples_iter, tgt_examples_iter)
            )
        else:
            examples_iter = src_examples_iter

        if dynamic_dict:
            examples_iter = self._dynamic_dict(examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None) for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(ex_values, out_fields)
            src_size += len(example.src)
            out_examples.append(example)

        def filter_pred(example):
            """ ? """
            return (
                0 < len(example.src) <= src_seq_length
                and 0 < len(example.tgt) <= tgt_seq_length
            )

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(out_examples, out_fields, filter_pred)

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices.data[b]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                scores[:, b].index_add_(1, fill, scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)
        return scores

    @staticmethod
    def make_text_examples_nfeats_tpl(text_iter, text_path, truncate, side):
        """
        Args:
            text_iter(iterator): an iterator (or None) that we can loop over
                to read examples.
                It may be an openned file, a string list etc...
            text_path(str): path to file or None
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".
        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ["src", "tgt"]

        if text_iter is None:
            if text_path is not None:
                text_iter = TextDataset.make_text_iterator_from_file(text_path)
            else:
                return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = TextDataset.make_examples(text_iter, truncate, side)

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)

    @staticmethod
    def make_examples(text_iter, truncate, side):
        """
        Args:
            text_iter (iterator): iterator of text sequences
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".
        Yields:
            (word, features, nfeat) triples for each line.
        """
        for i, line in enumerate(text_iter):
            line = line.strip().split()
            if truncate:
                line = line[:truncate]

            words, feats, n_feats = TextDataset.extract_text_features(line)

            example_dict = {side: words, "indices": i}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f) for j, f in enumerate(feats))
            yield example_dict, n_feats

    @staticmethod
    def make_text_iterator_from_file(path):
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.
        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["src"] = torchtext.data.Field(pad_token=PAD_WORD, include_lengths=True)

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD
        )

        for j in range(n_tgt_features):
            fields["tgt_feat_" + str(j)] = torchtext.data.Field(
                init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD
            )

        def make_src(data, vocab):
            """ ? """
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False,
            dtype=torch.float,
            postprocessing=make_src,
            sequential=False,
        )

        def make_tgt(data, vocab):
            """ ? """
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[: sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long, postprocessing=make_tgt, sequential=False
        )

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long, sequential=False
        )

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.
        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.
        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            f_line = cf.readline().strip().split()
            _, _, num_feats = TextDataset.extract_text_features(f_line)

        return num_feats

    # Below are helper functions for intra-class use only.
    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(
                Counter(src), specials=[UNK_WORD, PAD_WORD]
            )
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor([0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example


def build_dataset(
    fields,
    src_data_iter=None,
    src_path=None,
    src_dir=None,
    tgt_data_iter=None,
    tgt_path=None,
    tgt_seq_length=0,
    tgt_seq_length_trunc=0,
    use_filter_pred=True,
    image_channel_size=3,
):

    src_examples_iter, num_src_feats = ImageDataset.make_image_examples_nfeats_tpl(
        src_data_iter, src_path, src_dir, image_channel_size
    )

    tgt_examples_iter, num_tgt_feats = TextDataset.make_text_examples_nfeats_tpl(
        tgt_data_iter, tgt_path, tgt_seq_length_trunc, "tgt"
    )

    dataset = ImageDataset(
        fields,
        src_examples_iter,
        tgt_examples_iter,
        num_src_feats,
        num_tgt_feats,
        tgt_seq_length=tgt_seq_length,
        use_filter_pred=use_filter_pred,
        image_channel_size=image_channel_size,
    )

    return dataset

