import glob
from collections import Counter
from itertools import chain, cycle, islice, repeat, starmap

import torch
import torchtext.data.batch
from loguru import logger
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example, Field, Iterator, RawField
from torchtext.vocab import Vocab


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.
    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, src_field, tgt_field):
    """Create copy-vocab and numericalize with it.
    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.
    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.
    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src = src_field.tokenize(example["src"])
    # make a small vocab containing just the tokens in the source sequence
    unk = src_field.unk_token
    pad = src_field.pad_token
    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
    example["src_map"] = src_map
    example["src_ex_vocab"] = src_ex_vocab

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx]
        )
        example["alignment"] = mask
    return src_ex_vocab, example


class Dataset(TorchtextDataset):
    """Contain data and process it.
    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.
    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.
    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(
        self, fields, readers, data, dirs, sort_key, filter_pred=None, corpus_id=None
    ):
        self.sort_key = sort_key
        can_copy = "src_map" in fields and "alignment" in fields

        read_iters = [
            r.read(dat[1], dat[0], dir_) for r, dat, dir_ in zip(readers, data, dirs)
        ]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            if corpus_id is not None:
                ex_dict["corpus_id"] = corpus_id
            else:
                ex_dict["corpus_id"] = "train"
            if can_copy:
                src_field = fields["src"]
                tgt_field = fields["tgt"]
                # this assumes src_field and tgt_field are both text
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_field.base_field, tgt_field.base_field
                )
                self.src_vocabs.append(src_ex_vocab)
            ex_fields = {k: [(k, v)] for k, v in fields.items() if k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if "fields" not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    @staticmethod
    def config(fields):
        readers, data, dirs = [], [], []
        for name, field in fields:
            if field["data"] is not None:
                readers.append(field["reader"])
                data.append((name, field["data"]))
                dirs.append(field["dir"])
        return readers, data, dirs


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def _pool(
    data,
    batch_size,
    batch_size_fn,
    batch_size_multiple,
    sort_key,
    random_shuffler,
    pool_factor,
):
    for p in torchtext.data.batch(
        data, batch_size * pool_factor, batch_size_fn=batch_size_fn
    ):
        p_batch = list(
            batch_iter(
                sorted(p, key=sort_key),
                batch_size,
                batch_size_fn=batch_size_fn,
                batch_size_multiple=batch_size_multiple,
            )
        )
        for b in random_shuffler(p_batch):
            yield b


def batch_iter(data, batch_size, batch_size_fn=None, batch_size_multiple=1):
    """Yield elements from data in chunks of batch_size, where each chunk size
    is a multiple of batch_size_multiple.

    This is an extended version of torchtext.data.batch.
    """
    if batch_size_fn is None:

        def batch_size_fn(new, count, sofar):
            return count

    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far >= batch_size:
            overflowed = 0
            if size_so_far > batch_size:
                overflowed += 1
            if batch_size_multiple > 1:
                overflowed += (len(minibatch) - overflowed) % batch_size_multiple
            if overflowed == 0:
                yield minibatch
                minibatch, size_so_far = [], 0
            else:
                if overflowed == len(minibatch):
                    logger.warning(
                        "The batch will be filled until we reach %d,"
                        "its size may exceed %d tokens"
                        % (batch_size_multiple, batch_size)
                    )
                else:
                    yield minibatch[:-overflowed]
                    minibatch = minibatch[-overflowed:]
                    size_so_far = 0
                    for i, ex in enumerate(minibatch):
                        size_so_far = batch_size_fn(ex, i + 1, size_so_far)
    if minibatch:
        yield minibatch


class OrderedIterator(Iterator):
    def __init__(
        self,
        dataset,
        batch_size,
        pool_factor=1,
        batch_size_multiple=1,
        yield_raw_example=False,
        **kwargs
    ):
        super(OrderedIterator, self).__init__(dataset, batch_size, **kwargs)
        self.batch_size_multiple = batch_size_multiple
        self.yield_raw_example = yield_raw_example
        self.dataset = dataset
        self.pool_factor = pool_factor

    def create_batches(self):
        if self.train:
            if self.yield_raw_example:
                self.batches = batch_iter(
                    self.data(), 1, batch_size_fn=None, batch_size_multiple=1
                )
            else:
                self.batches = _pool(
                    self.data(),
                    self.batch_size,
                    self.batch_size_fn,
                    self.batch_size_multiple,
                    self.sort_key,
                    self.random_shuffler,
                    self.pool_factor,
                )
        else:
            self.batches = []
            for b in batch_iter(
                self.data(),
                self.batch_size,
                batch_size_fn=self.batch_size_fn,
                batch_size_multiple=self.batch_size_multiple,
            ):
                self.batches.append(sorted(b, key=self.sort_key))

    def __iter__(self):
        """
        Extended version of the definition in torchtext.data.Iterator.
        Added yield_raw_example behaviour to yield a torchtext.data.Example
        instead of a torchtext.data.Batch object.
        """
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a
                    # minibatch be sorted by decreasing order, which
                    #  requires reversing relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                if self.yield_raw_example:
                    yield minibatch[0]
                else:
                    yield torchtext.data.Batch(minibatch, self.dataset, self.device)
            if not self.repeat:
                return


class DatasetIterator:
    """Yield data from sharded dataset files.

    Args:
        dataset_paths: a list containing the locations of dataset files.
        fields (dict[str, Field]): fields dict for the
            datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: See :class:`OrderedIterator` ``device``.
        is_train (bool): train or valid?
    """

    def __init__(self, corpus_type, fields, opt, is_train=True, multi=False):

        dataset_glob = opt.data + "." + corpus_type + ".[0-9]*.pt"
        dataset_paths = list(
            sorted(glob.glob(dataset_glob), key=lambda p: int(p.split(".")[-2]))
        )

        if not dataset_paths:
            if is_train:
                raise ValueError("Training data %s not found" % dataset_glob)
            else:
                return None

        self._paths = dataset_paths
        self.fields = fields
        self.batch_size = opt.batch_size if is_train else opt.valid_batch_size
        self.batch_size_fn = (
            max_tok_len if is_train and opt.batch_type == "tokens" else None
        )
        self.batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
        self.device = "cpu"
        self.is_train = is_train
        self.repeat = not opt.single_pass
        self.num_batches_multiple = max(opt.accum_count) * opt.world_size
        self.yield_raw_example = multi
        self.pool_factor = opt.pool_factor

    def _iter_dataset(self, path):
        logger.info("Loading dataset from %s" % path)
        cur_dataset = torch.load(path)
        logger.info("number of examples: %d" % len(cur_dataset))
        cur_dataset.fields = self.fields
        cur_iter = OrderedIterator(
            dataset=cur_dataset,
            batch_size=self.batch_size,
            pool_factor=self.pool_factor,
            batch_size_multiple=self.batch_size_multiple,
            batch_size_fn=self.batch_size_fn,
            device=self.device,
            train=self.is_train,
            sort=False,
            sort_within_batch=True,
            repeat=False,
            yield_raw_example=self.yield_raw_example,
        )
        for batch in cur_iter:
            self.dataset = cur_iter.dataset
            yield batch

        # NOTE: This is causing some issues for consumer/producer,
        # as we may still have some of those examples in some queue
        # cur_dataset.examples = None
        # gc.collect()
        # del cur_dataset
        # gc.collect()

    def __iter__(self):
        num_batches = 0
        paths = self._paths
        if self.is_train and self.repeat:
            # Cycle through the shards indefinitely.
            paths = cycle(paths)
        for path in paths:
            for batch in self._iter_dataset(path):
                yield batch
                num_batches += 1
        if (
            self.is_train
            and not self.repeat
            and num_batches % self.num_batches_multiple != 0
        ):
            # When the dataset is not repeated, we might need to ensure that
            # the number of returned batches is the multiple of a given value.
            # This is important for multi GPU training to ensure that all
            # workers have the same number of batches to process.
            for path in paths:
                for batch in self._iter_dataset(path):
                    yield batch
                    num_batches += 1
                    if num_batches % self.num_batches_multiple == 0:
                        return


class IterOnDevice:
    """Sent items from `iterable` on `device_id` and yield."""

    def __init__(self, iterable, device_id):
        self.iterable = iterable
        self.device_id = device_id

    @staticmethod
    def batch_to_device(batch, device_id):
        """Move `batch` to `device_id`, cpu if `device_id` < 0."""
        curr_device = batch.indices.device
        device = torch.device(device_id) if device_id >= 0 else torch.device("cpu")
        if curr_device != device:
            if isinstance(batch.src, tuple):
                batch.src = tuple([_.to(device) for _ in batch.src])
            else:
                batch.src = batch.src.to(device)
            batch.tgt = batch.tgt.to(device)
            batch.indices = batch.indices.to(device)
            batch.alignment = (
                batch.alignment.to(device) if hasattr(batch, "alignment") else None
            )
            batch.src_map = (
                batch.src_map.to(device) if hasattr(batch, "src_map") else None
            )
            batch.align = batch.align.to(device) if hasattr(batch, "align") else None
            batch.corpus_id = (
                batch.corpus_id.to(device) if hasattr(batch, "corpus_id") else None
            )

    def __iter__(self):
        for batch in self.iterable:
            self.batch_to_device(batch, self.device_id)
            yield batch
