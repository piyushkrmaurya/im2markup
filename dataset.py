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

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, src_field, tgt_field):

    src = src_field.tokenize(example["src"])

    unk = src_field.unk_token
    pad = src_field.pad_token
    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]

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
    def __init__(
        self, fields, readers, data, dirs, sort_key, filter_pred=None, corpus_id=None
    ):
        self.sort_key = sort_key
        can_copy = "src_map" in fields and "alignment" in fields

        read_iters = [
            r.read(dat[1], dat[0], dir_) for r, dat, dir_ in zip(readers, data, dirs)
        ]

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

                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_field.base_field, tgt_field.base_field
                )
                self.src_vocabs.append(src_ex_vocab)
            ex_fields = {k: [(k, v)] for k, v in fields.items() if k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):

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

    global max_src_in_batch, max_tgt_in_batch

    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)

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

        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):

                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:

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
        self.num_batches_multiple = max(opt.accum_count)
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

    def __iter__(self):
        num_batches = 0
        paths = self._paths
        if self.is_train and self.repeat:

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

            for path in paths:
                for batch in self._iter_dataset(path):
                    yield batch
                    num_batches += 1
                    if num_batches % self.num_batches_multiple == 0:
                        return


class IterOnDevice:
    def __init__(self, iterable, device_id):
        self.iterable = iterable
        self.device_id = device_id

    @staticmethod
    def batch_to_device(batch, device_id):

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
