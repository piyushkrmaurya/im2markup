import torch
from torch import nn

from utils import Statistics


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(
            *(
                (k, [v_chunk for v_chunk in v_split])
                for k, (_, v_split) in non_none.items()
            )
        )

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(
                    zip(
                        torch.split(state[k], shard_size),
                        [v_chunk.grad for v_chunk in v_split],
                    )
                )
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)


class NMTLossCompute(nn.Module):
    """
    Standard NMT Loss Computation.
    """

    def __init__(
        self,
        criterion,
        generator,
        normalization="sents",
        lambda_coverage=0.0,
        lambda_align=0.0,
    ):
        super(NMTLossCompute, self).__init__()
        self.criterion = criterion
        self.generator = generator
        self.lambda_coverage = lambda_coverage
        self.lambda_align = lambda_align

    @classmethod
    def from_opt(cls, model, tgt_field, opt, train=True, device="cpu"):
        """
        Returns a LossCompute subclass which wraps around an nn.Module subclass
        (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
        object allows this loss to be computed in shards and passes the relevant
        data to a Statistics object which handles training/validation logging.
        Currently, the NMTLossCompute class handles all loss computation except
        for when using a copy mechanism.
        """

        padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
        unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction="sum")

        loss_gen = model.generator
        compute = cls(criterion, loss_gen)
        compute.to(device)

        return compute

    def _make_shard_state(self, batch, output, range_, attns=None):
        shard_state = {
            "output": output,
            "target": batch.tgt[range_[0] + 1 : range_[1], :, 0],
        }
        return shard_state

    def _compute_loss(
        self,
        batch,
        output,
        target,
        std_attn=None,
        coverage_attn=None,
        align_head=None,
        ref_align=None,
    ):

        bottled_output = self._bottle(output)

        scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)

        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def __call__(
        self,
        batch,
        output,
        attns,
        normalization=1.0,
        shard_size=0,
        trunc_start=0,
        trunc_size=None,
    ):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, **shard_state)
            return loss / float(normalization), stats
        batch_stats = Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

