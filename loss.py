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

    if eval_only:
        yield filter_shard_state(state)
    else:

        non_none = dict(filter_shard_state(state, shard_size))

        keys, values = zip(
            *(
                (k, [v_chunk for v_chunk in v_split])
                for k, (_, v_split) in non_none.items()
            )
        )

        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

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

        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))
