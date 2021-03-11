import functools
from math import sqrt

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


def fn_args(fun):

    return inspect.getfullargspec(fun).args


def build_torch_optimizer(model, opt):

    params = [p for p in model.parameters() if p.requires_grad]
    betas = [opt.adam_beta1, opt.adam_beta2]
    if opt.optim == "sgd":
        optimizer = optim.SGD(params, lr=opt.learning_rate)
    elif opt.optim == "adagrad":
        optimizer = optim.Adagrad(
            params,
            lr=opt.learning_rate,
            initial_accumulator_value=opt.adagrad_accumulator_init,
        )
    elif opt.optim == "adadelta":
        optimizer = optim.Adadelta(params, lr=opt.learning_rate)
    elif opt.optim == "adam":
        optimizer = optim.Adam(params, lr=opt.learning_rate, betas=betas, eps=1e-9)
    else:
        raise ValueError("Invalid optimizer type: " + opt.optim)

    return optimizer


def make_learning_rate_decay_fn(opt):

    if opt.decay_method == "noam":
        return functools.partial(
            noam_decay, warmup_steps=opt.warmup_steps, model_size=opt.rnn_size
        )
    elif opt.decay_method == "noamwd":
        return functools.partial(
            noamwd_decay,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps,
        )
    elif opt.decay_method == "rsqrt":
        return functools.partial(rsqrt_decay, warmup_steps=opt.warmup_steps)
    elif opt.start_decay_steps is not None:
        return functools.partial(
            exponential_decay,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps,
        )


def noam_decay(step, warmup_steps, model_size):

    return model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def noamwd_decay(step, warmup_steps, model_size, rate, decay_steps, start_step=0):

    return (
        model_size ** (-0.5)
        * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        * rate ** (max(step - start_step + decay_steps, 0) // decay_steps)
    )


def exponential_decay(step, rate, decay_steps, start_step=0):

    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):

    return 1.0 / sqrt(max(step, warmup_steps))


class Optimizer(object):
    def __init__(
        self, optimizer, learning_rate, learning_rate_decay_fn=None, max_grad_norm=None
    ):

        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._training_step = 1
        self._decay_step = 1
        self._fp16 = None
        self._scaler = None

    @classmethod
    def from_opt(cls, model, opt, checkpoint=None):

        optim_opt = opt
        optim_state_dict = None

        if opt.train_from and checkpoint is not None:
            optim = checkpoint["optim"]
            ckpt_opt = checkpoint["opt"]
            ckpt_state_dict = {}
            if isinstance(optim, Optimizer):
                ckpt_state_dict["training_step"] = optim._step + 1
                ckpt_state_dict["decay_step"] = optim._step + 1
                ckpt_state_dict["optimizer"] = optim.optimizer.state_dict()
            else:
                ckpt_state_dict = optim

            if opt.reset_optim == "none":

                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
            elif opt.reset_optim == "all":

                pass
            elif opt.reset_optim == "states":

                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
                del optim_state_dict["optimizer"]
            elif opt.reset_optim == "keep_states":

                optim_state_dict = ckpt_state_dict

        optimizer = cls(
            build_torch_optimizer(model, optim_opt),
            optim_opt.learning_rate,
            learning_rate_decay_fn=make_learning_rate_decay_fn(optim_opt),
            max_grad_norm=optim_opt.max_grad_norm,
        )
        if opt.model_dtype == "fp16":
            if opt.optim == "fusedadam":
                optimizer._fp16 = "legacy"
            else:
                optimizer._fp16 = "amp"
                from torch.cuda.amp import GradScaler

                optimizer._scaler = GradScaler()
        if optim_state_dict:
            optimizer.load_state_dict(optim_state_dict)
        return optimizer

    @property
    def training_step(self):

        return self._training_step

    @property
    def amp(self):

        return self._fp16 == "amp"

    def learning_rate(self):

        if self._learning_rate_decay_fn is None:
            return self._learning_rate
        scale = self._learning_rate_decay_fn(self._decay_step)
        return scale * self._learning_rate

    def state_dict(self):
        return {
            "training_step": self._training_step,
            "decay_step": self._decay_step,
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._training_step = state_dict["training_step"]

        if "decay_step" in state_dict:
            self._decay_step = state_dict["decay_step"]
        if "optimizer" in state_dict:
            self._optimizer.load_state_dict(state_dict["optimizer"])

    def zero_grad(self):

        self._optimizer.zero_grad()

    def backward(self, loss):

        if self.amp:
            self._scaler.scale(loss).backward()
        elif self._fp16 == "legacy":
            kwargs = {}
            if "update_master_grads" in fn_args(self._optimizer.backward):
                kwargs["update_master_grads"] = True
            self._optimizer.backward(loss, **kwargs)
        else:
            loss.backward()

    def step(self):

        learning_rate = self.learning_rate()

        if self.amp:
            self._scaler.unscale_(self._optimizer)
        elif self._fp16 == "legacy":
            if hasattr(self._optimizer, "update_master_grads"):
                self._optimizer.update_master_grads()
            if (
                hasattr(self._optimizer, "clip_master_grads")
                and self._max_grad_norm > 0
            ):
                self._optimizer.clip_master_grads(self._max_grad_norm)

        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
            if self._max_grad_norm > 0 and self._fp16 != "legacy":
                clip_grad_norm_(group["params"], self._max_grad_norm)

        if self.amp:

            self._scaler.step(self._optimizer)

            self._scaler.update()
        else:
            self._optimizer.step()
        self._decay_step += 1
        self._training_step += 1
