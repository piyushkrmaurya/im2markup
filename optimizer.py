import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from utils import use_gpu

def build_optim(model, opt, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if opt.train_from:
        optim = checkpoint["optim"]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            opt.optim,
            opt.learning_rate,
            opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_steps=opt.start_decay_steps,
            decay_steps=opt.decay_steps,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size,
        )

    optim.set_parameters(model.named_parameters())

    if opt.train_from:
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if use_gpu(opt):
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        if (optim.method == "adam") and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model"
                + " but optimizer state is empty"
            )

    return optim

class Optimizer:
    def __init__(
        self,
        method,
        learning_rate,
        max_grad_norm,
        lr_decay=1,
        start_decay_steps=None,
        decay_steps=None,
        beta1=0.9,
        beta2=0.999,
        adagrad_accum=0.0,
        decay_method=None,
        warmup_steps=4000,
        model_size=None,
    ):
        self.last_ppl = None
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.model_size = model_size

    def set_parameters(self, params):
        """ ? """
        self.params = []
        self.sparse_params = []
        for k, p in params:
            if p.requires_grad:
                if self.method != "sparseadam" or "embed" not in k:
                    self.params.append(p)
                else:
                    self.sparse_params.append(p)
        if self.method == "sgd":
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate)
        elif self.method == "adagrad":
            self.optimizer = optim.Adagrad(self.params, lr=self.learning_rate)
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    self.optimizer.state[p]["sum"] = self.optimizer.state[p][
                        "sum"
                    ].fill_(self.adagrad_accum)
        elif self.method == "adadelta":
            self.optimizer = optim.Adadelta(self.params, lr=self.learning_rate)
        elif self.method == "adam":
            self.optimizer = optim.Adam(
                self.params, lr=self.learning_rate, betas=self.betas, eps=1e-9
            )
        elif self.method == "sparseadam":
            self.optimizer = MultipleOptimizer(
                [
                    optim.Adam(
                        self.params, lr=self.learning_rate, betas=self.betas, eps=1e-8
                    ),
                    optim.SparseAdam(
                        self.sparse_params,
                        lr=self.learning_rate,
                        betas=self.betas,
                        eps=1e-8,
                    ),
                ]
            )
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        if self.method != "sparseadam":
            self.optimizer.param_groups[0]["lr"] = self.learning_rate
        else:
            for op in self.optimizer.optimizers:
                op.param_groups[0]["lr"] = self.learning_rate

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        # Decay method used in tensor2tensor.
        if self.decay_method == "noam":
            self._set_rate(
                self.original_lr
                * (
                    self.model_size ** (-0.5)
                    * min(
                        self._step ** (-0.5), self._step * self.warmup_steps ** (-1.5)
                    )
                )
            )
        # Decay based on start_decay_steps every decay_steps
        else:
            if (self.start_decay_steps is not None) and (
                self._step >= self.start_decay_steps
            ):
                self.start_decay = True
            if self.start_decay:
                if (self._step - self.start_decay_steps) % self.decay_steps == 0:
                    self.learning_rate = self.learning_rate * self.lr_decay

        if self.method != "sparseadam":
            self.optimizer.param_groups[0]["lr"] = self.learning_rate

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
