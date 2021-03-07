import argparse
import json
import os
import signal
import sys
import warnings
from datetime import datetime

import glob2 as glob
import torch
import torch.distributed
from loguru import logger

from distributed import all_gather_list, all_reduce_and_rescale_tensors, multi_init
from loss import NMTLossCompute
from models import build_model
from optimizer import build_optim

from utils import (
    ModelSaver,
    ReportMgr,
    Statistics,
    _collect_report_features,
    _load_fields,
    build_dataset_iter,
    lazily_load_dataset,
    make_features,
)

warnings.filterwarnings("ignore")


def init_logger(log_file):
    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | <level>{level: <8}</level> |"
        " <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    logger.add(
        log_file,
        backtrace=True,
        diagnose=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | <level>{level: <8}</level> |"
        " <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )



def use_gpu(opt):
    return (hasattr(opt, "gpu_ranks") and len(opt.gpu_ranks) > 0) or (
        hasattr(opt, "gpu") and opt.gpu > -1
    )

class Trainer(object):
    """
    Class that controls the training process.
    """

    def __init__(
        self,
        model,
        train_loss,
        valid_loss,
        optim,
        trunc_size=0,
        shard_size=32,
        data_type="text",
        norm_method="sents",
        grad_accum_count=1,
        n_gpu=1,
        gpu_rank=1,
        gpu_verbose_level=0,
        report_manager=None,
        model_saver=None,
    ):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert (
                self.trunc_size == 0
            ), """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`
        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):
        Return:
            None
        """
        logger.info("Start training...")

        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    if self.gpu_verbose_level > 1:
                        logger.info(
                            "GpuRank %d: index: %d accum: %d"
                            % (self.gpu_rank, i, accum)
                        )

                    true_batchs.append(batch)

                    if self.norm_method == "tokens":
                        num_tokens = batch.tgt[1:].ne(self.train_loss.padding_idx).sum()
                        normalization += num_tokens.item()
                    else:
                        normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.gpu_verbose_level > 0:
                            logger.info(
                                "GpuRank %d: reduce_counter: %d \
                                        n_minibatch %d"
                                % (self.gpu_rank, reduce_counter, len(true_batchs))
                            )
                        if self.n_gpu > 1:
                            normalization = sum(all_gather_list(normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats, report_stats
                        )

                        report_stats = self._maybe_report_training(
                            step, train_steps, self.optim.learning_rate, report_stats
                        )

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if step % valid_steps == 0:
                            if self.gpu_verbose_level > 0:
                                logger.info(
                                    "GpuRank %d: validate step %d"
                                    % (self.gpu_rank, step)
                                )
                            valid_iter = valid_iter_fct()
                            valid_stats = self.validate(valid_iter)
                            if self.gpu_verbose_level > 0:
                                logger.info(
                                    "GpuRank %d: gather valid stat \
                                            step %d"
                                    % (self.gpu_rank, step)
                                )
                            valid_stats = self._maybe_gather_stats(valid_stats)
                            if self.gpu_verbose_level > 0:
                                logger.info(
                                    "GpuRank %d: report stat step %d"
                                    % (self.gpu_rank, step)
                                )
                            self._report_step(
                                self.optim.learning_rate, step, valid_stats=valid_stats
                            )

                        if self.gpu_rank == 0:
                            self._maybe_save(step)
                        step += 1
                        if step > train_steps:
                            break
            if self.gpu_verbose_level > 0:
                logger.info(
                    "GpuRank %d: we completed an epoch \
                            at step %d"
                    % (self.gpu_rank, step)
                )
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            src = make_features(batch, "src", self.data_type)
            if self.data_type == "text":
                _, src_lengths = batch.src
            elif self.data_type == "audio":
                src_lengths = batch.src_lengths
            else:
                src_lengths = None

            tgt = make_features(batch, "tgt")

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def _gradient_accumulation(
        self, true_batchs, normalization, total_stats, report_stats
    ):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = make_features(batch, "src", self.data_type)
            if self.data_type == "text":
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum().item()
            elif self.data_type == "audio":
                src_lengths = batch.src_lengths
            else:
                src_lengths = None

            tgt_outer = make_features(batch, "tgt")

            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j : j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state = self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                    batch, outputs, attns, j, trunc_size, self.shard_size, normalization
                )
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [
                            p.grad.data
                            for p in self.model.parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        all_reduce_and_rescale_tensors(grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [
                    p.grad.data
                    for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                all_reduce_and_rescale_tensors(grads, float(1))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats, multigpu=self.n_gpu > 1
            )

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats, valid_stats=valid_stats
            )

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)


def check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if "encoder" in name:
            enc += param.nelement()
        elif "decoder" or "generator" in name:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size
        if opt.model_type == "text" and opt.enc_rnn_size != opt.dec_rnn_size:
            raise AssertionError(
                """We do not support different encoder and
                decoder rnn sizes for translation now."""
            )

    opt.brnn = opt.encoder_type == "brnn"

    if opt.rnn_type == "SRU" and not opt.gpu_ranks:
        raise AssertionError("Using SRU requires -gpu_ranks set.")

    if torch.cuda.is_available() and not opt.gpu_ranks:
        logger.info(
            "WARNING: You have a CUDA device, \
             should run with -gpu_ranks"
        )

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt

def build_loss_compute(model, tgt_vocab, opt, train=True):
    device = torch.device("cuda" if use_gpu(opt) else "cpu")
    compute = NMTLossCompute(
        model.generator,
        tgt_vocab,
        label_smoothing=opt.label_smoothing if train else 0.0,
    )
    compute.to(device)

    return compute

def build_trainer(opt, device_id, model, fields, optim, data_type, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*
    """

    train_loss = build_loss_compute(model, fields["tgt"].vocab, opt)
    valid_loss = build_loss_compute(model, fields["tgt"].vocab, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = opt.world_size
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    report_manager = ReportMgr(opt.report_every, start_time=-1)
    
    trainer = Trainer(
        model,
        train_loss,
        valid_loss,
        optim,
        trunc_size,
        shard_size,
        data_type,
        norm_method,
        grad_accum_count,
        n_gpu,
        gpu_rank,
        gpu_verbose_level,
        report_manager,
        model_saver=model_saver,
    )
    return trainer

def train_on_single_device(opt, device_id):
    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info("Loading checkpoint from %s" % opt.train_from)
        checkpoint = torch.load(
            opt.train_from, map_location=lambda storage, loc: storage
        )
        model_opt = checkpoint["opt"]
    else:
        checkpoint = None
        model_opt = opt

    # Peek the first dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)

    # Report src/tgt features.

    src_features, tgt_features = _collect_report_features(fields)
    for j, feat in enumerate(src_features):
        logger.info(" * src feature %d size = %d" % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        logger.info(" * tgt feature %d size = %d" % (j, len(fields[feat].vocab)))

    # Build model.
    model = build_model(opt, fields["tgt"].vocab, checkpoint, gpu=use_gpu(opt))
    n_params, enc, dec = tally_parameters(model)
    logger.info("encoder: %d" % enc)
    logger.info("decoder: %d" % dec)
    logger.info("* number of parameters: %d" % n_params)
    check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = ModelSaver(
        opt.save_model,
        model,
        model_opt,
        fields,
        optim,
        opt.save_checkpoint_steps,
        opt.keep_checkpoint,
    )

    trainer = build_trainer(
        opt, device_id, model, fields, optim, data_type, model_saver=model_saver
    )

    def train_iter_fct():
        return build_dataset_iter(lazily_load_dataset("train", opt), fields, opt)

    def valid_iter_fct():
        return build_dataset_iter(
            lazily_load_dataset("valid", opt), fields, opt, is_train=False
        )

    # Do training.
    trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps, opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


def main(opt):
    n_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        mp = torch.multiprocessing.get_context("spawn")
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(n_gpu):
            procs.append(
                mp.Process(target=run, args=(opt, device_id, error_queue,), daemon=True)
            )
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        for p in procs:
            p.join()

    elif n_gpu == 1:  # case 1 GPU only
        train_on_single_device(opt, opt.gpuid[0])
    else:  # case only CPU
        train_on_single_device(opt, -1)


def run(opt, device_id, error_queue):
    """ run process """
    try:
        gpu_rank = multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError(
                "An error occurred in \
                  Distributed initialization"
            )
        train_on_single_device(opt, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading

        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


if __name__ == "__main__":
    from argparse import Namespace

    params = {}
    with open("params.json", "r") as f:
        params = json.loads(f.read())
    opt = Namespace(**params)
    main(opt)
