import json
import math
import os
import random
import re
import sys
import time
import traceback
import warnings
from argparse import Namespace
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.init import xavier_uniform_

from dataset import DatasetIterator, IterOnDevice
from loss import NMTLossCompute
from models import (
    Cast,
    Embeddings,
    ImageEncoder,
    InputFeedRNNDecoder,
    NMTModel,
    build_model,
)
from optimizer import Optimizer
from utils import ModelSaver, ReportManager, Statistics, init_logger

torch.multiprocessing.set_sharing_strategy("file_system")

warnings.filterwarnings("ignore")

prefix = str(hex(int(time.time())))[2:] + "_"

def build_trainer(opt, gpu_id, model, fields, optim, model_saver=None):

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = NMTLossCompute.from_opt(model, tgt_field, opt, device=opt.device)
    valid_loss = NMTLossCompute.from_opt(
        model, tgt_field, opt, train=False, device=opt.device
    )

    trunc_size = opt.truncated_decoder
    shard_size = opt.max_generator_batches if opt.model_dtype == "fp32" else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if gpu_id >= 0:
        n_gpu = 1
    else:
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = (
        EarlyStopping(opt.early_stopping, scorers=scorers_from_opts(opt))
        if opt.early_stopping > 0
        else None
    )

    source_noise = None
    if len(opt.src_noise) > 0:
        src_field = dict(fields)["src"].base_field
        corpus_id_field = dict(fields).get("corpus_id", None)
        if corpus_id_field is not None:
            ids_to_noise = corpus_id_field.numericalize(opt.data_to_noise)
        else:
            ids_to_noise = None
        source_noise = source_noise.MultiNoise(
            opt.src_noise,
            opt.src_noise_prob,
            ids_to_noise=ids_to_noise,
            pad_idx=src_field.pad_token,
            end_of_sentence_mask=src_field.end_of_sentence_mask,
            word_start_mask=src_field.word_start_mask,
            gpu_id=gpu_id,
        )

    report_manager = ReportManager(opt.report_every, start_time=-1)
    trainer = Trainer(
        model,
        train_loss,
        valid_loss,
        optim,
        trunc_size,
        shard_size,
        norm_method,
        accum_count,
        accum_steps,
        n_gpu,
        gpu_id,
        gpu_verbose_level,
        report_manager,
        model_saver=model_saver,
        average_decay=average_decay,
        average_every=average_every,
        model_dtype=opt.model_dtype,
        earlystopper=earlystopper,
        dropout=dropout,
        dropout_steps=dropout_steps,
        source_noise=source_noise,
    )
    return trainer


class Trainer(object):
    def __init__(
        self,
        model,
        train_loss,
        valid_loss,
        optim,
        trunc_size=0,
        shard_size=32,
        norm_method="sents",
        accum_count=[1],
        accum_steps=[0],
        n_gpu=1,
        gpu_id=0,
        gpu_verbose_level=0,
        report_manager=None,
        model_saver=None,
        average_decay=0,
        average_every=1,
        model_dtype="fp32",
        earlystopper=None,
        dropout=[0.3],
        dropout_steps=[0],
        source_noise=None,
    ):

        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_id = gpu_id
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.source_noise = source_noise

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert (
                    self.trunc_size == 0
                ), "To enable accumulated gradients,you must disable target sequence truncating."

        self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info(
                    "Updated dropout to %f from step %d" % (self.dropout[i], step)
                )

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [
                params.detach().float() for params in self.model.parameters()
            ]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay, 1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(
                enumerate(self.moving_average), self.model.parameters()
            ):
                self.moving_average[i] = (
                    1 - average_decay
                ) * avg + cpt.detach().float() * average_decay

    def train(
        self,
        train_iter,
        train_steps,
        save_checkpoint_steps=5000,
        valid_iter=None,
        valid_steps=10000,
    ):

        if valid_iter is None:
            logger.info("Start training loop without validation...")
        else:
            logger.info(
                "Start training loop and validate every %d steps..." % valid_steps
            )

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        for i, (batches, normalization) in enumerate(self._accum_batches(train_iter)):
            step = self.optim.training_step

            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GPU Id %d: index: %d" % self.gpu_id, i)
            if self.gpu_verbose_level > 0:
                logger.info(
                    "GPU ID %d: reduce_counter: %d n_minibatch %d"
                    % (self.gpu_id, i + 1, len(batches))
                )

            self._gradient_accumulation(
                batches, normalization, total_stats, report_stats
            )

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps, self.optim.learning_rate(), report_stats
            )

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info("GPU Id %d: validate step %d" % (self.gpu_id, step))
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average
                )
                if self.gpu_verbose_level > 0:
                    logger.info(
                        "GPU Id %d: gather valid stat step %d" % (self.gpu_id, step)
                    )
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info("GPU Id %d: report stat step %d" % (self.gpu_id, step))
                self._report_step(
                    self.optim.learning_rate(), step, valid_stats=valid_stats
                )

                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)

                    if self.earlystopper.has_stopped():
                        break

            if self.model_saver is not None and (
                save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0
            ):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None):

        valid_model = self.model
        if moving_average:

            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = (
                    avg.data.half() if self.optim._fp16 == "legacy" else avg.data
                )

        valid_model.eval()

        with torch.no_grad():
            stats = Statistics()

            for batch in valid_iter:
                src, src_lengths = (
                    batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                )
                tgt = batch.tgt

                with torch.cuda.amp.autocast(enabled=self.optim.amp):

                    outputs, attns = valid_model(src, tgt, src_lengths)

                    _, batch_stats = self.valid_loss(batch, outputs, attns)

                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        valid_model.train()

        return stats

    def _gradient_accumulation(
        self, true_batches, normalization, total_stats, report_stats
    ):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)

            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            batch = self.maybe_noise_source(batch)

            src, src_lengths = (
                batch.src if isinstance(batch.src, tuple) else (batch.src, None)
            )
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size - 1, trunc_size):

                tgt = tgt_outer[j : j + trunc_size]

                if self.accum_count == 1:
                    self.optim.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    outputs, attns = self.model(src, tgt, src_lengths, bptt=bptt)
                    bptt = True

                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size,
                    )

                try:
                    if loss is not None:
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info(
                        "At step %d, we removed a batch - accum %d"
                        % (self.optim.training_step, k)
                    )

                if self.accum_count == 1:
                    self.optim.step()

                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        if self.accum_count > 1:
            self.optim.step()

    def _start_report_manager(self, start_time=None):

        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):

        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):

        if self.report_manager is not None:
            return self.report_manager.report_training(
                step,
                num_steps,
                learning_rate,
                None
                if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                report_stats,
                multigpu=self.n_gpu > 1,
            )

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):

        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate,
                None
                if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                step,
                train_stats=train_stats,
                valid_stats=valid_stats,
            )

    def maybe_noise_source(self, batch):
        if self.source_noise is not None:
            return self.source_noise(batch)
        return batch


def check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if "encoder" in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_device(opt):

    if opt.seed > 0:
        os.environ["PYTHONHASHSEED"] = str(opt.seed)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True


    if opt.device == "cuda":
        if opt.gpu_id >= 0:
            torch.cuda.set_device(opt.gpu_id)
            opt.device = torch.device(opt.device, opt.gpu_id)
        else:
            opt.device = torch.device(opt.device)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)
    else:
        opt.device = torch.device("cpu")


def main(opt):
    configure_device(opt)

    init_logger(prefix + opt.log_file)

    assert len(opt.accum_count) == len(
        opt.accum_steps
    ), "Number of accum_count values must match number of accum_steps"

    if opt.train_from:
        logger.info("Loading checkpoint from %s" % opt.train_from)
        checkpoint = torch.load(
            opt.train_from, map_location=lambda storage, loc: storage
        )
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info("Loading vocab from checkpoint at %s." % opt.train_from)
        vocab = checkpoint["vocab"]
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + ".vocab.pt")

    fields = vocab

    dvocab = torch.load(opt.data + ".vocab.pt")
    maybe_cid_field = dvocab.get("corpus_id", None)
    if maybe_cid_field is not None:
        fields.update({"corpus_id": maybe_cid_field})

    for side in ["src", "tgt"]:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(" * %s vocab size = %d" % (sn, len(sf.vocab)))

    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = tally_parameters(model)
    logger.info("encoder: %d" % enc)
    logger.info("decoder: %d" % dec)
    logger.info("* number of parameters: %d" % n_params)
    check_save_model_path(opt)

    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    model_saver = ModelSaver(
        prefix + opt.save_model, model, model_opt, fields, optim, opt.keep_checkpoint
    )

    trainer = build_trainer(
        opt, opt.gpu_id, model, fields, optim, model_saver=model_saver
    )

    train_iter = DatasetIterator("train", fields, opt)
    train_iter = IterOnDevice(train_iter, opt.gpu_id)

    valid_iter = DatasetIterator("valid", fields, opt, is_train=False)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, opt.gpu_id)

    if opt.gpu_id >= 0:
        logger.info("Starting training on GPU: %s" % opt.gpu_id)
    else:
        logger.info("Starting training on CPU, could be very slow")

    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    try:
        trainer.train(
            train_iter,
            train_steps,
            save_checkpoint_steps=opt.save_checkpoint_steps,
            valid_iter=valid_iter,
            valid_steps=opt.valid_steps,
        )
    except KeyboardInterrupt:
        print("Execution interrupted by the user.")
    except Exception:
        traceback.print_exc()
        os.remove(prefix + opt.log_file)


if __name__ == "__main__":
    opt = {}

    with open("opts_training.json", "r") as f:
        opt = json.loads(f.read())

    opt = Namespace(**opt)

    main(opt)
