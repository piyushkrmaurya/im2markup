import math
import sys
import time

import torch
from loguru import logger


def init_logger(log_file=None):
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

    return logger



class ModelSaver:
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, fields, optim, keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def save(self, step, moving_average=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        save_model = self.model
        if moving_average:
            model_params_data = []
            for avg, param in zip(moving_average, save_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data

        chkpt, chkpt_name = self._save(step, save_model)
        self.last_saved_step = step

        if moving_average:
            for param_data, param in zip(model_params_data, save_model.parameters()):
                param.data = param_data

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)


    def _save(self, step, model):
        model_state_dict = model.state_dict()
        model_state_dict = {
            k: v for k, v in model_state_dict.items() if "generator" not in k
        }
        generator_state_dict = model.generator.state_dict()

        # NOTE: We need to trim the vocab to remove any unk tokens that
        # were not originally here.

        vocab = deepcopy(self.fields)
        for side in ["src", "tgt"]:
            keys_to_pop = []
            if hasattr(vocab[side], "fields"):
                unk_token = vocab[side].fields[0][1].vocab.itos[0]
                for key, value in vocab[side].fields[0][1].vocab.stoi.items():
                    if value == 0 and key != unk_token:
                        keys_to_pop.append(key)
                for key in keys_to_pop:
                    vocab[side].fields[0][1].vocab.stoi.pop(key, None)

        checkpoint = {
            "model": model_state_dict,
            "generator": generator_state_dict,
            "vocab": vocab,
            "opt": self.model_opt,
            "optim": self.optim.state_dict(),
        }

        logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = "%s_step_%d.pt" % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        if os.path.exists(name):
            os.remove(name)


class ReportManager:
    """
    Args:
        report_every(int): Report status every this many sentences
        start_time(float): manually set report start time. Negative values
            means that you will need to set it later or use `start()`
    """
       
    def __init__(self, report_every, start_time=-1.0):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
        """
        self.report_every = report_every
        self.start_time = start_time


    def _report_training(self, step, num_steps, learning_rate, patience, report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps, learning_rate, self.start_time)

        report_stats = Statistics()

        return report_stats

    def _report_step(self, lr, patience, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log("Train perplexity: %g" % train_stats.ppl())
            self.log("Train accuracy: %g" % train_stats.accuracy())

        if valid_stats is not None:
            self.log("Validation perplexity: %g" % valid_stats.ppl())
            self.log("Validation accuracy: %g" % valid_stats.accuracy())

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(
        self, step, num_steps, learning_rate, patience, report_stats, multigpu=False
    ):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError(
                """ReportManager needs to be started
                                (set 'start_time' or use 'start()'"""
            )

        if step % self.report_every == 0:
            if multigpu:
                report_stats = Statistics.all_gather_stats(report_stats)
            self._report_training(
                step, num_steps, learning_rate, patience, report_stats
            )
            return Statistics()
        else:
            return report_stats

    def report_step(self, lr, patience, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            lr(float): current learning rate
            patience(int): current patience
            step(int): current step
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
        """
        self._report_step(
            lr, patience, step, train_stats=train_stats, valid_stats=valid_stats
        )


class Statistics:
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            (
                "Step %s; acc: %6.2f; ppl: %5.2f; xent: %4.2f; "
                + "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec"
            )
            % (
                step_fmt,
                self.accuracy(),
                self.ppl(),
                self.xent(),
                learning_rate,
                self.n_src_words / (t + 1e-5),
                self.n_words / (t + 1e-5),
                time.time() - start,
            )
        )
        sys.stdout.flush()

