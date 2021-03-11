from __future__ import division
import torch


class Beam:
    def __init__(
        self,
        size,
        pad,
        bos,
        eos,
        n_best=1,
        cuda=False,
        global_scorer=None,
        min_length=0,
        stepwise_penalty=False,
        block_ngram_repeat=0,
        exclusion_tokens=set(),
    ):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        self.prev_ks = []

        self.next_ys = [self.tt.LongTensor(size).fill_(pad)]
        self.next_ys[0][0] = bos

        self._eos = eos
        self.eos_top = False

        self.attn = []

        self.finished = []
        self.n_best = n_best

        self.global_scorer = global_scorer
        self.global_state = {}

        self.min_length = min_length

        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

    def get_current_state(self):
        return self.next_ys[-1]

    def get_current_origin(self):
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):

        num_words = word_probs.size(1)
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)

        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20

        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.scores.unsqueeze(1).expand_as(word_probs)

            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):

                        gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat :]

                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -10e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))

        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0

            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):

        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class PenaltyBuilder(object):
    def __init__(self, cov_pen, length_pen):
        self.length_pen = length_pen
        self.cov_pen = cov_pen

    def coverage_penalty(self):
        if self.cov_pen == "wu":
            return self.coverage_wu
        elif self.cov_pen == "summary":
            return self.coverage_summary
        else:
            return self.coverage_none

    def length_penalty(self):
        if self.length_pen == "wu":
            return self.length_wu
        elif self.length_pen == "avg":
            return self.length_average
        else:
            return self.length_none

    def coverage_wu(self, beam, cov, beta=0.0):

        penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        return beta * penalty

    def coverage_summary(self, beam, cov, beta=0.0):

        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(1)
        penalty -= cov.size(1)
        return beta * penalty

    def coverage_none(self, beam, cov, beta=0.0):

        return beam.scores.clone().fill_(0.0)

    def length_wu(self, beam, logprobs, alpha=0.0):

        modifier = ((5 + len(beam.next_ys)) ** alpha) / ((5 + 1) ** alpha)
        return logprobs / modifier

    def length_average(self, beam, logprobs, alpha=0.0):

        return logprobs / len(beam.next_ys)

    def length_none(self, beam, logprobs, alpha=0.0, beta=0.0):

        return logprobs


class GNMTGlobalScorer(object):
    def __init__(self, alpha=0, beta=0, cov_penalty=None, length_penalty=None):
        self.alpha = alpha
        self.beta = beta
        penalty_builder = PenaltyBuilder(cov_penalty, length_penalty)

        self.cov_penalty = penalty_builder.coverage_penalty()

        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):

        normalized_probs = self.length_penalty(beam, logprobs, self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam, beam.global_state["coverage"], self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):

        pass

    def update_global_state(self, beam):
        pass