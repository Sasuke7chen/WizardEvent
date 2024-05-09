from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
import torch
import numpy as np

class LMScorer(object):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def tgt_prob_for_batch(
        self, 
        probs,
        tgt_ids,
        tokenizer,
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        outputs: List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]] = []

        lm_probs = []
        for i in range(len(probs)):
            prob = probs[i]
            tids = tgt_ids[:, i]

            word_prob = [p[t].item() for p, t in zip(prob, tids)]
            lm_probs.append(word_prob)

        lm_probs = np.array(lm_probs)
        lm_probs = lm_probs.T
        
        mask = np.ones_like(tgt_ids)
        mask[tgt_ids == tokenizer.pad_token_id] = 0
        
        lm_probs = lm_probs * mask
        mean_prob = lm_probs.sum(1) / mask.sum(1)
        return mean_prob.tolist()

    def lm_prob_for_batch(
            self,
            logits,
            pos,
            target_ids,
            ):
        logits = torch.nn.functional.log_softmax(logits, -1)

        lm_probs = torch.gather(logits, -1, target_ids.unsqueeze(-1)).squeeze(-1)

        mask = torch.zeros_like(target_ids)
        for m, p in zip(mask, pos):
            m[p[0]: p[1]] = 1

        lm_probs = lm_probs * mask
        mean_prob = lm_probs.sum(1) / mask.sum(1)
        return mean_prob.cpu().numpy().tolist(), lm_probs


class GPTScorer(object):
    def lm_prob_for_batch(self, completion):
        choices = completion.choices

        probs = []
        for choice in choices:
            logprobs = choice.logprobs.token_logprobs
            logprobs = np.array(logprobs[1:])
            probs.append(logprobs.mean())


        return probs
