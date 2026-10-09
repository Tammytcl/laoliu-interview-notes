"""MTP classroom checks, CPU only; not paper training or GPU benchmarking.

python assets/infra/mtp-lab.py --output assets/infra/mtp-lab-results.json
Requires PyTorch for small tensor/gradient checks; no model or data downloads.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def future_labels(tokens, documents, offset):
    """Align x[t+offset] to h[t], masking out tails and document crossings."""
    labels = torch.full_like(tokens, -100)
    if offset < tokens.shape[1]:
        valid = documents[:, :-offset] == documents[:, offset:]
        labels[:, :-offset] = torch.where(valid, tokens[:, offset:], -100)
    return labels


class ToyMTP(nn.Module):
    """Causal prefix-mean trunk and linear toy heads, not a real Transformer."""
    def __init__(self, vocab=11, hidden=8, heads=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(heads)])
        self.unembed = nn.Linear(hidden, vocab, bias=False)

    def shared(self, tokens):
        emb = self.embed(tokens)
        counts = torch.arange(1, tokens.shape[1]+1, dtype=emb.dtype)[None, :, None]
        return self.proj(emb.cumsum(1) / counts)

    def loss(self, hidden, tokens, docs, index):
        logits = self.unembed(torch.tanh(self.heads[index](hidden)))
        labels = future_labels(tokens, docs, index+1)
        # Per-head valid-token mean is a teaching convention. Both schedules
        # must use the SAME reduction for the gradient comparison to be valid.
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)


def gradient_schedule_check():
    torch.manual_seed(7)
    direct = ToyMTP().double()
    staged = copy.deepcopy(direct)
    tokens = torch.tensor([[1,2,3,4,5,6]])
    docs = torch.zeros_like(tokens)  # one document; no packed-attention claim
    hidden = direct.shared(tokens)
    total = sum(direct.loss(hidden, tokens, docs, j) for j in range(3))
    total.backward()
    z = staged.shared(tokens)
    leaf = z.detach().requires_grad_(True)
    for j in range(3):
        loss = staged.loss(leaf, tokens, docs, j)
        loss.backward()  # sums leaf.grad and the shared unembedding gradient
    z.backward(leaf.grad)  # reconnect trunk exactly once
    difference = max((a.grad-b.grad).abs().max().item()
                     for a,b in zip(direct.parameters(), staged.parameters()))
    assert difference < 1e-10
    changed = tokens.clone(); changed[:, -1] = 9
    assert torch.allclose(direct.shared(tokens)[:, :3], direct.shared(changed)[:, :3])
    packed_docs = torch.tensor([[0,0,0,1,1,1]])
    masks = {str(j): future_labels(tokens, packed_docs, j).tolist() for j in [1,2,3]}
    assert masks['1'] == [[2,3,-100,5,6,-100]]
    assert masks['2'] == [[3,-100,-100,6,-100,-100]]
    assert masks['3'] == [[-100]*6]
    return {'max_gradient_difference': difference,
            'packed_label_examples': masks,
            'scope': 'Toy gradients and label masks; real packed attention must also reset document boundaries.'}


def draw(probs, rng):
    u=rng.random(); mass=0
    for i,p in enumerate(probs):
        mass+=p
        if u<mass:return i
    return len(probs)-1


def corrected_sampler_check():
    p=[.2,.5,.3]; q=[.5,.3,.2]
    accepted=[min(a,b) for a,b in zip(p,q)]
    acceptance=sum(accepted)
    positive=[max(a-b,0) for a,b in zip(p,q)]
    residual=[x/sum(positive) for x in positive]
    exact=[a+(1-acceptance)*r for a,r in zip(accepted,residual)]
    naive=[a+(1-acceptance)*b for a,b in zip(accepted,p)]
    assert max(abs(a-b) for a,b in zip(exact,p))<1e-12
    assert max(abs(a-b) for a,b in zip(naive,p))>.03
    rng=random.Random(23); counts=[0,0,0]; samples=100000
    for _ in range(samples):
        token=draw(q,rng)
        if rng.random()>min(1,p[token]/q[token]):token=draw(residual,rng)
        counts[token]+=1
    empirical=[n/samples for n in counts]
    assert max(abs(a-b) for a,b in zip(empirical,p))<.007
    # Greedy drafting has q=delta at the proposed token, not head softmax.
    deterministic_q=[1.,0.,0.]
    mass=[min(a,b) for a,b in zip(p,deterministic_q)]
    left=[max(a-b,0) for a,b in zip(p,deterministic_q)]
    corrected=[a+(1-sum(mass))*x/sum(left) for a,x in zip(mass,left)]
    assert max(abs(a-b) for a,b in zip(corrected,p))<1e-12
    return {'target':p,'proposal':q,'acceptance_probability':acceptance,
            'residual_distribution':residual,'corrected_exact':exact,
            'deterministic_proposal_check':{'proposal':deterministic_q,'corrected_exact':corrected},
            'naive_rejection_then_target_distribution':naive,
            'empirical_frequencies':empirical,'seed':23,'samples':samples,
            'scope':'Single fixed-prefix exact chain sampler; not tree/threshold kernel validation.'}


def serving_checks():
    # Conditional per-depth rates, not aggregate accepted/proposed ratio.
    rates=[.9,.9,.9]; survival=1.; expected=1.
    for a in rates:survival*=a;expected+=survival
    examples={'low_batch_cost_1.6':expected/1.6,'high_batch_cost_4.2':expected/4.2}
    assert abs(expected-3.439)<1e-12
    assert examples['high_batch_cost_4.2']<1
    b,t,v,n,d=1,4096,128000,4,2
    naive=b*t*v*n*d; sequential=b*t*v*d
    # Once draft X is rejected, its KV and all descendants are discarded.
    committed=['prefix','A']; discarded=['X','C']; pending='B'
    assert pending not in committed  # emitted correction may need next forward
    emitted_per_cycle=[1,2,4]
    aggregate_acceptance=(sum(emitted_per_cycle)-len(emitted_per_cycle))/(3*len(emitted_per_cycle))
    mean_emit=sum(emitted_per_cycle)/len(emitted_per_cycle)
    assert abs(mean_emit-(1+3*aggregate_acceptance))<1e-12
    return {'conditional_acceptance':rates,'expected_emitted_tokens':expected,
            'hypothetical_speedups':examples,'logits_only_bytes':{'naive':naive,'one_head':sequential},
            'logits_only_GiB':{'naive':naive/2**30,'one_head':sequential/2**30},
            'greedy_KV_example':{'committed_KV':committed,'discarded_draft_KV':discarded,'pending_emitted_token':pending},
            'counter_example':{'mean_emitted':mean_emit,'accepted_over_proposed':aggregate_acceptance},
            'scope':'Assumed costs and tensor storage, not measured GPU speed/memory.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    results={'updated':'2026-10-09','validation_scope':'CPU classroom checks only',
             'training':gradient_schedule_check(),'sampling':corrected_sampler_check(),
             'serving':serving_checks()}
    text=json.dumps(results,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(text+'\n')
    print(text)


if __name__=='__main__':main()
