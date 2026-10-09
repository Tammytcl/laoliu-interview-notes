"""Classroom tensor exercise, not a training implementation or paper reproduction.

Run: python assets/learning/policy-optimization-lab.py
Requires PyTorch. CPU only; no model, dataset, network, or tool environment.
All examples use small synthetic tensors. EPS=.2 is for explaining branches,
not the clipping range used in the original GSPO experiments.
"""
import math
import torch
import torch.nn.functional as F

DTYPE = torch.float64


def tensor(values, grad=False):
    return torch.tensor(values, dtype=DTYPE, requires_grad=grad)


def reduce_loss(values, mask, mode="sequence"):
    if mode == "token":
        return (values * mask).sum() / mask.sum()
    lengths = mask.sum(-1)
    valid = lengths > 0
    return ((values * mask).sum(-1) / lengths.clamp(min=1))[valid].mean()


def group_advantage(rewards):
    # [Q,G], sample std (correction=1); finite epsilon doesn't create signal.
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=True)
    return (rewards - mean) / (std + 1e-6)


def clipped_loss(ratio, advantages, mask, epsilon=.2):
    normal = ratio * advantages
    clipped = ratio.clamp(1-epsilon, 1+epsilon) * advantages
    return reduce_loss(-torch.minimum(normal, clipped), mask)


def gspo_loss(logp, old_logp, advantages, mask, token_form=False):
    logratio = logp - old_logp
    seq_logratio = (logratio * mask).sum(-1) / mask.sum(-1)
    if token_form:
        linked = logp - logp.detach() + seq_logratio.detach().unsqueeze(-1)
    else:
        linked = seq_logratio.unsqueeze(-1).expand_as(logp)
    return clipped_loss(linked.exp(), advantages, mask)


def sapo_loss(logp, old_logp, advantages, mask, tau_pos=1., tau_neg=1.05):
    ratio = (logp-old_logp).exp()
    tau = torch.where(advantages > 0, tau_pos, tau_neg)
    surrogate = 4/tau * torch.sigmoid(tau * (ratio-1))
    return reduce_loss(-surrogate * advantages, mask)


def main():
    print("Synthetic classroom examples; no actual training or paper benchmark.")

    # PPO: d(loss)/d(logprob), averaged over four individual examples.
    logp = tensor([[math.log(.7), math.log(1.3), math.log(.7), math.log(1.3)]], True)
    adv = tensor([[2., 2., -2., -2.]])
    loss = clipped_loss(logp.exp(), adv, torch.ones_like(logp))
    loss.backward()
    expected = tensor([[-.35, 0., 0., .65]])
    assert torch.allclose(logp.grad, expected)
    print("PPO gradients (positive low/high, negative low/high):", logp.grad.tolist())

    # DPO: full-answer scores, not token-wise chosen/rejected comparisons.
    chosen, rejected = tensor([-2.], True), tensor([-3.], True)
    ref_chosen, ref_rejected = tensor([-2.5]), tensor([-2.5])
    margin = .1*((chosen-rejected)-(ref_chosen-ref_rejected))
    dpo = -F.logsigmoid(margin).mean()
    dpo.backward()
    assert abs(dpo.item()-.6443966600735709) < 1e-10
    assert chosen.grad.item() < 0 and rejected.grad.item() > 0
    print("DPO margin/loss/gradient:", margin.item(), dpo.item(), chosen.grad.item(), rejected.grad.item())

    # GRPO: one relative scalar per completion, before broadcasting to tokens.
    rewards = tensor([[0., 0., 1.], [1., 1., 1.]])
    relative = group_advantage(rewards)
    assert torch.allclose(relative[1], torch.zeros(3, dtype=DTYPE))
    print("GRPO advantages (sample std):", relative.tolist())

    # GSPO: same values and gradients for direct and detached-token forms
    # when per-sequence advantages and reductions agree.
    values = [[math.log(2.), math.log(.5)]]
    direct, linked = tensor(values, True), tensor(values, True)
    old, mask, same_adv = tensor([[0., 0.]]), tensor([[1., 1.]]), tensor([[1., 1.]])
    direct_loss = gspo_loss(direct, old, same_adv, mask)
    linked_loss = gspo_loss(linked, old, same_adv, mask, token_form=True)
    direct_loss.backward(); linked_loss.backward()
    assert torch.allclose(direct.grad, linked.grad)
    assert torch.allclose(direct.grad, tensor([[-.5, -.5]]))
    print("GSPO geometric ratio:", direct.detach().mean().exp().item(), "gradient:", direct.grad.tolist())

    # GDPO: per-reward normalization then batch whitening across completions.
    # This lesson uses equal lengths; author token-mask whitening may differ.
    vector = tensor([[[0., 0.], [1., 0.]], [[0., 0.], [1., 1.]]])  # [Q,G,K]
    normalized = (vector-vector.mean(1, keepdim=True))/(vector.std(1, keepdim=True, unbiased=True)+1e-6)
    merged = normalized.sum(-1)
    whitened = (merged-merged.mean()) / torch.sqrt(merged.var(unbiased=True)+1e-8)
    assert abs(whitened[1,1].item()) > abs(whitened[0,1].item())
    complementary = tensor([[[1., 0.], [1., 0.], [0., 1.]]])
    parts = (complementary-complementary.mean(1, keepdim=True))/(complementary.std(1, keepdim=True, unbiased=True)+1e-6)
    assert torch.allclose(parts.sum(-1), torch.zeros((1,3), dtype=DTYPE))
    print("GDPO merged/whitened:", merged.tolist(), whitened.tolist(), "complementary rewards cancel")

    # DAPO: accept whole nondegenerate groups, and compare two reductions.
    groups = tensor([[0.,0.,0.,0.], [0.,0.,1.,1.], [1.,1.,1.,1.]])
    kept = groups.std(1, unbiased=False) > 0
    assert kept.tolist() == [False, True, False]
    values = tensor([[1.,1.,0.,0.], [3.,3.,3.,3.]])
    masks = tensor([[1.,1.,0.,0.], [1.,1.,1.,1.]])
    seq_mean, token_mean = reduce_loss(values, masks), reduce_loss(values, masks, "token")
    assert abs(seq_mean.item()-2.) < 1e-10
    assert abs(token_mean.item()-7/3) < 1e-10
    print("DAPO kept groups:", kept.tolist(), "sequence/token means:", seq_mean.item(), token_mean.item())
    penalty = min(-(18384-16384)/4096, 0)
    assert penalty == -.48828125
    print("DAPO soft penalty at length 18384:", penalty)

    # SAPO: the derivative gate w is not the forward objective f.
    x = tensor([[math.log(2.)]], True)
    one = tensor([[1.]])
    result = sapo_loss(x, torch.zeros_like(x), one, one)
    result.backward()
    p = torch.sigmoid(tensor(1.))
    w = 4*p*(1-p)
    assert abs(x.grad.item() - (-2*w.item())) < 1e-10
    print("SAPO w / r*w / loss gradient:", w.item(), 2*w.item(), x.grad.item())
    wrong_x = tensor([[math.log(2.)]], True)
    wrong_ratio = wrong_x.exp()
    wrong_p = torch.sigmoid(wrong_ratio-1)
    wrong_loss = -(4*wrong_p*(1-wrong_p)*wrong_ratio).mean()
    wrong_loss.backward()
    assert not torch.allclose(wrong_x.grad, x.grad)
    print("Wrong objective w(r)*r has a different gradient:", wrong_x.grad.item())
    print("All classroom checks passed; this is not a reproduction of the seven papers.")


if __name__ == "__main__":
    main()
