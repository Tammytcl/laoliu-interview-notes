"""RL theory-to-code classroom lab. Python + PyTorch, CPU float64 only.

Run: python assets/learning/rl-algorithms-unified-lab.py
Optional: --output /path/to/results.json
Own teaching implementation, NOT copied framework code or a paper benchmark.
The two-step MDP enumerates its four equiprobable paths under a frozen uniform
policy; this gives an exact initial batch, not a random rollout collector.
The shared comparison fixture isolates local objectives, not whole recipes.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

DTYPE = torch.float64


def tensor(data, grad=False):
    return torch.tensor(data, dtype=DTYPE, requires_grad=grad)


def reduce_tokens(values, mask, mode="sequence"):
    """[B,L] -> scalar; exclude empty rows, fail for an empty batch."""
    lengths = mask.sum(-1)
    if not (lengths > 0).any():
        raise ValueError("No action tokens")
    numerator = (values * mask).sum(-1)
    if mode == "token":
        return numerator.sum() / lengths.sum()
    if mode != "sequence":
        raise ValueError(mode)
    return (numerator / lengths.clamp(min=1))[lengths > 0].mean()


@torch.no_grad()
def gae(rewards, old_values, next_values, bootstrap, trace, gamma=1., lam=1.):
    """All inputs [B,T] are REAL transitions, not padded token positions.

    bootstrap=1 permits actual-next-state value; trace=1 connects next record.
    A truncated segment may have bootstrap=1 and trace=0 on its last record.
    """
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        delta = (rewards[:, t] + gamma * bootstrap[:, t] * next_values[:, t]
                 - old_values[:, t])
        carry = delta + gamma * lam * trace[:, t] * carry
        advantage[:, t] = carry
    return advantage, advantage + old_values


@torch.no_grad()
def group_advantage(rewards, method="grpo"):
    """[Q,G] -> [Q,G]. Fixed-size complete groups, sample std, G >= 2."""
    if rewards.shape[1] < 2:
        raise ValueError("This classroom group estimator requires G >= 2")
    centered = rewards - rewards.mean(1, keepdim=True)
    if method == "rloo":
        return centered * rewards.shape[1] / (rewards.shape[1] - 1)
    if method == "center":
        return centered
    if method != "grpo":
        raise ValueError(method)
    return centered / (rewards.std(1, keepdim=True, correction=1) + 1e-6)


@torch.no_grad()
def gdpo_advantage(reward_vectors, weights):
    """[Q,G,K]. Teaching choice: FINAL whitening across responses, not tokens.

    NVlabs' token-mask whitening can give unequal lengths different weights.
    """
    centered = reward_vectors - reward_vectors.mean(1, keepdim=True)
    normalized = centered / (reward_vectors.std(1, keepdim=True, correction=1) + 1e-6)
    merged = (normalized * weights).sum(-1)
    return (merged - merged.mean()) / (merged.std(correction=1) + 1e-6)


def clipped_items(ratio, advantage, low=.2, high=.2):
    plain = ratio * advantage.detach()
    bounded = ratio.clamp(1 - low, 1 + high) * advantage.detach()
    return -torch.minimum(plain, bounded)


def policy_loss(logp, old_logp, advantage, mask, method="grpo", mode="sequence"):
    """Same [B,L] interface, with fixed advantage and frozen old_logp.

    GSPO here uses its token-gradient form. For the original sequence form,
    broadcast a constant advantage within each response and use sequence mean.
    DAPO here isolates upper clip and token reduction ONLY; filtering, refill,
    overlong reward and KL choices live outside this function.
    """
    advantage = advantage.detach()
    logratio = logp - old_logp.detach()
    if method == "gspo":
        seq = (logratio * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        logratio = seq.detach()[:, None] + logp - logp.detach()
    ratio = logratio.exp()
    if method == "sapo":
        tau = torch.where(advantage > 0, 1., 1.05)
        items = -(4 / tau) * torch.sigmoid(tau * (ratio - 1)) * advantage
    elif method == "cispo":
        items = -ratio.clamp(.8, 1.2).detach() * advantage * logp
    elif method in {"ppo", "grpo", "gspo", "dapo"}:
        items = clipped_items(ratio, advantage, high=.28 if method == "dapo" else .2)
    else:
        raise ValueError(method)
    return reduce_tokens(items, mask, "token" if method == "dapo" else mode)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=.1):
    """Four [B] answer logprob SUMS, all conditioned on the same prompt."""
    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected).detach())
    return -F.logsigmoid(margin).mean()


def two_step_ppo():
    # State 0: choose first bit. State 1/2: first bit was 0/1; choose second.
    # Reward=1 only for bits (1,1), then terminate. All other rewards are zero.
    states = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 2]])
    actions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    rewards = tensor([[0, 0], [0, 0], [0, 0], [0, 1]])
    actor = torch.nn.Parameter(torch.zeros(3, 2, dtype=DTYPE))
    # Deliberately imperfect initialization; true frozen-policy V is [.25,0,.5].
    critic = torch.nn.Parameter(tensor([.4, .1, .6]))
    with torch.no_grad():
        old_logp = actor[states].log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
        old_values = critic[states].clone()
        next_values = torch.stack((critic[states[:, 1]], torch.zeros(4, dtype=DTYPE)), 1)
    boundary = tensor([[1, 0]]).expand(4, -1)
    adv, returns = gae(rewards, old_values, next_values, boundary, boundary)
    saved = old_logp.clone(), adv.clone(), returns.clone()
    optimizer = torch.optim.SGD([actor, critic], lr=.1)
    records = []
    for epoch in range(2):
        logp = actor[states].log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
        actor_loss = policy_loss(logp, old_logp, adv, torch.ones_like(logp), "ppo", "token")
        value_loss = .5 * (critic[states] - returns).square().mean()
        optimizer.zero_grad()
        (actor_loss + .5 * value_loss).backward()
        if epoch == 0:
            assert actor.grad[0, 1] < 0 and actor.grad[2, 1] < 0
        records.append({"epoch": epoch, "actor_loss": actor_loss.item(),
                        "value_loss": value_loss.item(), "actor_gradient": actor.grad.tolist(),
                        "critic_gradient": critic.grad.tolist(),
                        "mean_ratio_before_step": (logp - old_logp).exp().mean().item(),
                        "max_ratio_deviation_before_step": ((logp - old_logp).exp() - 1).abs().max().item()})
        optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(saved, (old_logp, adv, returns)))
    probs = actor.detach().softmax(-1)
    expected_reward = (probs[0, 1] * probs[2, 1]).item()
    assert expected_reward > .25
    return {"old_logprob": old_logp.tolist(), "advantages": adv.tolist(),
            "value_targets": returns.tolist(), "updates": records,
            "initial_exact_expected_reward": .25,
            "final_exact_expected_reward": expected_reward,
            "fixed_targets_unchanged": True}


def compare_shared_fixture():
    # Six responses, Q=2, G=3; token slots have lengths 2,3,1,3,2,1.
    # Normalized two-action distributions; selected actions are all action 0.
    mask = tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0],
                   [1, 1, 1], [1, 1, 0], [1, 0, 0]])
    initial = tensor([[.9, -.9, .1], [.1, .4, -.3], [.8, .1, -.3],
                      [-.9, .3, .8], [.6, .1, .4], [-.1, .6, .9]])
    old_logp = torch.full_like(mask, -math.log(2))
    rewards = tensor([[0, 0, 1], [0, 1, 1]])
    advantage = group_advantage(rewards).reshape(6, 1).expand_as(mask)
    out = {}
    for method in ("grpo", "gspo", "dapo", "sapo", "cispo"):
        logits = initial.clone().requires_grad_()
        logp = F.logsigmoid(logits)  # action-0 logprob of softmax([logits,0])
        loss = policy_loss(logp, old_logp, advantage, mask, method)
        loss.backward()
        assert torch.equal(logits.grad[mask == 0], torch.zeros_like(logits.grad[mask == 0]))
        out[method] = {"loss": loss.item(), "gradient_wrt_selected_action_logit": logits.grad.tolist()}
    out["mask"] = mask.tolist()
    out["group_rewards"] = rewards.tolist()
    out["grpo_advantages"] = advantage[:, 0].reshape(2, 3).tolist()
    out["note"] = "Different objectives have different loss offsets; compare gradients, not loss ranking."
    return out


def regression_checks():
    checks = {}
    logp = tensor([[math.log(.7), math.log(1.3), math.log(.7), math.log(1.3)]], True)
    policy_loss(logp, torch.zeros_like(logp), tensor([[2, 2, -2, -2]]),
                torch.ones_like(logp), "ppo").backward()
    assert torch.allclose(logp.grad, tensor([[-.35, 0, 0, .65]]))
    checks["ppo_four_branch_loss_gradients"] = logp.grad.tolist()

    # Direct original GSPO and the token form coincide for constant A + seq mean.
    mask = tensor([[1, 1, 0], [1, 1, 1]])
    a = tensor([[-2.8, -4.2, -6], [-3.8, -4.1, -4]], True)
    b = a.detach().clone().requires_grad_()
    old = torch.full_like(a, -4.)
    adv = tensor([[1, 1, 1], [-1, -1, -1]])
    seq = (((a - old) * mask).sum(-1) / mask.sum(-1)).exp()
    direct = clipped_items(seq, adv[:, 0]).mean()
    linked = policy_loss(b, old, adv, mask, "gspo")
    direct.backward(); linked.backward()
    assert torch.allclose(direct, linked) and torch.allclose(a.grad, b.grad)
    checks["gspo_equal_value_and_gradient"] = True

    # Different token advantages: same forward loss need not mean same gradient.
    process = tensor([[1, -1]])
    direct_lp, token_lp = tensor([[-2, -2]], True), tensor([[-2, -2]], True)
    process_old = tensor([[-2, -2]])
    direct_process = clipped_items((direct_lp - process_old).mean(-1).exp(), process.mean(-1)).mean()
    linked_process = policy_loss(token_lp, process_old, process, torch.ones_like(token_lp), "gspo")
    direct_process.backward(); linked_process.backward()
    assert torch.allclose(direct_process, linked_process)
    assert torch.equal(direct_lp.grad, tensor([[0, 0]]))
    assert torch.allclose(token_lp.grad, tensor([[-.5, .5]]))
    checks["gspo_process_advantages_gradient_difference"] = {
        "direct": direct_lp.grad.tolist(), "token": token_lp.grad.tolist()}

    checks["gspo_complete_ratio_versus_geometric_ratio"] = [4., 2.]
    assert math.exp(math.log(2) * 2) == 4.

    values = tensor([[1, 1, 0, 0], [3, 3, 3, 3]])
    m = tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    checks["sequence_mean"] = reduce_tokens(values, m).item()
    checks["token_mean"] = reduce_tokens(values, m, "token").item()
    assert checks["sequence_mean"] == 2.
    assert abs(checks["token_mean"] - 7/3) < 1e-12

    # Token-sum / GLOBAL denominator equals concatenated batch gradient.
    def accumulated(split):
        w = tensor(1., True)
        for indices in split:
            (w * values[indices] * m[indices]).sum().div(m.sum()).backward()
        return w.grad.item()
    assert abs(accumulated([[0], [1]]) - accumulated([[0, 1]])) < 1e-12
    checks["global_denominator_gradient_equivalence"] = True

    r, v, nv = tensor([[0]]), tensor([[2]]), tensor([[3]])
    terminal, _ = gae(r, v, nv, tensor([[0]]), tensor([[0]]), gamma=.9)
    truncated, _ = gae(r, v, nv, tensor([[1]]), tensor([[0]]), gamma=.9)
    checks["terminal_and_truncation_advantage"] = [terminal.item(), truncated.item()]
    assert abs(terminal.item() + 2) < 1e-12 and abs(truncated.item() - .7) < 1e-12

    centered = group_advantage(tensor([[0, 1, 3]]), "center")
    loo = group_advantage(tensor([[0, 1, 3]]), "rloo")
    assert torch.allclose(loo, 1.5 * centered)
    assert torch.equal(group_advantage(tensor([[1, 1, 1]])), tensor([[0, 0, 0]]))
    checks["rloo_scale"] = 1.5
    vector = tensor([[[0, 0], [1, 0]], [[0, 0], [1, 1]]])
    checks["gdpo_equal_length_response_whitening"] = gdpo_advantage(vector, tensor([1, 1])).tolist()
    comp = tensor([[[1, 0], [1, 0], [0, 1]]])
    assert torch.allclose(gdpo_advantage(comp, tensor([1, 1])), tensor([[0, 0, 0]]), atol=1e-8)
    checks["gdpo_complementary_rewards_cancel"] = True

    chosen, rejected = tensor([-2], True), tensor([-3], True)
    dpo = dpo_loss(chosen, rejected, tensor([-2.5]), tensor([-2.5]))
    dpo.backward()
    assert chosen.grad < 0 and rejected.grad > 0
    checks["dpo_loss_and_pair_gradients"] = [dpo.item(), chosen.grad.item(), rejected.grad.item()]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"scope": "CPU float64 synthetic teaching; no LLM or benchmark reproduction",
              "two_step_ppo": two_step_ppo(), "shared_fixture": compare_shared_fixture(),
              "checks": regression_checks()}
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
