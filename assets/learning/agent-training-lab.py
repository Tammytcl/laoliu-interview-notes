"""Actor-critic/PPO/agent OPD classroom checks, small CPU tensors only.

python assets/learning/agent-training-lab.py --output assets/learning/agent-training-lab-results.json
No model, dataset, network, sandbox service or GPU benchmark is used.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import torch

torch.set_num_threads(1)
DTYPE = torch.float64


def actor_baselines():
    p = 0.25
    probabilities, rewards, score = [1-p, p], [0., 2.], [-p, 1-p]
    results = {}
    for baseline in [0., 0.5, 1.5]:
        gradients = [(r-baseline)*s for r, s in zip(rewards, score)]
        mean = sum(q*g for q, g in zip(probabilities, gradients))
        variance = sum(q*(g-mean)**2 for q, g in zip(probabilities, gradients))
        assert abs(mean-0.375) < 1e-12
        results[str(baseline)] = {"expected_gradient": mean, "variance": variance}
    theta = torch.tensor(math.log(p/(1-p)), dtype=DTYPE, requires_grad=True)
    objective = 2*theta.sigmoid()
    objective.backward()
    assert abs(theta.grad.item()-0.375) < 1e-12
    # Finite i.i.d group: self-including baseline has (G-1)/G scaling.
    group_mean_gradient, loo_gradient = 0., 0.
    for actions in itertools.product([0, 1], repeat=2):
        probability = math.prod(probabilities[a] for a in actions)
        returns = [rewards[a] for a in actions]
        centered = [r-sum(returns)/2 for r in returns]
        loo = [returns[0]-returns[1], returns[1]-returns[0]]
        group_mean_gradient += probability*sum(a*score[t] for a, t in zip(centered, actions))/2
        loo_gradient += probability*sum(a*score[t] for a, t in zip(loo, actions))/2
    assert abs(group_mean_gradient-0.1875) < 1e-12
    assert abs(loo_gradient-0.375) < 1e-12
    return {"direct_autograd": theta.grad.item(), "baselines": results,
            "G2_group_mean_expected_gradient": group_mean_gradient,
            "G2_LOO_expected_gradient": loo_gradient}


def gae(rewards, values, next_values, bootstrap, trace, gamma=.9, lam=.8):
    advantages = [0.]*len(rewards)
    carry = 0.
    for t in reversed(range(len(rewards))):
        delta = rewards[t]+gamma*bootstrap[t]*next_values[t]-values[t]
        carry = delta+gamma*lam*trace[t]*carry
        advantages[t] = carry
    return advantages, [a+v for a, v in zip(advantages, values)]


def value_boundaries():
    args = ([0., 1.], [.4, .6], [.6, 0.], [1, 0], [1, 0])
    advantage, targets = gae(*args)
    mc_advantage, _ = gae(*args, lam=1.)
    assert max(abs(a-b) for a, b in zip(advantage, [.428, .4])) < 1e-12
    assert max(abs(a-b) for a, b in zip(mc_advantage, [.5, .4])) < 1e-12
    terminal, _ = gae([0.], [2.], [3.], [0], [0])
    truncated, _ = gae([0.], [2.], [3.], [1], [0])
    assert terminal == [-2.] and abs(truncated[0]-.7) < 1e-12
    correct, _ = gae([.5, 0.], [.2, .9], [0., 0.], [0, 0], [0, 0])
    wrong, _ = gae([.5, 0.], [.2, .9], [0., 0.], [0, 0], [1, 0])
    assert correct[0] > 0 and wrong[0] < 0
    # Observation positions must not advance decision-index discount/trace.
    mask = [1, 0, 0, 1]
    raw_rewards, raw_values = [0., 999., -777., 1.], [.4, 100., -100., .6]
    filtered = ([r for r, m in zip(raw_rewards, mask) if m],
                [v for v, m in zip(raw_values, mask) if m])
    assert gae(*filtered, [.6, 0.], [1, 0], [1, 0])[0] == advantage
    return {"gamma": .9, "lambda": .8, "two_step_advantages": advantage,
            "critic_targets": targets, "lambda1_advantages": mc_advantage,
            "terminal": terminal, "continuable_truncation": truncated,
            "separate_episodes": correct, "wrong_trace_connection": wrong,
            "observation_skip_matches_decision_sequence": True}


def update_objectives():
    ratios = torch.tensor([.7, 1.3, .7, 1.3], dtype=DTYPE)
    advantage = torch.tensor([2., 2., -2., -2.], dtype=DTYPE)
    logratio = ratios.log().requires_grad_()
    ratio = logratio.exp()
    loss = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean()
    loss.backward()
    ppo_gradient = logratio.grad.tolist()
    assert torch.allclose(logratio.grad, torch.tensor([-.35, 0., 0., .65], dtype=DTYPE))
    current = ratios.log().requires_grad_()
    weight = current.exp().clamp(.8, 1.2).detach()
    cispo = -(weight*advantage*current).mean()
    cispo.backward()
    cispo_gradient = current.grad.tolist()
    assert all(abs(x)>0 for x in cispo_gradient)
    origin = torch.tensor(0., dtype=DTYPE, requires_grad=True)
    (-origin.exp()*2).backward()
    assert origin.grad.item() == -2.
    return {"ratios": ratios.tolist(), "advantages": advantage.tolist(),
            "PPO_logprob_gradients_mean4": ppo_gradient,
            "CISPO_logprob_gradients_mean4": cispo_gradient,
            "ratio1_logprob_gradient": origin.grad.item(),
            "scope": "Local objectives, not a complete training implementation."}


def group_and_reduction():
    rewards = torch.tensor([0., 1., 3.], dtype=DTYPE)
    centered = rewards-rewards.mean()
    loo = rewards-(rewards.sum()-rewards)/(len(rewards)-1)
    assert torch.allclose(loo, centered*1.5)
    zeros = torch.ones(3, dtype=DTYPE)
    zero_adv = (zeros-zeros.mean())/(zeros.std()+1e-6)
    assert torch.isfinite(zero_adv).all() and zero_adv.abs().sum() == 0
    scores = torch.tensor([0., 0., 1., 1.], dtype=DTYPE)
    population = (scores-scores.mean())/scores.std(correction=0)
    sample = (scores-scores.mean())/scores.std(correction=1)
    sequence_mean, token_mean = (1+3)/2, (1*2+3*4)/6
    assert token_mean == 7/3 and sequence_mean == 2
    # Two complete episode returns, represented by one vs. three turn rows.
    assert (0+1)/2 == .5 and (0+1+1+1)/4 == .75
    # A pre-annotated two-state grouping; second episode has rewards [1,-1].
    episode_adv = [.5, -.5]
    step_adv = {"S0": [.5, -.5], "S1": [1., -1.]}
    joint = {s: [a+b for a, b in zip(episode_adv, values)] for s, values in step_adv.items()}
    return {"centered": centered.tolist(), "LOO": loo.tolist(),
            "population_std_advantages": population.tolist(),
            "sample_std_advantages": sample.tolist(), "all_equal_advantages": zero_adv.tolist(),
            "sequence_mean": sequence_mean, "token_mean": token_mean,
            "episode_mean": .5, "repeated_turn_row_mean": .75,
            "GiGPO_toy_joint_advantages": joint,
            "scope": "Fixed groups/returns; no actual environments or similarity clustering."}


def distillation_gradients():
    initial = torch.tensor([.2, .5, .3], dtype=DTYPE).log()
    teacher = torch.tensor([.5, .3, .2], dtype=DTYPE)
    def evaluate(kind):
        logits = initial.clone().requires_grad_()
        p = logits.softmax(-1); logp = logits.log_softmax(-1)
        delta = logp-teacher.log()
        if kind == "exact": loss = (p*delta).sum()
        elif kind == "score": loss = (p.detach()*delta.detach()*logp).sum()
        elif kind == "naive": loss = (p.detach()*delta).sum()
        elif kind == "forward": loss = (teacher*(teacher.log()-logp)).sum()
        elif kind == "SFT": loss = -(teacher*logp).sum()
        loss.backward()
        return loss.item(), logits.grad
    rkl, full_grad = evaluate("exact")
    _, score_grad = evaluate("score")
    _, naive_grad = evaluate("naive")
    fkl, forward_grad = evaluate("forward")
    _, sft_grad = evaluate("SFT")
    assert torch.allclose(full_grad, score_grad, atol=1e-12)
    assert naive_grad.abs().max() < 1e-12 and full_grad.norm() > .1
    assert torch.allclose(forward_grad, sft_grad, atol=1e-12)
    assert torch.allclose(forward_grad, initial.softmax(-1)-teacher, atol=1e-12)
    return {"student": initial.softmax(-1).tolist(), "teacher": teacher.tolist(),
            "reverse_KL": rkl, "forward_KL": fkl,
            "exact_rKL_gradient": full_grad.tolist(),
            "score_surrogate_gradient": score_grad.tolist(),
            "naive_sample_scalar_expected_gradient": naive_grad.tolist(),
            "forward_KL_and_SFT_gradient": forward_grad.tolist(),
            "scope": "Fixed-prefix conditional derivatives; no state-occupancy derivative."}


def intervention_and_timing():
    student, teacher = [.2, .5, .3], [.5, .3, .2]
    trigger = [-math.log(q)>1 for q in teacher]
    rate = sum(p*i for p, i in zip(student, trigger))
    gated = [p*(1-i)+rate*q for p, i, q in zip(student, trigger, teacher)]
    independent_mix = [(1-rate)*p+rate*q for p, q in zip(student, teacher)]
    assert max(abs(a-b) for a, b in zip(gated, independent_mix)) > .1
    assert abs(sum(gated)-1) < 1e-12
    T, fast, full = 10, 1., 8.
    speedup = T*full/((T-1)*fast+full)
    serial_speedup = full/(fast+full)
    assert speedup <= min(T, full/fast) and serial_speedup < 1
    shared, extra = 362.83, .01+.53
    overhead_percent = extra/shared*100
    assert .14 < overhead_percent < .16
    # Rewriting context creates a spurious ratio even with unchanged weights.
    weights = torch.tensor([[1., 0.], [0., 1.], [-1., 1.]], dtype=DTYPE)
    actual = (weights@torch.tensor([1., 0.], dtype=DTYPE)).log_softmax(-1)[0]
    rewritten = (weights@torch.tensor([0., 1.], dtype=DTYPE)).log_softmax(-1)[0]
    false_ratio = (rewritten-actual).exp().item()
    assert abs(false_ratio-1) > .1
    return {"actual_gate_rate": rate, "gate_execution_distribution": gated,
            "independent_mixture_same_rate": independent_mix,
            "Guided_cosine_beta_at_steps": {str(n): .5*(1+math.cos(math.pi*min(n/200, 1))) for n in [0,100,200,250]},
            "ActFirst_assumed_T": T, "ActFirst_ideal_speedup": speedup,
            "ActFirst_no_overlap_speedup": serial_speedup,
            "GiGPO_seconds_based_overhead_percent": overhead_percent,
            "same_model_rewritten_context_false_ratio": false_ratio,
            "scope": "Algebra and assumed latencies; not measured GPU throughput."}


def completion_event_gradient():
    """Fixed completion event: stable loss and full-softmax logit gradient."""
    initial = torch.tensor([.1, .2, .4, .3], dtype=DTYPE).log()
    logits = initial.clone().requires_grad_()
    logp = logits.log_softmax(-1)
    ids = torch.tensor([1, 2])
    mass = .7
    loss = -mass * torch.logsumexp(logp.index_select(0, ids), dim=0)
    loss.backward()
    p = initial.softmax(-1)
    probability = p.index_select(0, ids).sum()
    expected = mass * p
    expected[ids] -= mass * p[ids] / probability
    assert torch.allclose(logits.grad, expected, atol=1e-12)
    assert abs(logits.grad.sum().item()) < 1e-12
    assert abs(loss.item() + mass * math.log(.6)) < 1e-12
    return {"probabilities": p.tolist(), "completion_ids": ids.tolist(),
            "teacher_mass_weight": mass, "completion_probability": probability.item(),
            "loss": loss.item(), "logit_gradient": logits.grad.tolist(),
            "scope": "A fixed toy event; not real-tokenizer or distributed ESCD verification."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"updated": "2026-10-12", "validation_scope": "Small CPU classroom checks only",
              "actor_critic": actor_baselines(), "GAE": value_boundaries(),
              "PPO_CISPO": update_objectives(), "groups": group_and_reduction(),
              "OPD": distillation_gradients(), "agentic": intervention_and_timing(),
              "ESCD_event": completion_event_gradient()}
    output = json.dumps(result, ensure_ascii=False, indent=2)+"\n"
    if args.output: args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
