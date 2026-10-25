#!/usr/bin/env python3
"""OPD supervision on a fixed prefix: executable PyTorch CPU teaching lab.

p = student, q = frozen teacher, b = recorded behavior. All examples use a
shared vocabulary, finite logits, temperature 1 and fixed action-independent
weights. This is not an LLM rollout, framework reproduction or GPU benchmark.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def valid_logprobs(student_logits, teacher_logits, action_mask):
    # [B,L,V], [B,L,V], bool[B,L] -> two [N,V] tensors, N valid actions.
    if student_logits.ndim != 3 or teacher_logits.shape != student_logits.shape:
        raise ValueError("expected matching [B,L,V] logits")
    if action_mask.dtype != torch.bool or action_mask.shape != student_logits.shape[:2]:
        raise ValueError("expected bool[B,L] action_mask")
    if not action_mask.any().item():
        raise ValueError("batch has no valid model action")
    # Select before log_softmax: NaN * 0 is still NaN.
    s = student_logits[action_mask]
    t = teacher_logits.detach()[action_mask]
    if not torch.isfinite(s).all().item() or not torch.isfinite(t).all().item():
        raise ValueError("valid rows require finite, full-support logits")
    dtype = torch.float64 if s.dtype == torch.float64 else torch.float32
    return s.to(dtype).log_softmax(-1), t.to(dtype).log_softmax(-1)


def kl_per_row(logp, logq, direction="reverse"):
    logq = logq.detach()
    if direction == "forward":
        return (logq.exp() * (logq - logp)).sum(-1)
    if direction == "reverse":
        return (logp.exp() * (logp - logq)).sum(-1)
    raise ValueError("direction must be forward or reverse")


def full_vocab_opd(student_logits, teacher_logits, action_mask,
                   direction="reverse"):
    logp, logq = valid_logprobs(student_logits, teacher_logits, action_mask)
    return kl_per_row(logp, logq, direction).mean()


def sampled_opd_pg(current_logp, behavior_logp, teacher_logq, action_mask,
                   signal="behavior"):
    # Inputs: [B,L] on the SAME sampled actions; behavior records actual sampling.
    if (current_logp.shape != behavior_logp.shape
            or current_logp.shape != teacher_logq.shape
            or current_logp.shape != action_mask.shape
            or current_logp.ndim != 2 or action_mask.dtype != torch.bool):
        raise ValueError("expected matching [B,L] tensors and a boolean mask")
    if not action_mask.any().item():
        raise ValueError("batch has no valid model action")
    lp = current_logp[action_mask]
    lb = behavior_logp.detach()[action_mask]
    lq = teacher_logq.detach()[action_mask]
    dtype = torch.float64 if lp.dtype == torch.float64 else torch.float32
    lp, lb, lq = lp.to(dtype), lb.to(dtype), lq.to(dtype)
    if not all(torch.isfinite(v).all().item() for v in (lp, lb, lq)):
        raise ValueError("valid sampled logprobs must be finite")
    if signal == "behavior":
        advantage = (lq - lb).detach()  # Local RKL match only at p == b.
    elif signal == "current":
        advantage = (lq - lp).detach()  # Current local RKL, with full-support IS.
    else:
        raise ValueError("signal must be behavior or current")
    ratio = (lp - lb).exp()  # Keep the numerator differentiable.
    if not torch.isfinite(ratio).all().item():
        raise ValueError("importance ratio overflow; inspect policy mismatch")
    return (-ratio * advantage).mean()


def topk_opd(logp, logq, k, direction="forward", mode="tail"):
    # Already normalized logprobs [N,V], common vocabulary, finite full support.
    # This reference has full teacher logits; a real API may only supply q on S.
    if logp.ndim != 2 or logq.shape != logp.shape:
        raise ValueError("expected matching [N,V] logprobs")
    if not 1 <= k <= logp.shape[-1]:
        raise ValueError("k must be between 1 and V")
    logq = logq.detach()
    ids = logq.topk(k, dim=-1).indices  # SAME token IDs for both models.
    lp = logp.gather(-1, ids)
    lq = logq.gather(-1, ids)
    if mode == "renormalized":
        lp = lp - lp.logsumexp(-1, keepdim=True)
        lq = lq - lq.logsumexp(-1, keepdim=True)
    elif mode == "tail":
        if k < logp.shape[-1]:
            selected = torch.zeros_like(logp, dtype=torch.bool)
            selected.scatter_(-1, ids, True)
            # Stable log tail; avoids subtracting an almost-one head mass.
            pt = logp.masked_fill(selected, -torch.inf).logsumexp(-1, keepdim=True)
            qt = logq.masked_fill(selected, -torch.inf).logsumexp(-1, keepdim=True)
            lp, lq = torch.cat((lp, pt), -1), torch.cat((lq, qt), -1)
    elif mode != "partial":
        raise ValueError("mode must be partial, renormalized or tail")
    # In partial mode these are incomplete sums, not normalized KLs.
    return kl_per_row(lp, lq, direction).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('opd-supervision-lab-results.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks, comparison = [], []
    dtype = torch.float64

    def close(a, b, atol=1e-9):
        torch.testing.assert_close(torch.as_tensor(a, dtype=dtype),
                                   torch.as_tensor(b, dtype=dtype), atol=atol, rtol=1e-8)

    def record(name, **values):
        checks.append({'name': name, 'passed': True, **values})

    def grad_of(fn, z):
        v = z.detach().clone().requires_grad_()
        loss = fn(v)
        return loss.detach(), torch.autograd.grad(loss, v)[0]

    p = torch.tensor([.10, .15, .25, .50], dtype=dtype)
    q = torch.tensor([.30, .40, .20, .10], dtype=dtype)
    z, tq = p.log(), q.log()
    ones = torch.ones((1, 1), dtype=torch.bool)

    def full(v, direction):
        return full_vocab_opd(v[None, None, :], tq[None, None, :], ones, direction)

    for direction in ('forward', 'reverse'):
        value, gradient = grad_of(lambda v: full(v, direction), z)
        expected = p - q if direction == 'forward' else p * (p.log() - q.log() - value)
        close(gradient, expected)
        record('full_' + direction + '_analytic_logit_gradient', value=value.item(),
               gradient=gradient.tolist())

        v = z.clone().requires_grad_()
        assert torch.autograd.gradcheck(lambda w: full(w, direction), (v,))
        record('full_' + direction + '_finite_difference_gradcheck')

        if direction == 'forward':
            lib = F.kl_div(z, tq, reduction='sum', log_target=True)
        else:
            lib = F.kl_div(tq, z, reduction='sum', log_target=True)
        close(lib, value)
        record('pytorch_kl_div_argument_order_' + direction)
        def library_loss(v):
            lp = v.log_softmax(-1)
            return (F.kl_div(lp, tq, reduction='sum', log_target=True)
                    if direction == 'forward'
                    else F.kl_div(tq, lp, reduction='sum', log_target=True))
        _, lib_gradient = grad_of(library_loss, z)
        close(lib_gradient, gradient)
        record('pytorch_kl_div_student_gradient_' + direction)

        for mode in ('partial', 'renormalized', 'tail'):
            fn = lambda v: topk_opd(v.log_softmax(-1)[None, :], tq[None, :],
                                    2, direction, mode)
            val, g = grad_of(fn, z)
            comparison.append({'mode': mode, 'direction': direction,
                               'loss': val.item(), 'logit_gradient': g.tolist()})
            assert torch.autograd.gradcheck(fn, (z.clone().requires_grad_(),))
            record('top2_' + mode + '_' + direction + '_gradcheck')
            same = topk_opd(z[None, :], tq[None, :], len(p), direction, mode)
            close(same, value)
            record('topk_equals_full_at_k_V_' + mode + '_' + direction)

    # Same fixed prefix: enumerate every possible sampled action, no MC uncertainty.
    def pg_expectation(v, b, signal):
        logp = v.log_softmax(-1)
        terms = [b[a] * sampled_opd_pg(logp[a].reshape(1, 1),
                 b[a].log().reshape(1, 1), tq[a].reshape(1, 1), ones, signal)
                 for a in range(len(p))]
        return torch.stack(terms).sum()

    _, gr = grad_of(lambda v: full(v, 'reverse'), z)
    _, gf = grad_of(lambda v: full(v, 'forward'), z)
    for signal in ('behavior', 'current'):
        _, g = grad_of(lambda v: pg_expectation(v, p, signal), z)
        close(g, gr)
        record('sampled_' + signal + '_expected_gradient_at_behavior', gradient=g.tolist())
    b = torch.tensor([.25, .25, .40, .10], dtype=dtype)
    _, current = grad_of(lambda v: pg_expectation(v, b, 'current'), z)
    _, stale = grad_of(lambda v: pg_expectation(v, b, 'behavior'), z)
    close(current, gr)
    assert (stale - gr).abs().max() > .01
    record('current_signal_IS_matches_and_old_signal_does_not_away_from_behavior',
           current_gradient=current.tolist(), old_signal_gradient=stale.tolist())

    # Value and fixed-action autograd must be checked separately.
    for estimator, target in [('k1', torch.zeros_like(p)), ('k2', gr), ('k3', gf)]:
        def scalar(v):
            r = v.log_softmax(-1) - tq
            terms = {'k1': r, 'k2': r.square()/2,
                     'k3': torch.exp(-r) + r - 1}[estimator]
            return (p * terms).sum()  # p is frozen: enumerate sampled-action weights.
        value, gradient = grad_of(scalar, z)
        close(gradient, target)
        record('fixed_action_autograd_' + estimator, value=value.item(),
               expected_gradient=gradient.tolist())

    # One sampled EOS gives a dense vector, although only q_EOS is queried.
    _, sample_g = grad_of(lambda v: sampled_opd_pg(v.log_softmax(-1)[3].reshape(1, 1),
                         z[3].reshape(1, 1), tq[3].reshape(1, 1), ones), z)
    expected = (z[3] - tq[3]) * (F.one_hot(torch.tensor(3), 4).to(dtype) - p)
    close(sample_g, expected)
    assert torch.count_nonzero(sample_g).item() == 4
    record('one_sample_still_has_dense_softmax_logit_gradient', gradient=sample_g.tolist())

    # A state-only KL used as a frozen immediate PG weight has zero expected score.
    state_kl = full(z, 'reverse').detach()
    _, wrong = grad_of(lambda v: -(p*state_kl*v.log_softmax(-1)).sum(), z)
    close(wrong, torch.zeros_like(p))
    assert not torch.allclose(wrong, gr)
    record('state_constant_KL_is_not_a_valid_replacement_for_action_specific_PG_signal')

    for direction in ('forward', 'reverse'):
        s = z.reshape(1, 1, 4).clone().requires_grad_()
        t = tq.reshape(1, 1, 4).clone().requires_grad_()
        full_vocab_opd(s, t, ones, direction).backward()
        assert t.grad is None and s.grad is not None
        record('teacher_frozen_in_full_' + direction)
    s = z[None, :].clone().requires_grad_()
    t = tq[None, :].clone().requires_grad_()
    topk_opd(s.log_softmax(-1), t.log_softmax(-1), 2).backward()
    assert t.grad is None and s.grad is not None
    record('teacher_frozen_in_topk')
    s = torch.tensor([[-.7]], dtype=dtype, requires_grad=True)
    t = torch.tensor([[-1.1]], dtype=dtype, requires_grad=True)
    old = torch.tensor([[-.7]], dtype=dtype, requires_grad=True)
    sampled_opd_pg(s, old, t, ones, 'current').backward()
    assert t.grad is None and old.grad is None and s.grad is not None
    record('teacher_and_behavior_frozen_in_sampled')

    # Invalid rows are removed before arithmetic, valid nonfinite rows raise.
    mask = torch.tensor([[True, False]])
    s = torch.stack((z, torch.full_like(z, torch.nan)))[None, :].requires_grad_()
    t = torch.stack((tq, torch.full_like(tq, -torch.inf)))[None, :]
    value = full_vocab_opd(s, t, mask, 'forward')
    value.backward()
    close(value, full(z, 'forward'))
    close(s.grad[0, 1], torch.zeros_like(z))
    assert torch.isfinite(s.grad).all()
    record('invalid_rows_selected_before_softmax_and_backward')
    lp = torch.tensor([[-.7, float('nan')]], dtype=dtype, requires_grad=True)
    lb = torch.tensor([[-.7, float('nan')]], dtype=dtype)
    lq = torch.tensor([[-1.1, -float('inf')]], dtype=dtype)
    sampled_opd_pg(lp, lb, lq, mask).backward()
    assert torch.isfinite(lp.grad).all() and lp.grad[0, 1] == 0
    record('invalid_rows_selected_before_sampled_ratio')
    for name, fn in [
        ('empty_mask', lambda: full_vocab_opd(s, t, torch.zeros_like(mask))),
        ('valid_nonfinite_row', lambda: full_vocab_opd(s, t, torch.ones_like(mask))),
        ('invalid_k', lambda: topk_opd(z[None, :], tq[None, :], 0)),
    ]:
        try:
            fn()
        except ValueError:
            record('reject_' + name)
        else:
            raise AssertionError('expected ValueError: ' + name)

    # Compare the outside-token FKL gradients with independent closed forms.
    _, partial = grad_of(lambda v: topk_opd(v.log_softmax(-1)[None, :],
                        tq[None, :], 2, 'forward', 'partial'), z)
    _, normalized = grad_of(lambda v: topk_opd(v.log_softmax(-1)[None, :],
                           tq[None, :], 2, 'forward', 'renormalized'), z)
    _, tail = grad_of(lambda v: topk_opd(v.log_softmax(-1)[None, :],
                     tq[None, :], 2, 'forward', 'tail'), z)
    qs, ps = q[:2].sum(), p[:2].sum()
    close(partial[2:], qs*p[2:])
    close(normalized[2:], torch.zeros(2, dtype=dtype))
    close(tail[2:], p[2:]*(1-(1-qs)/(1-ps)))
    record('partial_normalized_tail_have_different_outside_gradients',
           partial_EOS=partial[-1].item(), normalized_EOS=normalized[-1].item(),
           tail_EOS=tail[-1].item(), full_EOS=gf[-1].item())
    val, g = grad_of(lambda v: topk_opd(v.log_softmax(-1)[None, :],
                    tq[None, :], 2, 'reverse', 'partial'), z)
    expected_partial = -p*(val+ps)
    expected_partial[:2] += p[:2]*(z[:2]-tq[:2]+1)
    close(g, expected_partial)
    record('partial_reverse_KL_gradient_includes_uncancelled_mass_term')

    # A fixed tail category preserves mass but loses its internal proportions.
    coarse_p = torch.cat((p[:2], p[2:].sum().reshape(1))).log()
    coarse_q = torch.cat((q[:2], q[2:].sum().reshape(1))).log()
    for direction, tail_weight in [('forward', 1-qs), ('reverse', 1-ps)]:
        fine = kl_per_row(z, tq, direction)
        coarse = kl_per_row(coarse_p, coarse_q, direction)
        internal = kl_per_row((p[2:]/(1-ps)).log(), (q[2:]/(1-qs)).log(), direction)
        close(fine, coarse+tail_weight*internal)
        assert coarse < fine
        record('multi_token_tail_omitted_internal_' + direction, full=fine.item(),
               coarse=coarse.item(), omitted=(tail_weight*internal).item())
        membership = kl_per_row(torch.stack((ps, 1-ps)).log(),
                                torch.stack((qs, 1-qs)).log(), direction)
        head = kl_per_row((p[:2]/ps).log(), (q[:2]/qs).log(), direction)
        head_weight = qs if direction == 'forward' else ps
        close(coarse, membership+head_weight*head)
        record('tail_is_membership_KL_plus_weighted_head_KL_' + direction)

    # Almost-one head mass: logsumexp tail is finite even when 1 - head == 0.
    extreme = torch.tensor([100., 99., -100., -101.], dtype=dtype)
    teacher_extreme = torch.tensor([0., -1., -10., -11.], dtype=dtype).log_softmax(-1)
    for direction in ('forward', 'reverse'):
        value, g = grad_of(lambda v: topk_opd(v.log_softmax(-1)[None, :],
                          teacher_extreme[None, :], 2, direction, 'tail'), extreme)
        assert torch.isfinite(value) and torch.isfinite(g).all()
        record('log_space_tiny_tail_is_finite_' + direction)

    # A numerical optimization of THREE independent logits, never an LLM.
    traces = {}
    for mode in ('full', 'renormalized', 'tail'):
        v = torch.tensor([.05, .05, .90], dtype=dtype).log().requires_grad_()
        teacher = torch.tensor([.45, .45, .10], dtype=dtype).log()[None, :]
        optimizer = torch.optim.SGD([v], lr=.5)
        history = [v.softmax(-1)[-1].item()]
        for _ in range(20):
            optimizer.zero_grad()
            lp = v.log_softmax(-1)[None, :]
            loss = (kl_per_row(lp, teacher, 'forward').mean() if mode == 'full'
                    else topk_opd(lp, teacher, 2, 'forward', mode))
            loss.backward()
            optimizer.step()
            history.append(v.softmax(-1)[-1].item())
        traces[mode] = history
    close(traces['full'], traces['tail'])  # Here the omitted set has only EOS.
    close(traces['renormalized'], [.90]*21)
    assert traces['full'][-1] < .2
    record('three_logit_EOS_demo_full_and_tail_correct_but_renormalized_is_blind',
           final_EOS={key: values[-1] for key, values in traces.items()})

    # Unbiased MC is an expectation claim, not an exact individual sample.
    generator = torch.Generator().manual_seed(20261025)
    actions = torch.multinomial(p, 100000, replacement=True, generator=generator)
    score = torch.eye(4, dtype=dtype) - p[None, :]
    sample_gradients = (z-tq)[:, None] * score
    close((p[:, None]*sample_gradients).sum(0), gr)
    estimate = sample_gradients[actions].mean(0)
    close(estimate, gr, atol=.01)
    record('seeded_100000_sample_gradient_mean', samples=100000,
           expected=gr.tolist(), empirical=estimate.tolist(),
           max_abs_error=(estimate-gr).abs().max().item())

    result = {
        'executed_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(), 'torch_version': torch.__version__,
        'device': 'cpu', 'dtype': 'float64 for analytic/finite-difference checks',
        'scope': 'Independent logits on fixed prefixes; no LLM, API, GPU, DDP or performance benchmark.',
        'student': p.tolist(), 'teacher': q.tolist(), 'vocabulary': ['A', 'B', 'C', 'EOS'],
        'check_count': len(checks), 'all_passed': True, 'checks': checks,
        'top2_comparison': comparison, 'three_logit_EOS_traces': traces,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(f'{len(checks)} PyTorch CPU checks passed; wrote {args.output}')


if __name__ == '__main__':
    main()
