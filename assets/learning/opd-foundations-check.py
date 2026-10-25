#!/usr/bin/env python3
"""Executable mathematical examples for the OPD foundations chapter.

Standard library only. All expectations are enumerated exactly: these checks
verify finite-distribution identities, not language-model training outcomes.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


def kl(p, q):
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a)


def score(p, action):
    return [float(i == action) - p[i] for i in range(len(p))]


def expectation_gradient(p, weights):
    return [sum(p[a] * weights[a] * score(p, a)[i]
                for a in range(len(p))) for i in range(len(p))]


def finite_gradient(fn, z, eps=1e-6):
    out = []
    for i in range(len(z)):
        plus, minus = list(z), list(z)
        plus[i] += eps
        minus[i] -= eps
        out.append((fn(plus) - fn(minus)) / (2 * eps))
    return out


def assert_close(actual, expected, tol=2e-8):
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_close(a, e, tol)
    else:
        assert math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), (actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('opd-foundations-check-results.json'))
    args = parser.parse_args()
    checks = []

    def record(name, **values):
        checks.append({'name': name, 'passed': True, **values})

    z = [0.2, -0.4, 0.7]
    p, q = softmax(z), [0.65, 0.25, 0.10]  # p: student; q: teacher.
    fkl, rkl = kl(q, p), kl(p, q)
    gf = [a - b for a, b in zip(p, q)]
    gr = [a * (math.log(a / b) - rkl) for a, b in zip(p, q)]
    assert_close(finite_gradient(lambda v: kl(q, softmax(v)), z), gf)
    assert_close(finite_gradient(lambda v: kl(softmax(v), q), z), gr)
    record('full_vocabulary_logit_gradients', student=p, teacher=q,
           forward_kl=fkl, reverse_kl=rkl, forward_gradient=gf, reverse_gradient=gr)

    cross_entropy = -sum(a * math.log(b) for a, b in zip(q, p))
    teacher_entropy = -sum(a * math.log(a) for a in q)
    assert_close(fkl, cross_entropy - teacher_entropy)
    record('forward_kl_equals_cross_entropy_minus_teacher_entropy',
           cross_entropy=cross_entropy, teacher_entropy=teacher_entropy)

    log_ratio = [math.log(a / b) for a, b in zip(p, q)]
    expected_score = expectation_gradient(p, [1.0] * len(p))
    expected_pg = expectation_gradient(p, log_ratio)
    assert_close(expected_score, [0.0] * len(p))
    assert_close(expected_pg, gr)
    record('sampled_kl_value_and_gradient_are_different_questions',
           expected_naive_k1_gradient=expected_score, expected_pg_gradient=expected_pg)

    # Different scalar estimators can have different fixed-index autograd paths.
    k2_value = sum(a * 0.5 * r * r for a, r in zip(p, log_ratio))
    k3_value = sum(a * (math.exp(-r) + r - 1) for a, r in zip(p, log_ratio))
    k3_gradient = expectation_gradient(p, [1 - b / a for a, b in zip(p, q)])
    assert_close(k3_value, rkl)
    assert_close(k3_gradient, gf)
    record('scalar_estimator_value_does_not_determine_its_gradient',
           k2_expected_value=k2_value, k2_expected_gradient=expected_pg,
           k3_expected_value=k3_value, k3_fixed_index_expected_gradient=k3_gradient,
           note='K3 here uses q_teacher/p_student; definitions vary between libraries.')

    behavior = [0.2, 0.2, 0.6]
    is_fkl_gradient = [sum(behavior[a] * q[a] / behavior[a] * -score(p, a)[i]
                           for a in range(len(p))) for i in range(len(p))]
    assert_close(is_fkl_gradient, gf)
    record('student_samples_can_estimate_forward_kl_gradient_with_full_support_is',
           behavior=behavior, expected_gradient=is_fkl_gradient)

    truncated_behavior = [0.5, 0.5, 0.0]
    incomplete_gradient = [sum(q[a] * -score(p, a)[i] for a in [0, 1])
                           for i in range(len(p))]
    assert max(abs(a - b) for a, b in zip(incomplete_gradient, gf)) > 1e-3
    record('truncated_support_cannot_recover_missing_teacher_actions',
           behavior=truncated_behavior, biased_gradient=incomplete_gradient,
           required_gradient=gf)

    eos_student, eos_teacher = [0.05, 0.05, 0.90], [0.45, 0.45, 0.10]
    p_head = [a / sum(eos_student[:2]) for a in eos_student[:2]]
    q_head = [a / sum(eos_teacher[:2]) for a in eos_teacher[:2]]
    assert_close(kl(q_head, p_head), 0.0)
    assert_close(kl(eos_teacher, eos_student), 0.8 * math.log(9))
    assert_close(eos_student[-1] - eos_teacher[-1], 0.8)
    record('normalized_top2_can_hide_premature_eos', student=eos_student,
           teacher=eos_teacher, normalized_top2_forward_kl=0.0,
           full_forward_kl=kl(eos_teacher, eos_student), eos_logit_gradient=0.8)

    p4, q4 = [0.10, 0.15, 0.25, 0.50], [0.30, 0.40, 0.20, 0.10]
    ap, aq = sum(p4[:2]), sum(q4[:2])
    coarse_p, coarse_q = p4[:2] + [1-ap], q4[:2] + [1-aq]
    pt, qt = [v/(1-ap) for v in p4[2:]], [v/(1-aq) for v in q4[2:]]
    assert_close(kl(p4, q4), kl(coarse_p, coarse_q)+(1-ap)*kl(pt, qt))
    assert_close(kl(q4, p4), kl(coarse_q, coarse_p)+(1-aq)*kl(qt, pt))
    record('tail_bucket_is_coarsened_kl_not_full_kl',
           full_reverse_kl=kl(p4, q4), coarse_reverse_kl=kl(coarse_p, coarse_q),
           omitted_reverse_kl=(1-ap)*kl(pt, qt),
           full_forward_kl=kl(q4, p4), coarse_forward_kl=kl(coarse_q, coarse_p))

    partial_rkl = eos_student[0] * math.log(eos_student[0] / eos_teacher[0])
    assert partial_rkl < 0
    record('unnormalized_partial_kl_can_be_negative', partial_reverse_sum=partial_rkl)

    # Current-signal IS is a local conditional correction, not an occupancy correction.
    current_rkl_grad = [sum(behavior[a] * p[a] / behavior[a] * log_ratio[a] * score(p, a)[i]
                           for a in range(len(p))) for i in range(len(p))]
    stale_ratio = [math.log(a / b) for a, b in zip(behavior, q)]
    stale_signal_grad = expectation_gradient(p, stale_ratio)
    assert_close(current_rkl_grad, gr)
    assert max(abs(a-b) for a,b in zip(stale_signal_grad, gr)) > 1e-3
    record('action_is_with_current_signal_and_stale_signal_differ',
           current_signal_gradient=current_rkl_grad, stale_signal_gradient=stale_signal_grad)

    # A two-step tree: root policy is trainable; leaf policies are fixed.
    leaf_cost = kl([0.8, 0.2], [0.2, 0.8])
    def sequence_objective(v):
        root = softmax([v[0], 0.0])
        return kl(root, [0.5, 0.5]) + root[0] * leaf_cost
    sequence_grad = finite_gradient(sequence_objective, [0.0])[0]
    local_frozen_prefix_grad = 0.0
    assert_close(sequence_grad, 0.25 * leaf_cost)
    record('local_frozen_prefix_gradient_is_not_full_sequence_gradient',
           root_student=[0.5,0.5], root_teacher=[0.5,0.5],
           fixed_leaf_reverse_kl=leaf_cost, full_sequence_gradient=sequence_grad,
           local_gradient=local_frozen_prefix_grad)

    # Same scalar parameter participates in four tokens, split into lengths 1 and 3.
    token_gradients = [0.4, -0.2, -0.2, -0.2]
    token_mean = sum(token_gradients)/4
    sequence_mean = (token_gradients[0] + sum(token_gradients[1:])/3)/2
    weighted_split = (token_gradients[0] + sum(token_gradients[1:]))/4
    assert_close(token_mean, weighted_split)
    assert not math.isclose(token_mean, sequence_mean)
    record('global_token_denominator_and_sequence_mean_differ',
           sequence_lengths=[1,3], token_mean_gradient=token_mean,
           sequence_mean_gradient=sequence_mean, weighted_microbatch_gradient=weighted_split)

    q1, q2, alpha = [0.8,0.15,0.05], [0.15,0.6,0.25], 0.4
    arithmetic = [alpha*a + (1-alpha)*b for a,b in zip(q1,q2)]
    geometric = softmax([alpha*math.log(a)+(1-alpha)*math.log(b) for a,b in zip(q1,q2)])
    mix_f_grad = [arithmetic[i]-(alpha*q1[i]+(1-alpha)*q2[i]) for i in range(3)]
    mix_r_grad = [0.0]*3
    for weight, teacher in [(alpha,q1),(1-alpha,q2)]:
        value=kl(geometric,teacher)
        for i in range(3):
            mix_r_grad[i] += weight*geometric[i]*(math.log(geometric[i]/teacher[i])-value)
    assert_close(mix_f_grad,[0.0]*3)
    assert_close(mix_r_grad,[0.0]*3)
    record('multi_teacher_forward_and_reverse_have_different_targets',
           forward_optimum=arithmetic, reverse_optimum=geometric)

    double_weighted = sum(a*a*r for a,r in zip(p,log_ratio))
    assert not math.isclose(double_weighted,rkl)
    record('do_not_multiply_a_student_sample_by_student_probability_again',
           correct_expected_log_ratio=rkl, double_weighted_value=double_weighted)

    tokens=['PROMPT','A','B','EOS']
    pairs=[{'context':tokens[:t+1],'target':tokens[t+1]} for t in range(len(tokens)-1)]
    assert pairs[0]['context']==['PROMPT'] and pairs[0]['target']=='A'
    assert pairs[-1]['target']=='EOS'
    record('causal_shift_predicts_the_next_token_including_real_eos', pairs=pairs)

    result={
        'reviewed_at':'2026-10-25',
        'executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'validation_scope':'CPU finite-distribution enumeration and finite differences; no LLM or GPU training.',
        'student_symbol':'p', 'teacher_symbol':'q',
        'check_count':len(checks), 'all_passed':True, 'checks':checks,
    }
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(checks)} checks passed; wrote {args.output}')


if __name__=='__main__':
    main()
