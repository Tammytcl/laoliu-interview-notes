"""CPU-only teaching checks. Run: python infra-foundations-check.py (stdlib only)."""
import json
import math


def close(a, b, tolerance=1e-9):
    assert abs(a - b) <= tolerance, (a, b)


x, y, w = [1., 2., 3., 4.], [2., 4., 6., 8.], 0.5
terms = [2 * (w * a - b) * a for a, b in zip(x, y)]
grad = sum(terms) / 4
weighted = sum(sum(part) / 4 for part in [terms[:3], terms[3:]])
wrong = (sum(terms[:3]) / 3 + terms[3]) / 2
loss = lambda weight: sum((weight * a - b) ** 2 for a, b in zip(x, y)) / 4
finite_difference = (loss(w + 1e-5) - loss(w - 1e-5)) / 2e-5
close(grad, weighted)
close(grad, finite_difference, 1e-7)
close(grad, -22.5)
close(wrong, -31)

# Model states: BF16 param/grad + FP32 master + two FP32 Adam moments.
state_bytes = 7_000_000_000 * (2 + 2 + 4 + 8)
close(state_bytes / 1e9, 112)
close(state_bytes / 2**30, 104.3081283569336)
zero_gb = [7 * b for b in [16, 4 + 12/8, 2 + 14/8, 16/8]]
assert zero_gb == [112, 38.5, 26.25, 14]
kv_bytes = 2 * 32 * 8 * 8192 * 8 * 128 * 2
close(kv_bytes / 2**30, 8)

roofline = []
for batch in [1, 128]:
    flops = 2 * batch * 4096**2
    byte_count = 2 * (4096**2 + 2 * batch * 4096)
    roofline.append(dict(batch=batch, flops=flops, bytes=byte_count,
                         intensity=flops/byte_count,
                         compute_us=flops/1e14*1e6,
                         memory_us=byte_count/1e12*1e6))
close(roofline[1]['intensity'], 120.47058823529412)
ring_send_bytes = 2 * (8-1)/8 * 2**30
close(ring_send_bytes / 2**30, 1.75)

# Blockwise online softmax versus full attention, including extreme scores.
scores, values = [-1000., 0.2, 1.1, 100., 99.5], [1., 3., -2., 5., 9.]
m, denominator, numerator = -math.inf, 0., 0.
for start, end in [(0, 2), (2, 3), (3, 5)]:
    block = scores[start:end]
    new_m = max(m, max(block))
    scale = math.exp(m-new_m)
    denominator = scale*denominator + sum(math.exp(s-new_m) for s in block)
    numerator = scale*numerator + sum(math.exp(s-new_m)*v for s, v in zip(block, values[start:end]))
    m = new_m
blockwise = numerator/denominator
full = sum(math.exp(s-max(scores))*v for s, v in zip(scores, values)) / sum(math.exp(s-max(scores)) for s in scores)
close(blockwise, full)

# Column-parallel W1 and row-parallel W2 with a local pointwise nonlinearity.
def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, column)) for column in zip(*b)] for row in a]

inputs = [[1., -1., 0.5, 2.], [-0.5, 0.1, 1., 3.]]
w1 = [[(i-j)/10 for j in range(8)] for i in range(4)]
w2 = [[(i+j)/20 for j in range(4)] for i in range(8)]
relu = lambda matrix: [[max(0, v) for v in row] for row in matrix]
full_mlp = matmul(relu(matmul(inputs, w1)), w2)
partials = [matmul(relu(matmul(inputs, [row[start:end] for row in w1])), w2[start:end]) for start, end in [(0, 4), (4, 8)]]
parallel_mlp = [[sum(part[row][column] for part in partials) for column in range(4)] for row in range(2)]
for a, b in zip(sum(full_mlp, []), sum(parallel_mlp, [])):
    close(a, b)

result = dict(
    scope='CPU arithmetic, finite differences and algebraic equivalence; no GPU training or performance benchmark',
    gradient=dict(full=grad, weighted=weighted, mean_of_means=wrong, finite_difference=finite_difference),
    memory=dict(state_GB=state_bytes/1e9, state_GiB=state_bytes/2**30, zero_GB=zero_gb, kv_GiB=kv_bytes/2**30),
    roofline=roofline,
    ring=dict(send_GiB=ring_send_bytes/2**30, transmission_ms=ring_send_bytes/25e9*1e3),
    online_softmax=dict(full=full, blockwise=blockwise, absolute_error=abs(full-blockwise)),
    tensor_parallel=dict(full=full_mlp, partitioned=parallel_mlp),
    pipeline_bubble=dict(p4_m4=3/7, p4_m16=3/19),
    queue=dict(unbounded_backlog_at_300s=(12-8)*300, stable_ready_wait_s=80/8)
)
print(json.dumps(result, indent=2, ensure_ascii=False))
