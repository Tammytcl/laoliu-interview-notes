"""CPU arithmetic only; no GPU benchmark and no real large-model allocation."""
import json
import math

# Current article's explicitly selected product/datasheet columns.
# C from dense BF16/FP16 Tensor Core peak, W from advertised HBM bandwidth.
products = [
    dict(name='A100 80GB SXM', capacity_GB=80, hbm_TBs=2.039, dense_TFs=312),
    dict(name='H100 SXM', capacity_GB=80, hbm_TBs=3.35, dense_TFs=1979/2),
    dict(name='H200 SXM', capacity_GB=141, hbm_TBs=4.8, dense_TFs=1979/2),
    dict(name='HGX B200 per GPU, fixed datasheet', capacity_GB=180, hbm_TBs=7.7, dense_TFs=4500/2),
    dict(name='GB200 per GPU, fixed datasheet', capacity_GB=186, hbm_TBs=8, dense_TFs=5000/2),
    dict(name='HGX B300 per GPU, fixed datasheet', capacity_GB=270, hbm_TBs=7.7, dense_TFs=4500/2),
]
for product in products:
    product['roofline_knee_FLOP_per_byte'] = product['dense_TFs']/product['hbm_TBs']
    product['one_T_BF16_weight_only_80percent_capacity_device_lower_bound'] = math.ceil(2e12/(product['capacity_GB']*1e9*.8))

state_70B = 70e9*16
assert state_70B/1e9 == 1120
assert state_70B/8/1e9 == 140
assert state_70B/16/1e9 == 70
latencies = {}
for name, bandwidth in [('H100', 3.35), ('H200', 4.8)]:
    compute = 100e12/(1979/2*1e12)
    memory = 400e9/(bandwidth*1e12)
    latencies[name] = dict(compute_lower_bound_ms=compute*1000, memory_lower_bound_ms=memory*1000, ideal_max_ms=max(compute, memory)*1000)
ratio = latencies['H100']['ideal_max_ms']/latencies['H200']['ideal_max_ms']
assert 1.18 < ratio < 1.19
assert math.ceil(2e12/(80e9*.8)) == 32
assert math.ceil(2e12/(180e9*.8)) == 14
assert math.ceil(2e12/(270e9*.8)) == 10

print(json.dumps(dict(scope='CPU derivation of advertised-spec arithmetic; not measured GPU throughput, procurement sizing or model deployment',
                     reviewedAt='2026-10-06', products=products,
                     model_70B_state_GB=state_70B/1e9,
                     model_70B_zero3_D8_state_per_rank_GB=140,
                     model_70B_zero3_D16_state_per_rank_GB=70,
                     H100_H200_hypothetical=latencies,
                     ideal_bound_ratio=ratio,
                     bandwidth_ratio=4.8/3.35,
                     network_400Gbps_byte_rate_GBs=400/8), ensure_ascii=False, indent=2))
