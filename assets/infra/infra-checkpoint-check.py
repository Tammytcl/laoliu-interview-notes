"""Tiny CPU checkpoint experiments. No model downloads, GPU tasks, or persistent temp dirs.
Run: python infra-checkpoint-check.py --output results.json
Requires torch and safetensors. Tests CPU/Gloo tensor resharding, not FSDP/ZeRO training.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
import torch.multiprocessing as mp
from safetensors.torch import save_file, load_file
import safetensors


def worker(rank, world, mode, folder, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=world, init_method=Path(rendezvous).as_uri())
    try:
        mesh = init_device_mesh('cpu', (world,))
        original = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        source = original if mode == 'save' else torch.zeros_like(original)
        weight = distribute_tensor(source, mesh, [Shard(0)])
        state = {'weight': weight}
        checkpoint = str(Path(folder) / 'dcp')
        if mode == 'save':
            dcp.save(state, checkpoint_id=checkpoint)
        else:
            dcp.load(state, checkpoint_id=checkpoint)
            full = state['weight'].full_tensor()
            assert torch.equal(full, original)
            if rank == 0:
                Path(folder, 'reshard.json').write_text(json.dumps({'saved_world_size': 2, 'loaded_world_size': world, 'shape': list(full.shape), 'max_error': (full-original).abs().max().item()}))
    finally:
        dist.destroy_process_group()


def run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)/10
    y = torch.zeros(4, 2)

    def step(module, opt):
        opt.zero_grad(set_to_none=True)
        loss = ((module(x)-y)**2).mean()
        loss.backward()
        opt.step()
        return loss.item()

    step(model, optimizer)
    with tempfile.TemporaryDirectory(prefix='infra-checkpoint-cpu-') as directory:
        folder = Path(directory)
        torch.save(model.state_dict(), folder/'weights.pt')
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': 1, 'rng': torch.get_rng_state()}, folder/'training.ckpt')
        state = torch.load(folder/'weights.pt', map_location='cpu', weights_only=True)
        save_file({k: v.contiguous() for k, v in state.items()}, str(folder/'weights.safetensors'))
        restored_weights = load_file(str(folder/'weights.safetensors'))
        assert all(torch.equal(state[k], restored_weights[k]) for k in state)
        raw = (folder/'weights.safetensors').read_bytes()
        header_length = struct.unpack('<Q', raw[:8])[0]
        header = json.loads(raw[8:8+header_length])
        data_bytes = len(raw)-8-header_length
        tensor_bytes = sum(v.numel()*v.element_size() for v in state.values())
        assert data_bytes == tensor_bytes == 32
        for key, value in header.items():
            if key == '__metadata__':
                continue
            assert value['data_offsets'][1]-value['data_offsets'][0] == restored_weights[key].numel()*restored_weights[key].element_size()

        # A toy HF-style file index: whole tensors, not TP slices.
        names = list(state)
        weight_map = {}
        for index, name in enumerate(names, start=1):
            filename = f'model-{index:05}-of-00002.safetensors'
            save_file({name: state[name]}, str(folder/filename))
            weight_map[name] = filename
        index = {'metadata': {'total_size': tensor_bytes}, 'weight_map': weight_map}
        (folder/'model.safetensors.index.json').write_text(json.dumps(index))
        combined = {}
        for filename in sorted(set(weight_map.values())):
            combined.update(load_file(str(folder/filename)))
        assert all(torch.equal(state[k], combined[k]) for k in state)

        checkpoint = torch.load(folder/'training.ckpt', map_location='cpu', weights_only=True)
        resumed = torch.nn.Linear(3, 2)
        resumed.load_state_dict(checkpoint['state_dict'])
        resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.01)
        resumed_optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        loss_a, loss_b = step(model, optimizer), step(resumed, resumed_optimizer)
        assert loss_a == loss_b
        parameter_error = max((a-b).abs().max().item() for a, b in zip(model.parameters(), resumed.parameters()))
        assert parameter_error == 0
        a_states, b_states = optimizer.state_dict()['state'], resumed_optimizer.state_dict()['state']
        for key in a_states:
            for field in ['step', 'exp_avg', 'exp_avg_sq']:
                assert torch.equal(a_states[key][field], b_states[key][field])
        sizes = {name: (folder/name).stat().st_size for name in ['weights.pt', 'training.ckpt', 'weights.safetensors']}
        mp.spawn(worker, args=(2, 'save', directory, str(folder/'save-rendezvous')), nprocs=2, join=True)
        mp.spawn(worker, args=(1, 'load', directory, str(folder/'load-rendezvous')), nprocs=1, join=True)
        reshared = json.loads((folder/'reshard.json').read_text())
        dcp_files = sorted(path.name for path in (folder/'dcp').iterdir())

    capacity = []
    for name, byte_per_parameter in [('FP32', 4), ('BF16/FP16', 2), ('8-bit payload', 1), ('packed 4-bit payload', .5)]:
        size = 1e12*byte_per_parameter
        capacity.append({'encoding': name, 'bytes': int(size), 'TB': size/1e12, 'TiB': size/2**40, 'weight_only_gpus_80GB': math.ceil(size/80e9), 'weight_only_gpus_64GB': math.ceil(size/64e9)})
    kv = 2*128*8*32768*8*128*2
    assert kv/2**30 == 128
    assert math.ceil((2e12+kv)/64e9) == 34
    return {'scope': 'Tiny CPU serialization/Adam recovery and CPU Gloo DTensor 2-to-1 reshard only; no GPU training or actual 1T model',
            'torch_version': torch.__version__, 'safetensors_version': safetensors.__version__,
            'serialization': {'unique_parameters': 8, 'tensor_payload_bytes': tensor_bytes, 'file_bytes': sizes, 'header_bytes': header_length, 'header': header},
            'hf_style_index': index, 'adam_resume': {'next_loss': loss_a, 'max_parameter_error': parameter_error, 'optimizer_states_equal': True},
            'dcp_reshard': {**reshared, 'files': dcp_files},
            'capacity_1T': capacity, 'int4_group128_scale_zero_4bytes_GB': 1e12*(.5+4/128)/1e9,
            'kv_example_GiB': kv/2**30, 'BF16_plus_KV_uniform_capacity_gpus_64GB': 34,
            'storage_examples_TB': {'FP32_params_plus_FP32_Adam': 12, 'BF16_plus_master_plus_Adam': 14, 'three_12TB_checkpoints_plus_2TB_export': 38}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = run()
    text = json.dumps(results, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text)
