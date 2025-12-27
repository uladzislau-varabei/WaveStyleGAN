# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy
import re
import contextlib
import numpy as np
import torch
import warnings


#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


#----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0


#----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)


#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')


#----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().

def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator


#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


#----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def get_num_params_and_buffers_message(module):
    assert isinstance(module, torch.nn.Module)
    num_params = int(sum([p.numel() for p in module.parameters()]))
    num_buffers = int(sum([p.numel() for p in module.buffers()]))
    return f'params={num_params:,}, buffers={num_buffers:,}'


#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


#----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname


#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    is_data_parallel = isinstance(module, torch.nn.DataParallel)
    module = copy.deepcopy(module.module if is_data_parallel else module)
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append({'mod': mod, 'outputs': outputs})
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e['unique_params'] = [t for t in e['mod'].parameters() if id(t) not in tensors_seen]
        e['unique_buffers'] = [t for t in e['mod'].buffers() if id(t) not in tensors_seen]
        e['unique_outputs'] = [t for t in e['outputs'] if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e['unique_params'] + e['unique_buffers'] + e['unique_outputs']}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [
            e for e in entries if len(e['unique_params']) or len(e['unique_buffers']) or len(e['unique_outputs'])
        ]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e['mod'] is module else submodule_names[e['mod']]
        param_size = sum(t.numel() for t in e['unique_params'])
        buffer_size = sum(t.numel() for t in e['unique_buffers'])
        output_shapes = [str(list(t.shape)) for t in e['outputs']]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e['outputs']]
        rows += [[
            name + (':0' if len(e['outputs']) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e['outputs'])):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    del module # remove deepcopy result
    return outputs


def make_top_values_table(values, num_top, name):
    def make_row(name, shape, num, ratio, name_size, shape_size, num_size, ratio_size, format=True):
        num_str = f'{num:,}' if format else num
        ratio_str = f'{ratio:.2f} %' if format else ratio
        shape_str = str(list(shape)) if format else shape
        row = f'{name:<{name_size}} {shape_str:<{shape_size}} {num_str:<{num_size}} {ratio_str:<{ratio_size}}\n'
        return row

    sorted_values = sorted(values, key=lambda x: x[2], reverse=True)[:num_top]
    name_size = max([len(v[0]) for v in sorted_values]) + 5
    shape_size = max([len(str(list(v[1]))) for v in sorted_values]) + 5
    total_num = sum([v[2] for v in values])

    kwargs = dict(name_size=name_size, shape_size=shape_size, num_size=12, ratio_size=8)
    table = '   ' + make_row(name, 'Shape', 'Number', 'Ratio', format=False, **kwargs)
    table_line = '-' * (len(table) + 5) + '\n'
    table += table_line
    top_total_num = 0
    for idx, (name, shape, value) in enumerate(sorted_values, 1):
        top_total_num += value
        ratio = 100 * (value / total_num)
        table += f'{idx}) ' + make_row(name, shape, value, ratio, **kwargs)
    top_ratio = 100 * (top_total_num / total_num)
    table += table_line
    table += '   ' + make_row(f'Total (top={num_top})', [], top_total_num, top_ratio, **kwargs)
    table += table_line
    table += '   ' + make_row('Total', [], total_num, 100, **kwargs)
    return table


def print_module_summary_v2(module, inputs, include_top_summary=True, num_top=10):
    is_data_parallel = isinstance(module, torch.nn.DataParallel)
    module = copy.deepcopy(module.module if is_data_parallel else module)
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    def check_is_resolution_block(name):
        parts = name.split('.')
        numbers = sum([p.isdigit() for p in parts])
        return (parts[-1].isdigit() and numbers == 1)

    def count_params(m):
        return sum([x.numel() for x in m.parameters()])

    def count_buffers(m):
        return sum([x.numel() for x in m.buffers()])

    def make_row(name, params, buffers, name_size, param_size, buffer_size, format=True):
        params_str = f'{params:,}' if format else params
        buffers_str = f'{buffers:,}' if format else buffers
        row = f'{name:<{name_size}} {params_str:<{param_size}} {buffers_str:<{buffer_size}}\n'
        return row

    named_modules = [(name, mod) for name, mod in module.named_modules() if len(name) > 0]
    # TODO: update res_modules for source StyleGAN models
    res_modules = [(name, mod) for name, mod in named_modules if check_is_resolution_block(name)]
    if len(res_modules) > 0:
        max_name_size = max([len(name) for name, mod in res_modules]) + 10
    else:
        # For source StyleGAN models
        max_name_size = 30
    module_name = type(module).__name__
    kwargs = dict(name_size=max_name_size, param_size=12, buffer_size=12)
    table = make_row(module_name, 'Params', 'Buffers', format=False, **kwargs)
    table_line = '-' * (len(table) + 5) + '\n'
    table += table_line
    total_params = 0
    total_buffers = 0
    for name, mod in res_modules:
        num_params = count_params(mod)
        total_params += num_params
        num_buffers = count_buffers(mod)
        total_buffers += num_buffers
        table += make_row(name, num_params, num_buffers, **kwargs)
    table += table_line
    table += make_row('Total', total_params, total_buffers, **kwargs)

    summary = f'{table}\n'
    if include_top_summary:
        params_values = [(name, p.shape, p.numel()) for name, p in module.named_parameters()]
        buffers_values = [(name, p.shape, p.numel()) for name, p in module.named_buffers()]
        params_top_table = make_top_values_table(params_values, num_top, f'{module_name}: params')
        buffers_top_table = make_top_values_table(buffers_values, num_top, f'{module_name}: buffers')
        summary += f'\n{params_top_table}\n{buffers_top_table}'

    print(summary)

    total_params = sum([p.numel() for name, p in module.named_parameters()])
    total_buffers = sum([p.numel() for name, p in module.named_buffers()])

    return None, summary, total_params, total_buffers


#----------------------------------------------------------------------------
