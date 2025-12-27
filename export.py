from copy import deepcopy

import numpy as np
import onnx
import onnxsim
import torch

from models.networks import build_G_model

def export_to_onnx(G_model, gen_z, gen_c, config, onnx_model_path):
    print(f'Exporting ONNX model to {onnx_model_path}...')
    # 2. Set params
    dynamo_export = False
    opset_version = 20  # set to another value later, good values for 09.06.2025 are 20, 21
    use_onnxsim = False
    export_truncation_params = False  # error with True
    # 3. Prepare model and inputs
    export_device = torch.device('cpu')
    src_weights = G_model.state_dict()
    model_export_config = deepcopy(config['models_params']['Generator'])
    model_export_config['Synthesis']['fused_modconv_default'] = False
    export_config = deepcopy(config)
    export_config['general_params']['num_fp16_res'] = 0  # probably enable for GPU inference, but disable for CPU
    export_config['general_params']['use_compilation'] = False
    export_config['training_params']['use_custom_conv2d_op'] = False
    export_config['training_params']['upfirdn2d_impl'] = 'ref'
    export_config['training_params']['bias_act_impl'] = 'ref'
    export_model = build_G_model(model_export_config, export_config)
    export_model.load_state_dict(src_weights)
    print('Built export model and set source weights')
    export_model = export_model.to(memory_format=torch.contiguous_format).to(export_device).eval().requires_grad_(False)
    # [z, c, truncation_psi=1, truncation_cutoff=None]
    input_names = ['z', 'c', 'truncation_psi', 'truncation_cutoff']
    input_args = (
        gen_z.to(device=export_device),
        gen_c.to(device=export_device),
        torch.from_numpy(np.array(0.99, dtype=np.float32)),
        torch.from_numpy(np.array(G_model.mapping.num_ws, dtype=np.int32)),
    )
    if not export_truncation_params:
        input_args = input_args[:2]
        input_names = input_names[:2]
    # 4. Convert model
    torch.onnx.export(
        export_model,
        input_args,
        onnx_model_path,
        input_names=input_names,
        opset_version=opset_version,
        dynamo=dynamo_export,
        export_params=True,
        do_constant_folding=True,
    )
    print(f'Model converted: {onnx_model_path}')
    if use_onnxsim:
        print('Generating simplified ONNX model...')
        onnx_model = onnx.load(onnx_model_path)
        simplified_onnx_model, check_status = onnxsim.simplify(onnx_model)
        assert check_status, 'Simplified model could not be validated'
        onnx.save(simplified_onnx_model, onnx_model_path)
        print(f'Simplified ONNX model: {onnx_model_path}')
    # return status if eveyting is fine
    return True
