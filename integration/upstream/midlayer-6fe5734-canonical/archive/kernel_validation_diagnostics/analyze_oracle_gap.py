"""Analyze why oracle is worse than forward on C500."""
import json, sys, os
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.kernel.hardware_spec import HWSpec
from src.kernel.formula import PerformanceFormula
from tools.kernel.validation.reference_models.base import create_predictor
from tools.kernel.validation.reference_models.common import is_non_tensor

hw = HWSpec('configs/hardware/metax_c500.yaml')
formula = PerformanceFormula(hw)

for op in ['FLASHATTN', 'GEMM', 'FUSEDMOE']:
    fname = {'FLASHATTN': 'flash_attn', 'GEMM': 'gemm', 'FUSEDMOE': 'fused_moe'}[op]
    ds = json.load(open(f'kernel_dataset/metax_c500/{fname}.json'))
    predictor = create_predictor(op)

    print(f'\n{"="*70}')
    print(f'{op}: Oracle vs Forward demand comparison')
    print(f'{"="*70}')

    count = 0
    for key in sorted(ds.keys()):
        entry = ds[key]
        attr = entry['op_info']['Attr']
        kernels = entry['kernels']

        # Skip incomplete shapes
        has_tensor = any(not is_non_tensor(op, k['Metadata']['Kernel Name'],
                                           ascend_op_type=k['Metadata'].get('Op Type'))
                         for k in kernels)
        if not has_tensor:
            continue

        o_pred, o_act, f_pred, f_act = 0.0, 0.0, 0.0, 0.0
        first_tensor_orc = None
        first_tensor_fwd = None
        bn_list = []

        for kdata in kernels:
            from evaluation.kernel.profiler_observation import extract_ncu_metrics
            actual_ns = extract_ncu_metrics(kdata).get("duration_ns", 0.0)
            inp_o = predictor.oracle(kdata, hw, attr)
            inp_f = predictor.forward(kdata, hw, attr)
            if inp_o is None or inp_f is None:
                continue
            po = formula.predict_kernel(inp_o)
            pf = formula.predict_kernel(inp_f)
            o_pred += po.t_total_ns
            o_act += actual_ns
            f_pred += pf.t_total_ns
            f_act += actual_ns
            bn_list.append(po.bottleneck_group)

            if not inp_o.is_non_tensor and first_tensor_orc is None:
                first_tensor_orc = inp_o
                first_tensor_fwd = inp_f

        if o_act <= 0:
            continue

        oe = (o_pred - o_act) / o_act * 100
        fe = (f_pred - f_act) / f_act * 100

        if count < 8:
            print(f'\n{key}:')
            print(f'  actual={o_act/1e6:.3f}ms  oracle={o_pred/1e6:.3f}ms({oe:+.1f}%)  fwd={f_pred/1e6:.3f}ms({fe:+.1f}%)')

            if first_tensor_orc and first_tensor_fwd:
                od = first_tensor_orc.demand_per_resource
                fd = first_tensor_fwd.demand_per_resource
                print(f'  First tensor kernel demand comparison:')
                for k in sorted(set(list(od.keys()) + list(fd.keys()))):
                    ov = od.get(k, 0)
                    fv = fd.get(k, 0)
                    if ov > 0 or fv > 0:
                        ratio = ov / fv if fv > 0 else float('inf')
                        print(f'    {k:20s}: oracle={ov:>14.0f}  fwd={fv:>14.0f}  ratio={ratio:.2f}')
                print(f'  oracle eta={first_tensor_orc.eta:.3f}  fwd eta={first_tensor_fwd.eta:.3f}')
                print(f'  oracle grid={first_tensor_orc.grid_size}  fwd grid={first_tensor_fwd.grid_size}')
            bc = Counter(bn_list)
            print(f'  bottlenecks: {dict(bc)}')
        count += 1
