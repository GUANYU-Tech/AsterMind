#!/usr/bin/env python3
"""Apply WP3 (LDGSTS / staged-operand) H200 references to the H200 yaml."""
import hashlib

from nvidia_h200_yaml_edit import REFS, load, ref, save, set_component_throughput, set_key, set_service

H = 'microbench/nvidia_h200/references/'


def sha(name):
    return hashlib.sha256((REFS / name).read_bytes()).hexdigest()


def main():
    t = load()
    t = set_component_throughput(
        t, 'staged_ldgsts_l2_read',
        ref('staged_ldgsts_l2_service_sm90.json')['selected_service']['throughput_per_sm_clock'])
    t = set_component_throughput(
        t, 'staged_ldgsts_k32_depth6_l2_read',
        ref('staged_ldgsts_k32_depth6_l2_service_sm90.json')['selected_service']['throughput_per_sm_clock'])
    depth3 = ref('staged_ldgsts_depth3_curve_sm90.json')['selected_services']
    t = set_service(t, 'sm80_depth3_operand_l2_hit', depth3['l2_resident']['logical_byte_capacity_per_s'],
                    H + 'staged_ldgsts_depth3_curve_sm90.json#selected_services/l2_resident')
    t = set_service(t, 'sm80_depth3_operand_dram_miss', depth3['dram_unique']['logical_byte_capacity_per_s'],
                    H + 'staged_ldgsts_depth3_curve_sm90.json#selected_services/dram_unique')
    d6 = ref('staged_ldgsts_64x64_depth6_curve_sm90.json')['selected_services']
    t = set_service(t, 'sm80_64x64_depth6_operand_l2_hit', d6['l2_resident']['logical_byte_capacity_per_s'],
                    H + 'staged_ldgsts_64x64_depth6_curve_sm90.json#selected_services/l2_resident')
    t = set_service(t, 'sm80_64x64_depth6_operand_dram_miss', d6['dram_unique']['logical_byte_capacity_per_s'],
                    H + 'staged_ldgsts_64x64_depth6_curve_sm90.json#selected_services/dram_unique')
    t = set_service(t, 'sm80_fa2_paged_q64_kv128_period32_ldgsts',
                    ref('fa2_queued_ldgsts_period32_steady_frontend_sm90.json')['selected_service']['logical_byte_capacity_per_s'],
                    H + 'fa2_queued_ldgsts_period32_steady_frontend_sm90.json#selected_service')
    t = set_service(t, 'sm80_fa2_chunked_q8192_s4096_two_resident_ldgsts',
                    ref('fa2_chunked_q8192_two_resident_ingress_sm90.json')['selected_service']['logical_byte_capacity_per_s'],
                    H + 'fa2_chunked_q8192_two_resident_ingress_sm90.json#selected_service')
    initial = ref('ldgsts_initial_32b_transactions_sm90.json')['parameters']
    for key, value in initial.items():
        t = set_key(t, key, value)
    resident = ref('fa2_resident_ldgsts_finite_sm90.json')
    t = set_key(t, 'ldgsts_finite_resident_full_chip_ctas', resident['resource_contract']['full_chip_ctas'])
    t = set_key(t, 'ldgsts_finite_resident_single_panel_cycles', resident['single_panel_cycles'])
    t = set_key(t, 'ldgsts_finite_resident_affine_startup_cycles', resident['recurrent_affine_startup_cycles'])
    t = set_key(t, 'ldgsts_finite_resident_affine_panel_cycles', resident['recurrent_affine_panel_cycles'])
    t = set_key(t, 'ldgsts_finite_resident_reference_sha256', sha('fa2_resident_ldgsts_finite_sm90.json'))
    queued = ref('fa2_queued_ldgsts_steady_frontend_sm90.json')['selected_service']
    t = set_key(t, 'ldgsts_queued_steady_minimum_grid_ctas', queued['minimum_queued_grid_ctas'])
    t = set_key(t, 'ldgsts_queued_steady_full_chip_initial_32b_transactions_per_cycle',
                queued['full_chip_initial_32b_transactions_per_model_sm_cycle'])
    t = set_key(t, 'ldgsts_queued_steady_reference_sha256', sha('fa2_queued_ldgsts_steady_frontend_sm90.json'))
    save(t)


if __name__ == '__main__':
    main()
