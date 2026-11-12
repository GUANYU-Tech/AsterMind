"""Runnable assumptions alongside, never inside, published hardware capacities."""
from copy import deepcopy

REPORT24 = 'https://starlink.com/public-files/starlinkProgressReport_2024.pdf'
REPORT25 = 'https://starlink.com/public-files/starlinkProgressReport_2025.pdf'
V3_SPEC = 'https://starlink.com/en-ps/updates/starlink-version-3-satellites'
FCC = 'https://docs.fcc.gov/public/attachments/FCC-22-91A1.pdf'
FCC_E = 'https://docs.fcc.gov/public/attachments/DOC-404553A1.pdf'


def simulation_defaults(generation, *, report_2025=False):
    """Single-carrier RF budgets, plus satellite aggregate provisioning.

    Aggregate user capacity is NOT assigned to each GSL. The selected carrier
    widths, power, effective apertures and losses are editable assumptions.
    V3 aggregate improvement does not imply every legacy user terminal changes.
    """
    v2 = generation == 2
    uplink = 160e9 / 24 if v2 else 200e9 if report_2025 else 160e9
    downlink = 96e9 if v2 else 1e12
    optical = 100e9 if v2 else 200e9
    assumptions = [
        '网关方向以地面为参考：网关→星的配置容量按用户总下行预留，星→网关按用户总上行预留；这是流量匹配的工程配置，不是公开回传硬件指标。',
    ]
    if v2:
        assumptions.append('V2 Mini 上行估算为 160/24 ≈ 6.667 Gbps，依据 2024 报告 V3 上行 160 Gbps、约 24 倍对比；倍数有取整，非厂商直接公布的 V2 测量值。')
    if report_2025:
        assumptions.append('2025 报告的 >1000 / >200 Gbps 仅以 1000 / 200 Gbps 下界作可运行假设，不能解释为精确上限；与 V3 专门规格页 1000 / 160 版本分开。')
    capacities = dict(feeder_uplink_bps=downlink, feeder_downlink_bps=uplink)
    if v2:
        capacities.update(user_uplink_bps=uplink, feeder_combined_bps=downlink+uplink)
    common = dict(capacity_model='free_space_shannon', setup_time_s=2., max_distance_km=2500.,
                  noise_temperature_k=290., system_loss_db=2., atmospheric_loss_db=1.,
                  implementation_efficiency=.7, sun_exclusion_angle_deg=3.,
                  parameter_sources=[FCC, REPORT24, V3_SPEC],
                  assumptions=[
                      '代表性单载波预算，不把整星容量或用户测速中位数当成单条链路速率。',
                      '中心频率取已用频段内的代表点；载波宽度、发射功率、有效孔径、损耗、效率、温度和建链时间均为可编辑工程假设，不是恢复的 Starlink 专有参数。',
                      'V2/V3 使用相同参考用户终端；代际总容量主要反映波束/资源并行度，未假定单用户速率等比增长。',
                  ])

    def rf(frequency, width, power, tx, rx, ceiling, note):
        return dict(deepcopy(common), frequency_hz=frequency, channel_bandwidth_hz=width,
                    transmit_power_w=power, transmit_aperture_m=tx, receive_aperture_m=rx,
                    bandwidth_bps=ceiling, assumptions=[*common['assumptions'], note])

    profiles = {
        'user_uplink': rf(14.25e9, 62.5e6, 2., .5, .3, uplink,
                          '用户→星：62.5 MHz 和 2 W 是参考终端假设；不是用下行预算复制得到。'),
        'user_downlink': rf(12e9, 240e6, 10., .3, .5, downlink,
                            '星→用户：240 MHz 代表性载波；单链路仍受该方向整星共享池约束。'),
        'feeder_uplink': rf(83.5e9, 2e9, 10., 1.85, .3, downlink,
                            '网关→星：代表性 E-band 载波；1.85 m 地面天线参照 FCC 2024 部分站点许可，不代表所有网关；2 GHz 为场景分配。'),
        'feeder_downlink': rf(73.5e9, .5e9, 10., .3, 1.85, uplink,
                              '星→网关：1.85 m 地面天线参照 FCC 2024 部分站点；500 MHz 为场景分配；不能根据用户下载/上传比断言回传硬件比例。'),
        'communication_isl': dict(capacity_model='islsim', bandwidth_bps=optical,
            islsim_config={'rx': {'terminal_cap_gbps': optical / 1e9}},
            setup_time_s=3., max_distance_km=6000., wavelength_m=1550e-9,
            transmit_power_w=.2, transmit_aperture_m=.05, receive_aperture_m=.05,
            transmit_aperture_efficiency=.6, receive_aperture_efficiency=.6,
            required_photons_per_bit=20., pointing_loss_db=1., system_loss_db=2.,
            implementation_efficiency=.8, parameter_sources=[REPORT24 if v2 else V3_SPEC],
            assumptions=[
                '公开光口标称 200 / 400 Gbps 的单双向口径未明确，默认按双向合计保守解释，每方向 100 / 200 Gbps；可在场景中修改。',
                '每方向光口工程上限同时传入 ISLSim terminal_cap_gbps；功率、口径、波特率、调制池等未公开项继承选定的 ISLSim 配置，可在本场景覆盖，不代表 Starlink 实测参数。',
                '默认使用外部 ISLSim 的 tx/rx/channel 配置计算速率；3 s 建链与 6000 km 限距为 OrbitServe 场景假设。',
                '光子预算参数仅供显式选择 optical_photon_budget 时使用；ISLSim 使用 islsim_config。'])
    }
    for kind in ('feeder_uplink', 'feeder_downlink'):
        profiles[kind]['parameter_sources'].append(FCC_E)
    return dict(radio_capacity_bps=capacities, capacity_assumptions=assumptions,
                optical_terminal_bandwidth_bps=optical,
                terminal_assumptions=[
                    '光口标称速率的单双向口径未经厂商确认，默认每方向取标称一半。',
                    '星地并发 peer 数默认取 RF 网关天线数，属于一副天线一个 peer 的场景假设，不等于用户波束数。'],
                hardware_link_profiles=profiles)
