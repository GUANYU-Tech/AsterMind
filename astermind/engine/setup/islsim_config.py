"""Portable ISLSim input contract; generation never imports the external model."""
from copy import deepcopy
import math


# OrbitServe's input defaults are explicit, versioned scenario assumptions.
# These are configuration values, not a copy of ISLSim's implementation.
DEFAULT_CONFIG = {
    'tx': {'tx_power_w': 5., 'tx_aperture_m': .1, 'wavelength_m': 1.55e-6,
           'baud_rate_ghz': 60., 'wdm_channels': 2, 'fec_overhead': .15,
           'dual_polarization': True, 'wdm_power_mode': 'total_power_fixed',
           'modulation_pool': [[16, 'DP-16QAM'], [8, 'DP-8QAM'], [4, 'DP-QPSK']]},
    'rx': {'rx_aperture_m': .1, 'preamp_gain_db': 40., 'preamp_nf_db': 7.5,
           'ber_threshold': .001, 'terminal_cap_gbps': 400.},
    'channel': {'link_margin_db': 7., 'optical_efficiency': .7},
}


def prepare_config(parameters=None):
    """Merge partial scenario inputs and validate the JSON sent to ISLSim.

    No checkout, model calculation, environment setting or local config is read.
    External source identities are attached separately when actually computing.
    """
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict) or not set(parameters) <= DEFAULT_CONFIG.keys():
        raise ValueError('islsim_config accepts only tx, rx and channel objects')
    result = deepcopy(DEFAULT_CONFIG)
    for section, values in parameters.items():
        if not isinstance(values, dict) or not set(values) <= DEFAULT_CONFIG[section].keys():
            raise ValueError(f'invalid ISLSim {section} fields')
        result[section].update(deepcopy(values))

    def number(section, key, *, lower=0., upper=math.inf, lower_open=False, upper_open=False):
        value = result[section][key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < lower or value > upper
                or lower_open and value == lower or upper_open and value == upper):
            raise ValueError(f'invalid ISLSim {section}.{key}')

    for section, keys in (('tx', ('tx_power_w', 'tx_aperture_m', 'baud_rate_ghz')),
                           ('rx', ('rx_aperture_m',))):
        for key in keys:
            number(section, key, lower_open=True)
    number('tx', 'wavelength_m', lower=.4e-6, upper=10e-6)
    number('tx', 'fec_overhead', upper=1., upper_open=True)
    number('rx', 'preamp_gain_db')
    number('rx', 'preamp_nf_db', lower=1.)
    number('rx', 'ber_threshold', upper=1., lower_open=True, upper_open=True)
    if result['rx']['terminal_cap_gbps'] is not None:
        number('rx', 'terminal_cap_gbps', lower_open=True)
    number('channel', 'link_margin_db')
    number('channel', 'optical_efficiency', upper=1., lower_open=True)
    tx = result['tx']
    if type(tx['wdm_channels']) is not int or tx['wdm_channels'] < 1:
        raise ValueError('ISLSim tx.wdm_channels must be a positive integer')
    if type(tx['dual_polarization']) is not bool:
        raise ValueError('ISLSim tx.dual_polarization must be boolean')
    if tx['wdm_power_mode'] not in ('total_power_fixed', 'per_channel_power_fixed'):
        raise ValueError('invalid ISLSim tx.wdm_power_mode')
    pool = tx['modulation_pool']
    if not isinstance(pool, (list, tuple)) or not pool:
        raise ValueError('ISLSim modulation_pool must contain candidates')
    for item in pool:
        if (not isinstance(item, (list, tuple)) or len(item) != 2 or type(item[0]) is not int
                or item[0] < 4 or item[0] & (item[0] - 1) or not isinstance(item[1], str) or not item[1]):
            raise ValueError('ISLSim modulation_pool requires power-of-two orders >= 4 and labels')
    tx['modulation_pool'] = [list(item) for item in pool]
    return result
