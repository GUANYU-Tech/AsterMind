"""Auditable per-direction engineering budgets, not vendor performance claims."""

from __future__ import annotations

import math

C = 299792458.0
K = 1.380649e-23


def calculate_budget(profile, distance_km):
    """Return the per-direction net bit rate before schedules and shared limits."""
    if not math.isfinite(distance_km) or distance_km < 0:
        raise ValueError("distance_km must be finite and nonnegative")
    p = profile
    if p.capacity_model == 'islsim':
        from astermind.engine.setup.islsim import calculate
        return calculate(p, distance_km)
    result = dict(capacity_model=p.capacity_model, snr_db=None, path_loss_db=None,
                  terminal_limit_bps=p.bandwidth_bps, rate_scope="per_direction",
                  parameter_sources=p.parameter_sources, assumptions=p.assumptions)
    if p.capacity_model == "fixed":
        return {**result, "budget_bandwidth_bps": p.bandwidth_bps}
    if p.capacity_model != 'free_space_shannon':
        raise ValueError(f'Unsupported link budget model: {p.capacity_model}')
    wavelength = C / p.frequency_hz
    loss = 20 * math.log10(4 * math.pi * max(distance_km * 1000, 1e-6) / wavelength)
    def gain(aperture, efficiency, fallback):
        return (10 * math.log10(efficiency * (math.pi * aperture / wavelength) ** 2)
                if aperture is not None else fallback)
    tx = gain(p.transmit_aperture_m, p.transmit_aperture_efficiency, p.transmit_gain_db)
    rx = gain(p.receive_aperture_m, p.receive_aperture_efficiency, p.receive_gain_db)
    received_dbw = (10 * math.log10(p.transmit_power_w) + tx + rx - loss -
                    p.system_loss_db - p.atmospheric_loss_db - p.pointing_loss_db)
    noise = K * p.noise_temperature_k * p.channel_bandwidth_hz
    snr_db = received_dbw - 10 * math.log10(noise)
    x = snr_db * math.log(10) / 10
    efficiency = (max(x, 0) + math.log1p(math.exp(-abs(x)))) / math.log(2)
    rate = p.channel_bandwidth_hz * efficiency * p.implementation_efficiency
    result.update(received_power_w=10 ** (received_dbw / 10), noise_power_w=noise,
                  snr_db=snr_db, path_loss_db=loss, transmit_gain_db=tx, receive_gain_db=rx,
                  receiver_model="thermal_noise_shannon_upper_bound")
    return {**result, "budget_bandwidth_bps": min(p.bandwidth_bps, rate),
            "uncapped_bandwidth_bps": rate}
