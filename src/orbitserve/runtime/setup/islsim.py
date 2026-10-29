"""Read-only adapter to an external, unmodified ISLSim checkout.

The upstream modules use top-level names (config/link_model/validator). Load them
in a private namespace so unrelated applications keep their own module names.
No upstream formula or result is rewritten by this adapter.
"""
from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import fields
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
from types import ModuleType

_LOCK = threading.RLock()
_FILES = ('config.py', 'link_model.py', 'validator.py')
_PROJECT = Path(__file__).resolve().parents[4]


class ISLSimUnavailable(RuntimeError):
    pass


def _path(value):
    value = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[/\\]', value):
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    return Path(value).expanduser().resolve()


def checkout():
    settings = _PROJECT / 'local/config/communication-modules.json'
    return _checkout(os.environ.get('ORBITSERVE_ISLSIM_ROOT'),
                     settings.stat().st_mtime_ns if settings.is_file() else None)


@lru_cache(maxsize=8)
def _checkout(environment, modified):
    settings = _PROJECT / 'local/config/communication-modules.json'
    local = json.loads(settings.read_text(encoding='utf-8')).get('islsim', {}) if modified is not None else {}
    location = environment or local.get('root') or _PROJECT.parent / 'ISLSim'
    location = _path(location)
    if any(not (location / name).is_file() for name in _FILES):
        raise ISLSimUnavailable('ISLSim checkout unavailable; set ORBITSERVE_ISLSIM_ROOT '
                               'or explicitly select another capacity_model')
    return location


def source_identity(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _FILES}


@lru_cache(maxsize=8)
def _modules(location, identity):
    root = Path(location)
    sources = {name[:-3]: (root / name).read_bytes() for name in _FILES}
    if {name + '.py': hashlib.sha256(data).hexdigest() for name, data in sources.items()} != dict(identity):
        raise ISLSimUnavailable('ISLSim source changed before loading the frozen model')
    prefix = '_orbitserve_islsim_' + hashlib.sha256((location + repr(identity)).encode()).hexdigest()[:16]
    modules = {}
    original_import = builtins.__import__

    def upstream_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in {'config', 'link_model', 'validator'}:
            return load(name)
        return original_import(name, globals, locals, fromlist, level)

    def load(name):
        if name not in modules:
            spec = importlib.util.spec_from_file_location(prefix + '.' + name, root / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            module.__dict__['__builtins__'] = {**vars(builtins), '__import__': upstream_import}
            modules[name] = module
            sys.modules[spec.name] = module
            try:
                # Execute the original source without writing __pycache__ into
                # the independently maintained upstream checkout.
                filename = root / (name + '.py')
                exec(compile(sources[name], str(filename), 'exec'), module.__dict__)
            except BaseException:
                modules.pop(name, None)
                sys.modules.pop(spec.name, None)
                raise
        return modules[name]

    with _LOCK:
        package = ModuleType(prefix)
        package.__path__ = [location]
        sys.modules[prefix] = package
        return load('config'), load('link_model'), load('validator')


def _api(expected=None):
    root = checkout()
    identity = source_identity(root)
    if expected and identity != expected:
        raise ISLSimUnavailable('ISLSim source differs from the physical input; regenerate physics with the intended version')
    return root, identity, _modules(str(root), tuple(sorted(identity.items())))


def resolve_config(parameters=None, expected=None):
    root, identity, (config, _, validator) = _api(expected)
    sections = {'tx', 'rx', 'channel'}
    if parameters is not None and (not isinstance(parameters, dict)
            or not set(parameters) <= sections
            or any(not isinstance(section, dict) for section in parameters.values())):
        raise ValueError('islsim_config accepts only tx, rx and channel objects')
    # Frozen configurations are self-contained. Partial scenario overrides use
    # the selected native JSON as their base, without changing that file.
    complete = parameters and set(parameters) == sections and all(
        {f.name for f in fields(cls)} <= set(parameters[name])
        for name, cls in (('tx', config.TXConfig), ('rx', config.RXConfig), ('channel', config.ChannelConfig)))
    if complete:
        values = parameters
    else:
        filename = os.environ.get('ORBITSERVE_ISLSIM_CONFIG')
        settings = _PROJECT / 'local/config/communication-modules.json'
        if not filename and not os.environ.get('ORBITSERVE_ISLSIM_ROOT') and settings.is_file():
            filename = json.loads(settings.read_text(encoding='utf-8')).get('islsim', {}).get('config')
        base = config.load_config_from_json(str(_path(filename) if filename else root / 'configs/google_constellation.json'))
        values = config.config_to_dict(base)
        for section, overrides in (parameters or {}).items():
            values[section].update(overrides)
    cfg = config.FSOConfig(tx=config.TXConfig(**values['tx']),
                          rx=config.RXConfig(**values['rx']),
                          channel=config.ChannelConfig(**values['channel']))
    validator.validate_config(cfg)
    # These guards belong to OrbitServe's JSON/input contract, not its physics.
    values = json.loads(json.dumps(config.config_to_dict(cfg), allow_nan=False))
    if type(cfg.dual_polarization) is not bool or cfg.wdm_power_mode not in {'total_power_fixed', 'per_channel_power_fixed'}:
        raise ValueError('invalid ISLSim polarization or WDM power mode')
    return values, identity


@lru_cache(maxsize=128)
def _configuration(location, identity, serialized):
    config, _, validator = _modules(location, identity)
    data = json.loads(serialized)
    cfg = config.FSOConfig(tx=config.TXConfig(**data['tx']), rx=config.RXConfig(**data['rx']),
                           channel=config.ChannelConfig(**data['channel']))
    validator.validate_config(cfg)
    return cfg


def calculate(profile, distance_km):
    if not math.isfinite(distance_km) or distance_km <= 0:
        raise ValueError('ISLSim requires a finite positive link distance')
    # Source identity is verified when the immutable LinkProfile is constructed.
    # Hot per-sample calls reuse that binding instead of rereading source files.
    root = Path(profile._islsim_checkout)
    identity = profile.islsim_source_sha256
    _, model, _ = _modules(str(root), tuple(sorted(identity.items())))
    cfg = _configuration(str(root), tuple(sorted(identity.items())),
                         json.dumps(profile.islsim_config, sort_keys=True, allow_nan=False))
    value = model.compute_link(cfg, distance_km)
    rate = value.total_bw_gbps * 1e9
    if not math.isfinite(rate) or rate < 0 or type(value.feasible) is not bool:
        raise ValueError('ISLSim returned an invalid link result')
    return dict(capacity_model='islsim', rate_scope='per_direction',
                terminal_limit_bps=profile.bandwidth_bps,
                budget_bandwidth_bps=min(profile.bandwidth_bps, rate) if value.feasible else 0.,
                uncapped_bandwidth_bps=value.raw_total_bw_gbps * 1e9,
                received_power_w=1e-3 * 10 ** (value.received_power_dbm / 10),
                snr_db=value.snr_db, path_loss_db=value.path_loss_db,
                modulation_order=value.modulation_order, modulation_label=value.modulation_label,
                link_feasible=value.feasible, limiting_reason=value.limiting_reason,
                islsim_total_bw_gbps=value.total_bw_gbps,
                islsim_per_channel_bw_gbps=value.per_channel_bw_gbps,
                islsim_latency_ms=value.latency_ms,
                islsim_source_sha256=identity,
                parameter_sources=profile.parameter_sources, assumptions=profile.assumptions,
                receiver_model='external_islsim_unmodified')


def freeze_profiles(network):
    """Default user-facing ISLs to ISLSim; preserve explicitly selected models."""
    value = deepcopy(network)
    profiles = value.get('link_profiles', {})
    gsl = {'sgl'}
    for profile in profiles.values():
        if isinstance(profile, dict):
            gsl.update(p for p in (profile.get('uplink_profile'), profile.get('downlink_profile')) if p)
    for node in value.get('nodes', ()):
        gsl.update((node.get('sgl_profiles', {}) if isinstance(node, dict)
                    else getattr(node, 'sgl_profiles', {})).values())
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue  # Explicit low-level LinkProfile objects retain their selected model.
        profile.setdefault('capacity_model', 'fixed' if name in gsl else 'islsim')
        if profile['capacity_model'] == 'islsim':
            profile['islsim_config'], profile['islsim_source_sha256'] = resolve_config(
                profile.get('islsim_config'), profile.get('islsim_source_sha256'))
    return value
