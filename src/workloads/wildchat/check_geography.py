#!/usr/bin/env python3
"""Regression checks for WildChat region fallback and nearest-gateway mapping."""

from __future__ import annotations

import json
from pathlib import Path

from wildchat_adapter import find_region, load_country_centroids, load_state_centroids, select_gateway


BASE = Path(__file__).resolve().parent


def main() -> int:
    config = json.loads((BASE / "wildchat_experiment_config_2024-11-11.json").read_text(encoding="utf-8"))
    states = load_state_centroids(config, BASE)
    countries = load_country_centroids(config, BASE)
    cases = [
        ("China", "Guangdong", "china_guangdong", "gw_hong_kong"),
        ("China", "Hebei", "cn_hebei", "gw_beijing"),
        ("United States", "California", "us_california", "gw_los_angeles"),
        ("Canada", "Ontario", "canada_ontario", "gw_toronto"),
        ("Australia", "Victoria", "australia_victoria", "gw_sydney"),
        ("Brazil", "Sao Paulo", "brazil_sao_paulo", "gw_sao_paulo"),
    ]
    for country, state, expected_region, expected_gateway in cases:
        region = find_region({"country": country, "state": state}, config["region_mapping"], states, countries)
        assert region is not None, (country, state)
        assert region["region_id"] == expected_region, (country, state, region)
        gateway = select_gateway(region, config["gateways"]["items"], config["gateways"]["selection_policy"])
        assert gateway["gateway_id"] == expected_gateway, (country, state, gateway)
        print(f"PASS {country}|{state} -> {region['region_id']} -> {gateway['gateway_id']}")

    country_fallback = find_region({"country": "France", "state": None}, config["region_mapping"], states, countries)
    assert country_fallback is not None and country_fallback["region_id"] == "country_france"
    print("PASS France|<none> -> country_france (country fallback)")
    assert len(states) == 162
    assert len(countries) == 193
    gateway_ids = [gateway["gateway_id"] for gateway in config["gateways"]["items"]]
    assert len(gateway_ids) == 26 and len(set(gateway_ids)) == 26
    print(f"PASS catalogs: {len(states)} state entries, {len(countries)} countries, {len(gateway_ids)} gateways")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
