"""Thin adapter between manuscript workflows and the epilink package API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from numpy.random import Generator, default_rng

from epilink import (
    InfectiousnessToTransmissionTime,
    MolecularClock,
    NaturalHistoryParameters,
    SymptomOnsetToTransmissionTime,
    estimate_genetic_linkage_probability,
    estimate_linkage_probability,
    estimate_presymptomatic_transmission_fraction,
    estimate_temporal_linkage_probability,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

from .config import config_value


@dataclass(frozen=True)
class ModelComponents:
    """Container for the core epilink model objects used in workflows."""

    transmission_profile: InfectiousnessToTransmissionTime
    molecular_clock: MolecularClock
    rng: Generator


def _section(config: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    return dict(current) if isinstance(current, Mapping) else {}


def build_natural_history_parameters(config: Mapping[str, Any]) -> NaturalHistoryParameters:
    """Build NaturalHistoryParameters from either the new or legacy config shape."""

    natural_history_config = _section(config, "model", "natural_history")
    return NaturalHistoryParameters(
        incubation_shape=float(natural_history_config.get("incubation_shape", 5.807)),
        incubation_scale=float(natural_history_config.get("incubation_scale", 0.948)),
        latent_shape=float(natural_history_config.get("latent_shape", 3.38)),
        symptomatic_rate=float(natural_history_config.get("symptomatic_rate", 0.37)),
        symptomatic_shape=float(natural_history_config.get("symptomatic_shape", 1.0)),
        rel_presymptomatic_infectiousness=float(
            natural_history_config.get("rel_presymptomatic_infectiousness", 2.29)
        ),
    )


def build_molecular_clock(
    config: Mapping[str, Any],
    *,
    rng: Generator | None = None,
    rng_seed: int | None = None,
) -> MolecularClock:
    """Build MolecularClock from either the new or legacy config shape."""

    clock_config = _section(config, "model", "molecular_clock")
    return MolecularClock(
        substitution_rate=float(clock_config.get("substitution_rate", clock_config.get("subs_rate", 1e-3))),
        use_relaxed_clock=bool(clock_config.get("use_relaxed_clock", clock_config.get("relax_rate", True))),
        relaxed_clock_sigma=float(
            clock_config.get("relaxed_clock_sigma", clock_config.get("subs_rate_sigma", 0.33))
        ),
        genome_length=int(clock_config.get("genome_length", clock_config.get("gen_len", 29903))),
        rng=rng,
        rng_seed=rng_seed,
    )


def build_transmission_profile(
    config: Mapping[str, Any],
    *,
    rng: Generator | None = None,
    rng_seed: int | None = None,
) -> InfectiousnessToTransmissionTime:
    """Build the infectiousness-to-transmission profile used across analyses."""

    return InfectiousnessToTransmissionTime(
        parameters=build_natural_history_parameters(config),
        rng=rng,
        rng_seed=rng_seed,
    )


def build_symptom_onset_profile(
    config: Mapping[str, Any],
    *,
    rng: Generator | None = None,
    rng_seed: int | None = None,
) -> SymptomOnsetToTransmissionTime:
    """Build the symptom-onset-to-transmission profile used in characterisation."""

    return SymptomOnsetToTransmissionTime(
        parameters=build_natural_history_parameters(config),
        rng=rng,
        rng_seed=rng_seed,
    )


def build_model_components(config: Mapping[str, Any]) -> ModelComponents:
    """Construct the epilink model objects from a merged workflow config."""

    rng_seed = int(config_value(config, ["project", "rng_seed"]))
    rng = default_rng(rng_seed)
    transmission_profile = build_transmission_profile(config, rng=rng)
    molecular_clock = build_molecular_clock(config, rng=rng)
    return ModelComponents(
        transmission_profile=transmission_profile,
        molecular_clock=molecular_clock,
        rng=rng,
    )


def build_inference_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise inference settings to the current epilink estimator names."""

    inference_config = _section(config, "inference")
    included_intermediate_counts = inference_config.get("included_intermediate_counts", [0])
    if isinstance(included_intermediate_counts, int):
        included_intermediate_counts = [included_intermediate_counts]
    return {
        "num_simulations": int(inference_config.get("num_simulations", 10_000)),
        "included_intermediate_counts": tuple(int(value) for value in included_intermediate_counts),
        "max_intermediate_hosts": int(inference_config.get("max_intermediate_hosts", 10)),
    }


def build_surveillance_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise surveillance settings for epidemic date simulation."""

    surveillance_config = _section(config, "surveillance")
    delay_config = _section(surveillance_config, "sampling_delay")
    if not surveillance_config:
        return {
            "prop_sampled": float(config.get("prop_sampled", 1.0)),
            "sampling_shape": float(config.get("sampling_shape", 3.0)),
            "sampling_scale": float(config.get("sampling_scale", 1.0)),
            "root_start_range": int(config.get("root_start_range", 30)),
        }
    return {
        "prop_sampled": float(surveillance_config.get("sampled_fraction", 1.0)),
        "sampling_shape": float(delay_config.get("shape", 3.0)),
        "sampling_scale": float(delay_config.get("scale", 1.0)),
        "root_start_range": int(surveillance_config.get("root_start_range", 30)),
    }


def simulate_epidemic_tree(
    transmission_profile: InfectiousnessToTransmissionTime,
    tree,
    config: Mapping[str, Any],
):
    """Populate a transmission tree with epidemic dates using merged config settings."""

    return simulate_epidemic_dates(
        transmission_profile=transmission_profile,
        tree=tree,
        **build_surveillance_kwargs(config),
    )


def simulate_genomic_outputs(molecular_clock: MolecularClock, tree) -> dict[str, Any]:
    """Generate simulated genomic outputs using the current epilink package API."""

    return simulate_genomic_sequences(clock=molecular_clock, tree=tree)


def estimate_linkage_scores(
    transmission_profile: InfectiousnessToTransmissionTime,
    molecular_clock: MolecularClock,
    *,
    genetic_distance,
    temporal_distance,
    config: Mapping[str, Any],
):
    """Estimate pairwise linkage probabilities using the merged config settings."""

    return estimate_linkage_probability(
        transmission_profile=transmission_profile,
        clock=molecular_clock,
        genetic_distance=genetic_distance,
        temporal_distance=temporal_distance,
        **build_inference_kwargs(config),
    )


def estimate_temporal_scores(
    transmission_profile: InfectiousnessToTransmissionTime,
    *,
    temporal_distance,
    config: Mapping[str, Any],
):
    """Estimate the temporal-only component for one or more time gaps."""

    inference_kwargs = build_inference_kwargs(config)
    return estimate_temporal_linkage_probability(
        temporal_distance=temporal_distance,
        transmission_profile=transmission_profile,
        num_simulations=inference_kwargs["num_simulations"],
    )


def estimate_genetic_scores(
    transmission_profile: InfectiousnessToTransmissionTime,
    molecular_clock: MolecularClock,
    *,
    genetic_distance,
    config: Mapping[str, Any],
    output_mode: str = "normalized",
    include_all_intermediate_counts: bool = False,
):
    """Estimate the genetic-only component for one or more SNP distances."""

    inference_kwargs = build_inference_kwargs(config)
    normalized_output_mode = "normalized" if output_mode == "relative" else output_mode
    included_intermediate_counts = (
        None if include_all_intermediate_counts else inference_kwargs["included_intermediate_counts"]
    )
    return estimate_genetic_linkage_probability(
        transmission_profile=transmission_profile,
        clock=molecular_clock,
        genetic_distance=genetic_distance,
        num_simulations=inference_kwargs["num_simulations"],
        max_intermediate_hosts=inference_kwargs["max_intermediate_hosts"],
        included_intermediate_counts=included_intermediate_counts,
        output_mode=normalized_output_mode,
    )


def estimate_presymptomatic_fraction(config: Mapping[str, Any]) -> float:
    """Return the presymptomatic transmission fraction implied by the config."""

    return float(estimate_presymptomatic_transmission_fraction(build_natural_history_parameters(config)))
