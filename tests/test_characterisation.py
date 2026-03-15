from evaluating_epilink.characterisation import (
    build_surface_specs,
    format_intermediate_label,
    override_intermediate_counts,
)


def test_override_intermediate_counts_does_not_mutate_original_config() -> None:
    config = {"inference": {"included_intermediate_counts": [0], "num_simulations": 1000}}

    updated = override_intermediate_counts(config, (0, 1, 2))

    assert config["inference"]["included_intermediate_counts"] == [0]
    assert updated["inference"]["included_intermediate_counts"] == [0, 1, 2]


def test_format_intermediate_label_prefers_readable_manuscript_labels() -> None:
    assert format_intermediate_label((0,)) == "Direct only (M=0)"
    assert format_intermediate_label((0, 1, 2)) == "Allow up to 2 intermediates (M<=2)"
    assert format_intermediate_label((0, 2)) == "M in {0, 2}"


def test_build_surface_specs_avoids_duplicate_default_variant() -> None:
    specs = build_surface_specs((0,), (0,))

    assert len(specs) == 1
    assert specs[0].scenario == "default"
