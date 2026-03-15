from __future__ import annotations

from evaluating_epilink.epilink_adapter import build_inference_kwargs


def test_build_inference_kwargs_accepts_new_names() -> None:
    config = {
        "inference": {
            "num_simulations": 123,
            "included_intermediate_counts": [0, 1],
            "max_intermediate_hosts": 7,
        }
    }

    kwargs = build_inference_kwargs(config)

    assert kwargs["num_simulations"] == 123
    assert kwargs["included_intermediate_counts"] == (0, 1)
    assert kwargs["max_intermediate_hosts"] == 7


def test_build_inference_kwargs_accepts_legacy_names() -> None:
    config = {
        "inference": {
            "num_simulations": 321,
            "intermediate_generations": [0],
            "intermediate_hosts": 4,
        }
    }

    kwargs = build_inference_kwargs(config)

    assert kwargs["num_simulations"] == 321
    assert kwargs["included_intermediate_counts"] == (0,)
    assert kwargs["max_intermediate_hosts"] == 4
