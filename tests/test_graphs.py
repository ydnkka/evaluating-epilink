from __future__ import annotations

import pandas as pd

from evaluating_epilink.graphs import build_weighted_graph, subset_pairs_for_nodes


def test_build_weighted_graph() -> None:
    pairwise_frame = pd.DataFrame(
        {
            "CaseA": ["A", "A"],
            "CaseB": ["B", "C"],
            "ProbLinearDist": [0.7, 0.4],
        }
    )

    graph = build_weighted_graph(
        pairwise_frame,
        weight_column="ProbLinearDist",
        minimum_weight=0.5,
        vertex_ids=["A", "B", "C"],
    )

    assert set(graph.vs["case_id"]) == {"A", "B", None}
    assert graph.ecount() == 1
    assert graph.es["ProbLinearDist"] == [0.7]


def test_subset_pairs_for_nodes_filters() -> None:
    pairwise_frame = pd.DataFrame(
        {
            "CaseA": ["A", "A", "B"],
            "CaseB": ["B", "C", "C"],
            "IsRelated": [True, False, True],
        }
    )

    subset = subset_pairs_for_nodes(pairwise_frame, {"A", "B"})

    assert subset[["CaseA", "CaseB"]].to_dict("records") == [{"CaseA": "A", "CaseB": "B"}]
