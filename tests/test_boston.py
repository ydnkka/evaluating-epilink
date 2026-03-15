import pandas as pd

from evaluating_epilink.boston import summarise_cluster_sizes


def test_summarise_cluster_sizes_sorts_and_flags_focus_clusters() -> None:
    cluster_results = pd.DataFrame(
        {
            "cluster_id": [4, 2, 7],
            "size": [3, 10, 3],
        }
    )

    summary = summarise_cluster_sizes(cluster_results, {7})

    assert summary["cluster_id"].tolist() == [2, 4, 7]
    assert summary["rank"].tolist() == [1, 2, 3]
    assert summary["is_focus_cluster"].tolist() == [False, False, True]
