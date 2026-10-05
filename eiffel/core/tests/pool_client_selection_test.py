"""Regression tests for explicit logical malicious-client selection."""

from functools import partial

import pandas as pd

from eiffel.core.pool import Pool
from eiffel.datasets.partitioners import PreassignedPartitioner
from eiffel.datasets.preprocessed_network import PreprocessedNetworkDataset


def test_pool_selects_exact_preassigned_malicious_client_ids():
    train_rows = 10
    test_rows = 3
    dataset = PreprocessedNetworkDataset(
        X=pd.DataFrame(
            {"x": [float(i) for i in range(train_rows + test_rows)]}
        ),
        y=pd.Series([i % 2 for i in range(train_rows + test_rows)], dtype="int64"),
        m=pd.DataFrame(
            {
                "ClassName": [
                    "A" if i % 2 == 0 else "B"
                    for i in range(train_rows + test_rows)
                ],
                "ClassId": [i % 2 for i in range(train_rows + test_rows)],
                "Split": ["train"] * train_rows + ["test"] * test_rows,
                "ClientHint": list(range(10)) + [-1] * test_rows,
            }
        ),
        key="logical-clients",
        _default_target=["*"],
    )

    pool = Pool(
        dataset=dataset,
        model_fn=lambda _: None,
        n_benign=8,
        n_malicious=2,
        malicious_client_ids=[3, 7],
        partitioner=partial(
            PreassignedPartitioner,
            column="ClientHint",
            df_key="m",
        ),
        seed=2026,
    )

    malicious_hints = set()
    for cid, (train, _) in pool.shards.items():
        hints = set(int(v) for v in train.m["ClientHint"].unique())
        assert len(hints) == 1
        logical_id = next(iter(hints))
        if cid in pool.malicious_ids:
            malicious_hints.add(logical_id)

    assert malicious_hints == {3, 7}
