"""Client pool API for Eiffel."""

import logging
import random
from typing import Callable, Optional

import keras
import tensorflow as tf
from ray import ObjectRef
from ray.actor import ActorHandle

from eiffel.core.errors import ConfigError
from eiffel.datasets.dataset import Dataset, DatasetHandle
from eiffel.datasets.partitioners import DumbPartitioner, Partitioner
from eiffel.datasets.poisoning import PoisonIns, PoisonTask
from eiffel.utils.time import timeit

from ..utils.typing import EiffelCID

logger = logging.getLogger(__name__)


class Pool:
    """Pool of clients.

    A pool is a collection of clients that share the same dataset and attack type.

    Attributes
    ----------
    pool_id : str
        The pool ID.
    attack : PoisonIns
        The attack to perform. If `None`, the pool is benign.
    shards : dict[EiffelCID, tuple[Dataset, Dataset]]
        The different client partitions. The keys are the client IDs, and the values
        are tuples of the training and test datasets.
    """

    # pool_id: str
    # attack: PoisonIns | None
    # shards: dict[EiffelCID, tuple[Dataset, Dataset]]
    # holders: dict[EiffelCID, ActorHandle]
    # model_fn: Callable[..., tf.keras.Model]

    @timeit
    def __init__(
        self,
        dataset: Dataset,
        model_fn: Callable[..., tf.keras.Model],
        n_benign: int,
        *,
        n_malicious: int = 0,
        malicious_client_ids: list[int] | None = None,
        attack: PoisonIns | dict | None = None,
        pool_id: str | None = None,
        test_ratio: float = 0.2,
        common_test: bool = True,
        partitioner: Partitioner | Callable | None = None,
        seed: int,
    ) -> None:
        """Initialize the pool.

        Parameters
        ----------
        dataset : Dataset | str
            The dataset used by the clients. If a DictConfig is provided, it should be a
            valid OmegaConf configuration that can be passed to Hydra's instantiation
            logic, and return a `Dataset` object. Otherwise, it should be a `Dataset`
            object.
        benign : int
            The number of benign clients in the pool.
        malicious : int, optional
            The number of malicious clients in the pool. Defaults to 0.
        attack : Optional[dict | PoisonIns], optional
            The attack to perform. If a dictionary is provided, it should be a valid
            dictionary
        """
        self.seed = seed
        self.model_fn = model_fn
        self.attack = attack
        self.holders = {}

        if not pool_id:
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
            pool_id = "".join(random.choices(alphabet, k=6))
        self.pool_id = pool_id

        requested_malicious_ids = [
            int(value) for value in (malicious_client_ids or [])
        ]
        if len(requested_malicious_ids) != len(set(requested_malicious_ids)):
            raise ConfigError("malicious_client_ids must be unique.")
        if requested_malicious_ids and len(requested_malicious_ids) != n_malicious:
            raise ConfigError(
                "len(malicious_client_ids) must match n_malicious: "
                f"{len(requested_malicious_ids)} != {n_malicious}."
            )
        if n_malicious == 0 and attack is not None:
            logger.warning(
                "Ignoring attack instructions: no malicious clients in the pool."
            )
        self.malicious_ids: set[EiffelCID] = set()

        if not isinstance(dataset, Dataset):
            raise TypeError(
                "Pool now requires a concrete Dataset object; Hydra call configs "
                f"are no longer supported (got {type(dataset)})."
            )

        # Synthetic/replay datasets can provide an explicit train/test split in
        # metadata. This preserves an exact federated training sample count and a
        # separate common test distribution instead of randomly re-splitting them.
        if "Split" in dataset.m.columns and set(dataset.m["Split"].unique()) >= {
            "train",
            "test",
        }:
            train_mask = dataset.m["Split"] == "train"
            test_mask = dataset.m["Split"] == "test"

            _train = dataset.copy()
            _train.X = dataset.X.loc[train_mask].copy()
            _train.y = dataset.y.loc[train_mask].copy()
            _train.m = dataset.m.loc[train_mask].copy()

            _test = dataset.copy()
            _test.X = dataset.X.loc[test_mask].copy()
            _test.y = dataset.y.loc[test_mask].copy()
            _test.m = dataset.m.loc[test_mask].copy()
        else:
            _test, _train = dataset.split(at=test_ratio, seed=self.seed)

        if not partitioner:
            partitioner = DumbPartitioner

        partitioner = partitioner(n_partitions=n_benign + n_malicious, seed=self.seed)

        partitioner.load(_train)
        _train_shards = partitioner.all()
        if common_test:
            _test_shards = [_test.copy() for _ in _train_shards]
        else:
            partitioner.load(_test)
            _test_shards = partitioner.all()

        # Attach a stable logical ID to every shard. For fixed preassigned
        # datasets this is the original ClientHint; otherwise partition position is
        # used. Explicit malicious IDs therefore refer to the real logical client
        # created during preprocessing, not merely to "the first N attackers".
        records: list[tuple[int, Dataset, Dataset]] = []
        for partition_idx, (train_shard, test_shard) in enumerate(
            zip(_train_shards, _test_shards)
        ):
            logical_id = partition_idx
            if "ClientHint" in train_shard.m.columns:
                hints = sorted(
                    int(value)
                    for value in train_shard.m["ClientHint"].unique()
                    if int(value) >= 0
                )
                if len(hints) == 1:
                    logical_id = hints[0]
                elif requested_malicious_ids:
                    raise ConfigError(
                        "Explicit malicious_client_ids require each partition to "
                        "contain exactly one non-negative ClientHint."
                    )
            records.append((logical_id, train_shard, test_shard))

        logical_ids = [record[0] for record in records]
        if len(logical_ids) != len(set(logical_ids)):
            if requested_malicious_ids:
                raise ConfigError(
                    "Explicit malicious_client_ids require unique logical client "
                    f"IDs, got {logical_ids}."
                )
            # Historical/non-preassigned partitioners can split a dataset whose
            # ClientHint metadata no longer identifies one logical client per shard.
            # In that case preserve backward compatibility by using partition order.
            records = [
                (partition_idx, train_shard, test_shard)
                for partition_idx, (_, train_shard, test_shard)
                in enumerate(records)
            ]
            logical_ids = [record[0] for record in records]

        if requested_malicious_ids:
            missing = sorted(set(requested_malicious_ids) - set(logical_ids))
            if missing:
                raise ConfigError(
                    "Requested malicious client IDs are absent from the partition: "
                    f"{missing}."
                )
            malicious_logical_ids = set(requested_malicious_ids)
        else:
            # Preserve historical Eiffel behaviour for fixed/preassigned datasets:
            # after benign shards were popped from the end, the lowest logical
            # partitions remained malicious.
            malicious_logical_ids = set(logical_ids[:n_malicious])

        if len(malicious_logical_ids) != n_malicious:
            raise ConfigError(
                "Resolved malicious client count does not match n_malicious."
            )

        if attack is not None:
            assert isinstance(attack, PoisonIns)
            p_task = attack.base
        else:
            p_task = None

        self.shards = {}
        for logical_id, train_shard, test_shard in records:
            malicious = logical_id in malicious_logical_ids
            role = "malicious" if malicious else "benign"
            cid = f"{pool_id}_{role}_{logical_id}"
            if malicious:
                self.malicious_ids.add(cid)
                # Model-poisoning clients need no data-poisoning instructions. If a
                # PoisonIns is present, only apply its non-zero base poisoning here;
                # scheduled tasks remain EiffelClient's responsibility.
                if p_task is not None and p_task.fraction > 0.0:
                    train_shard.poison(
                        p_task.fraction,
                        p_task.operation,
                        target_classes=attack.target,
                        source_class=attack.source_class,
                        destination_class=attack.destination_class,
                        seed=self.seed,
                    )
            self.shards[cid] = (train_shard, test_shard)

    def __len__(self) -> int:
        """Return the number of clients in the pool."""
        return len(self.shards)

    def __contains__(self, cid: EiffelCID) -> bool:
        """Return whether the pool contains the client."""
        return cid in self.shards

    @timeit
    def deploy(self) -> None:
        """Deploy the dataset onto the Ray object store."""
        if not self.holders:
            self.holders = {}
        for cid, (train, test) in self.shards.items():
            if cid in self.holders:
                raise ValueError(f"Client `{cid}` already deployed.")

            self.holders[cid] = DatasetHandle.remote({"train": train, "test": test})

    def deployed(self) -> bool:
        """Return whether the pool is deployed."""
        return len(self.holders) == len(self.shards)

    def gen_mappings(
        self,
    ) -> dict[
        EiffelCID,
        tuple[ObjectRef, PoisonIns | None, keras.Model, bool],
    ]:
        """Generate mappings between CIDs, and their handle and poisoning instructions.

        Returns
        -------
        dict[EiffelCID, tuple[ObjectRef, PoisonIns | None, keras.Model, bool]]
            The mappings. Each entry carries the dataset handle, optional data
            poisoning instructions, model factory, and explicit malicious role.
        """
        if not self.deployed():
            raise RuntimeError(
                "Attempting to access the handles of an undeployed pool."
            )

        mappings = {}
        for cid, handle in self.holders.items():
            is_malicious = cid in self.malicious_ids
            mappings[cid] = (
                handle,
                self.attack if is_malicious else None,
                self.model_fn,
                is_malicious,
            )
        return mappings

    @property
    def shards_stats(self) -> dict:
        """Return the pool data statistics."""
        return {
            cid: {"train": train.stats, "test": test.stats}
            for cid, (train, test) in self.shards.items()
        }

    @property
    def ids(self) -> list[EiffelCID]:
        """Return the list of client IDs."""
        return list(self.shards.keys())
