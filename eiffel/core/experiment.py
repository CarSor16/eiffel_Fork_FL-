"""Eiffel engine."""

import functools
import json
import logging
import math
from functools import partial, reduce
from typing import Callable

import numpy as np
import psutil
import ray
import tensorflow as tf
from flwr.common import ndarrays_to_parameters
from flwr.server import Server, ServerConfig
from flwr.server.strategy import FedAvg, Strategy
from flwr.simulation import start_simulation
from keras.models import Model

from eiffel.core.errors import ConfigError
from eiffel.datasets.dataset import Dataset
from eiffel.datasets.partitioners import DumbPartitioner, Partitioner
from eiffel.datasets.poisoning import PoisonIns
from eiffel.utils.time import timeit
from eiffel.utils.typing import ConfigDict, MetricsDict

from .client import mk_client, mk_client_init_fn
from .pool import Pool
from .results import Results

logger = logging.getLogger(__name__)

class Experiment:
    """Eiffel experiment.

    Attributes
    ----------
    server : flwr.server.Server
        The Flower server. It administrates the entire FL process, and is responsible
        for aggregating the models, based on the function provided in the
        `flwr.server.Strategy` object.
    strategy : flwr.server.Strategy
        The strategy used by the Flower server to aggregate the models. It is passed to
        Flower's `start_simulation` function.
    pools : list[Pool]
        The different client pools. Each pool is a collection of clients that share the
        same dataset and attack type.
    n_clients : int
        The total number of clients in the experiment.
    """

    n_clients: int
    n_rounds: int
    n_concurrent: int
    seed: int

    server: Server | None
    strategy: Strategy
    pools: list[Pool]

    @timeit
    def __init__(
        self,
        seed: int,
        num_rounds: int,
        num_epochs: int,
        batch_size: int,
        model_fn: Callable[..., tf.keras.Model],
        pools: list[Pool | dict],
        datasets: list[Dataset],
        attacks: list[PoisonIns | dict | None],
        strategy: partial[Strategy] | Strategy | None = None,
        server: Server | None = None,
        partitioner: Callable[..., Partitioner] | None = None,
        storage: dict | None = None,
        max_concurrent_clients: int | None = None,
    ):
        """Initialize one Flower experiment from concrete Python components.

        The direct TOML runtime resolves datasets, model factories, partitioners,
        attack instructions and strategies before constructing this object. A pool
        entry may be either an already-built :class:`Pool` or a plain dictionary
        containing the arguments required to build one.

        Parameters
        ----------
        seed : int
            Reproducibility seed.
        num_rounds : int
            Number of Flower communication rounds.
        num_epochs : int
            Local epochs per client and round.
        batch_size : int
            Local training batch size.
        model_fn : Callable
            Factory returning a compiled Keras model.
        pools : list[Pool | dict]
            Client pools or plain pool constructor arguments.
        datasets : list[Dataset]
            Concrete datasets, one per pool or one shared by all pools.
        attacks : list[PoisonIns | dict | None]
            Optional data-poisoning instructions.
        strategy : Strategy | partial[Strategy] | None
            Flower aggregation strategy. Defaults to FedAvg.
        partitioner : Callable[..., Partitioner] | None
            Partitioner factory. Defaults to DumbPartitioner.
        storage : dict | None
            Round-state capture options used by instrumented strategies.
        """
        self.seed = seed
        # set_seed(seed)

        self.server = server
        self.n_rounds = num_rounds
        self.pools = []

        pools = obj_to_list(pools)
        attacks = obj_to_list(attacks, expected_length=len(pools))
        datasets = obj_to_list(datasets, expected_length=len(pools))

        pools_mapping = zip(pools, attacks, datasets)

        for pool, attack, dataset in pools_mapping:
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    "datasets must contain concrete Dataset objects; "
                    f"got {type(dataset)}."
                )

            if attack is not None and not isinstance(attack, PoisonIns):
                if not isinstance(attack, dict):
                    raise TypeError(
                        "`attack` must be a PoisonIns, dictionary or None; "
                        f"got {type(attack)}."
                    )
                attack_config = dict(attack)
                attack_config.setdefault("n_rounds", num_rounds)
                attack = PoisonIns.from_dict(
                    attack_config, default_target=dataset.default_target
                )

            if isinstance(pool, dict):
                pool = Pool(
                    dataset=dataset,
                    model_fn=model_fn,
                    partitioner=partitioner or DumbPartitioner,
                    attack=attack,
                    seed=self.seed,
                    **{str(k): v for k, v in pool.items()},
                )
            elif not isinstance(pool, Pool):
                raise ConfigError(
                    f"Invalid pool type: {type(pool)}. Expected Pool or dict."
                )
            self.pools.append(pool)

        self.n_clients = sum([len(p) for p in self.pools])
        if max_concurrent_clients is None:
            self.n_concurrent = self.n_clients
        else:
            if int(max_concurrent_clients) < 1:
                raise ConfigError("max_concurrent_clients must be >= 1")
            self.n_concurrent = min(self.n_clients, int(max_concurrent_clients))

        if strategy is None:
            strategy = FedAvg()

        if isinstance(strategy, partial):
            strategy_name = getattr(strategy.func, "__name__", "")
            capture_inference = (
                strategy_name == "InstrumentedFedAvg"
                and storage is not None
                and bool(storage.get("enabled", True))
                and bool(storage.get("capture_inference", True))
            )
            probe_config = {
                "capture_inference": capture_inference,
                "capture_logits": bool(storage.get("capture_logits", True))
                if storage is not None else True,
                "capture_probe_features": bool(
                    storage.get("capture_probe_features", True)
                ) if storage is not None else True,
                "probe_size": int(storage.get("probe_size", 256))
                if storage is not None else 256,
            }
            self.strategy = strategy(
                min_fit_clients=self.n_clients,
                min_evaluate_clients=self.n_clients,
                min_available_clients=self.n_clients,
                on_fit_config_fn=mk_config_fn({
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    **probe_config,
                }),
                evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
                fit_metrics_aggregation_fn=aggregate_metrics_fn,
                on_evaluate_config_fn=mk_config_fn(
                    {"batch_size": batch_size, **probe_config},
                    stats_when=self.n_rounds,
                ),
                initial_parameters=get_random_weights(model_fn, datasets[0].X.shape[1]),
            )
        else:
            self.strategy = strategy

        assert isinstance(self.strategy, Strategy), (
            "Invalid strategy type: "
            f"{type(self.strategy)}. Expected a flwr.server.Strategy object."
        )

    def run(self, **ray_kwargs) -> None:
        """Run the experiment."""
        init_kwargs = (
            (ray_kwargs or {})
            | {
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_gpus": len(tf.config.list_physical_devices("GPU")),
            }
            # | {"local_mode": True}  # in debugger
            # if gettrace is not None and gettrace()
            # else {}
        )

        ray.init(**init_kwargs)
        logger.info(
            "Ray client concurrency: at most %s/%s clients at once.",
            self.n_concurrent,
            self.n_clients,
        )

        try:
            for pool in self.pools:
                pool.deploy()

            mappings = reduce(lambda a, b: a | b, [p.gen_mappings() for p in self.pools])

            fn = functools.partial(
                mk_client,
                mappings=mappings,
                seed=self.seed,
            )

            self.hist = start_simulation(
                client_fn=fn,
                num_clients=self.n_clients,
                config=ServerConfig(num_rounds=self.n_rounds),
                strategy=self.strategy,
                client_resources=compute_client_resources(self.n_concurrent),
                actor_kwargs={"on_actor_init_fn": mk_client_init_fn(seed=self.seed)},
                clients_ids=reduce(lambda a, b: a + b, [p.ids for p in self.pools]),
                server=self.server,
                keep_initialised=True,
            )

            if not self.hist.metrics_distributed_fit:
                raise RuntimeError(
                    "Flower completed without any distributed fit metrics. "
                    "This indicates that no client fit result reached aggregation; "
                    "inspect the client failure printed above."
                )
        finally:
            store = getattr(self.strategy, "store", None)
            if store is not None:
                try:
                    store.close()
                except Exception:
                    logger.exception("Failed to close round-state storage cleanly.")
            ray.shutdown()

    @property
    def results(self) -> Results:
        """Return the experiment's results."""
        return Results.from_flwr(self.hist)

    def data_stats(self) -> dict[str, dict[str, int]]:
        """Return the data statistics for each pool."""
        return {p.pool_id: p.shards_stats for p in self.pools}


def get_random_weights(
    model_fn: Callable[..., tf.keras.Model], n_features: int
) -> list[np.ndarray]:
    """Get deterministic initial weights from a model factory."""
    model: Model = model_fn(n_features)
    return ndarrays_to_parameters(model.get_weights())


def mk_config_fn(
    config: ConfigDict, stats_when: int = -1
) -> Callable[[int], ConfigDict]:
    """Return a function which creates a config for the given round.

    Optionally, the function can be configured to return a config with the `stats` flag
    enabled for a given round. This is useful to compute attack-wise statistics.

    Parameters
    ----------
    config : ConfigDict
        The configuration to return.
    stats_when : int, optional
        The round for which to enable the `stats` flag. Defaults to -1, which disables
        the flag entirely.
    """
    if stats_when > 0:

        def config_fn(r: int) -> ConfigDict:
            cfg = config | {"round": r}
            if r == stats_when:
                return cfg | {"stats": True}
            return cfg

        return config_fn

    return lambda r: config | {"round": r}


def compute_client_resources(
    n_concurrent: int, headroom: float = 0.1
) -> dict[str, float]:
    """Compute the number of CPUs and GPUs to allocate to each client.

    Parameters
    ----------
    n_concurrent : int
        The number of concurrent clients.
    headroom : float, optional
        The headroom to leave for the system. Defaults to 0.1.

    Returns
    -------
    dict[str, float]
        The number of CPUs and GPUs to allocate to each client.
    """
    available_cpus = psutil.cpu_count() * (1 - headroom)
    available_gpus = len(tf.config.list_physical_devices("GPU"))
    if n_concurrent > available_cpus:
        logger.warning(
            f"Number of concurrent clients ({n_concurrent}) is greater than the number"
            f" of available CPUs ({available_cpus}). Some clients will be run"
            " sequentially."
        )
    # Ray 2.6 requires CPU resource quantities greater than one to be whole
    # numbers. Use ceil(total_cpus / requested_concurrency): this prevents Ray from
    # scheduling more than the requested number of client actors without passing an
    # invalid fractional quantity such as 2.7 CPUs.
    total_cpus = max(1, int(psutil.cpu_count() or 1))
    num_cpus = (
        float(math.ceil(total_cpus / n_concurrent))
        if n_concurrent <= total_cpus
        else 1.0
    )
    return {
        "num_cpus": num_cpus,
        "num_gpus": available_gpus / min(n_concurrent, available_cpus),
    }


def obj_to_list(config_obj, expected_length: int = 0) -> list:
    """Normalize a scalar/list configuration value to a plain list."""
    if not isinstance(config_obj, (list, dict, type(None), Pool, Dataset, PoisonIns)):
        raise ConfigError(
            f"Invalid config object: {type(config_obj)}."
        )

    if not isinstance(config_obj, list):
        config_obj = [config_obj]

    if expected_length > 0:
        if len(config_obj) > 1 and len(config_obj) != expected_length:
            raise ConfigError(
                "The number of items in config_obj should be equal to"
                f" {expected_length}, or 1."
            )

        elif len(config_obj) == 1:
            config_obj = list(config_obj) * expected_length

    return list(config_obj)


def aggregate_metrics_fn(metrics_mapping: list[tuple[int, MetricsDict]]) -> MetricsDict:
    """Collect all metrics client-per-client.

    Eiffel processes metrics after the experiment's ending, which permits more versatile
    analytics. However, Flower expects a single metrics dictionary. This serializes each
    client's metrics into a single dictionary, indexed by the client's ID and containing
    collected metrtics (recall and missrate) for each attack class, as well as global
    metrics for the entire test set: accuracy, precision, recall, F1-score, missrate,
    and fallout.

    Parameters
    ----------
    metrics_mapping : list[tuple[int, MetricsDict]]
        A list of tuples containing the number of samples in the testing set and the
        collected metrics for each client.

    Returns
    -------
    MetricsDict
        A single dictionary containing all metrics.
    """
    round_metrics: MetricsDict = {}
    for _, m in metrics_mapping:
        met = {}
        for k, v in m.items():
            try:
                met[k] = json.loads(str(v))
            except json.JSONDecodeError:
                met[k] = v

        cid = str(met.pop("_cid"))
        round_metrics[cid] = json.dumps(met)

    return round_metrics
