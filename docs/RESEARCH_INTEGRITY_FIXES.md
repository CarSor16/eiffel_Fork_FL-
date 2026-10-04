# Research integrity fixes (2026-10-04)

## Probe data and threat model

Fit-time probabilities used by `targeted_family_poisoning` now come exclusively
from the client's existing local training shard. No new split or preprocessing
is performed. Diagnostic fit evaluation still measures held-out test performance,
but cannot replace the attack probe. The strategy rejects targeted attacks with
test or unspecified probe provenance.

This changes the attack protocol relative to historical test-informed runs.
Historical and new results must be kept separate. The implemented attack still
uses server-side benign-update information: this is an oracle-assisted model
poisoning experiment with FedAvg, not proof of evading robust aggregation.

## Metrics

`macro_f1` averages over classes with positive support in the fixed test set.
`macro_f1_all_model_classes` averages over the fixed full model output domain;
absent-class zeros in that second convention are not empirical recall evidence.
`num_observed_test_classes`, `test_class_coverage`, and
`unobserved_test_class_ids` make coverage explicit. Confusion matrices count all
model outputs, including predictions of classes absent from test.

Binary evaluation now includes both class precision/recall/F1, full confusion
matrix, observed-class macro/minimum recall, and class coverage, while preserving
family detection recall and benign false positive rate. A binary model predicts
Attack, not a particular attack family: family precision is not inferred from it.

The per-class comparison pipeline now retains Benign precision/recall/F1;
previously it discarded that class when extracting per-family metrics.

## HDF5 and campaign checkpoints

HDF5 format version 3 separates the probe sources. `/fit_probe` contains local
training probe inputs, original labels when present,
and class metadata. `/probe` retains held-out evaluation probes. Client fit
probabilities are aligned with `/fit_probe/clients/<cid>`; global evaluation
inference is aligned with `/probe/clients/<cid>`. Legacy clients without source
metadata retain historical test-probe storage for non-targeted mechanisms.

Campaign manifests are written atomically before starting each subprocess and
after it returns. Each variant retains `campaign_profile.json`. An interruption
preserves completed rows and an `interrupted`/`running` marker for the unfinished
variant. This does not implement model-training resume from partial rounds.

## Runtime limits and planned research runs

TensorFlow 2.10 requires `/proc/self/exe` and Ray requires Linux `/proc` statistics.
The current Work sandbox does not provide them. Local code changes cannot repair
that runtime prerequisite; tensor and Flower smoke tests reproduce the block.
MIRAGE Parquet input and historical MIRAGE results are also absent.

The core plan retains 30 rounds, ten fixed logical clients and automatic safe
Ray concurrency. Use seeds 2026, 2027, 2028 with paired clean runs per task/seed,
then check convergence and replicate important findings with five seeds. Thirty
rounds are an initial screening budget, not a universal guarantee of convergence.
No efficacy claim is warranted before real runs complete. CICIoT missing test
classes and source overlap require source-data evidence before changing splits.
