# Dataset Manifest

The repository keeps the current preprocessing statistics; manuscript tables must be synchronized to newly verified outputs rather than legacy paper counts.

| Dataset | Source | Current configured preprocessing |
|---|---|---|
| MovieLens | GroupLens ML-1M | rating ≥4, iterative user/item filtering, chronological leave-one-out |
| Epinions | MSU Epinions rating/timestamp archive | iterative user/item filtering, chronological leave-one-out when timestamps parse |
| Gowalla | loc-gowalla check-ins | iterative user/item filtering, chronological leave-one-out |
| Demo | `data/demo/interactions.csv` | bundled synthetic smoke-test data |

Run `scripts/verify_dataset.py --dataset NAME` to print source checksum, users, items, interactions, split counts, sparsity, and split hash. Raw-to-remapped IDs can be persisted with `Dataset.save_id_mappings()`.

Expected historical workspace counts (must be re-verified after the corrected deduplication and timestamp policy):

- MovieLens: 6,034 users, 3,125 items, 574,376 interactions.
- Epinions: 20,382 users, 30,989 items, 542,856 interactions.
- Gowalla: 29,495 users, 40,358 items, 2,001,700 interactions.

No result should claim these counts unless `verify_dataset.py` reproduces them for the exact source checksum.
