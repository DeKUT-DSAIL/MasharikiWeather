---
name: Feature Proposal
about: Propose a new spatial extraction method, ML feature, or data source.
title: "[EXP] "
labels: enhancement, experiment
assignees: 'John Doe'
---

## The Goal / Hypothesis
**What are you trying to achieve?**
*(e.g., "I want to test if using Conditional Merging on TAHMO and IMERG improves the Earth Mover's Distance loss during PyTorch training.")*

## Scientific & ML Justification
**Why is this mathematically or scientifically necessary?**
*(e.g., "Currently, we only use `patch_max`, but for hydrological mass balance, we need an areal average `patch_mean` to conserve water volume.")*

## Proposed Implementation
**How do you plan to build this into the data engine?**
*(e.g., "I will add a `patch_mean` condition to the `get_gridded_for_stations` method in `dataset.py` using `xarray.mean()`.")*

## Data Sources & Shapes
**Will this require new Hugging Face data or change the tensor shapes?**
- [ ] No new data required.
- [ ] New dataset required: `[List the dataset, e.g., TAMSAT]`
- [ ] Expected output tensor shape: `(Samples, Timesteps, Nodes, Features)`

## Definition of Done
**How do we know this experiment/feature is successful and ready to merge?**
- [ ] Code is written and tested.
- [ ] It runs without throwing Zarr/Fsspec chunking errors.
- [ ] Mathematical output is verified against a known truth.