# Geometry-Aware Sample Reweighting (GSA)

> **Accepted at KDD 2026 (AI for Science Track)**
>
> This work was conducted during an internship at **A*STAR, Singapore**, under the supervision of **Dr. Hangwei Qian**.

## Overview

Geometry-Aware Sample Reweighting (GSA) is a training-time reweighting method for improving **out-of-distribution (OOD) generalization**.

Given a set of in-domain (ID) training samples and unlabeled OOD samples, GSA:

1. Learns a latent representation space.
2. Approximates the latent geometry using a graph.
3. Identifies regions of the ID manifold that are closest to OOD samples via geodesic distances.
4. Assigns higher weights to ID samples that are most relevant to the OOD regime.

Unlike density-ratio estimation or domain-adversarial approaches, GSA leverages the intrinsic geometry of the learned representation space to estimate sample importance.

<p align="center">
  <img src="figures/gsa_overview.png" width="700">
</p>

## Method

GSA consists of four steps:

### 1. Representation Learning
Obtain latent embeddings for ID and OOD samples using any representation learning model.

### 2. Geometry Construction
- Build a mutual k-NN graph over ID embeddings.
- Select anchor points on the ID manifold.

### 3. Anchor Scoring
- Assign OOD samples to nearby anchors using geodesic distances.
- Score anchors according to their proximity to OOD regions.

### 4. Sample Reweighting
- Assign each ID sample to its nearest anchor.
- Inherit the anchor score as the sample weight.

## Algorithm

```text
1. Build a geodesic graph on ID embeddings.
2. Assign each OOD sample to its nearest anchor.
3. Score anchors using OOD proximity.
4. Assign each ID sample to its nearest anchor.
5. Use anchor scores as sample weights.
```

## Repository Structure

```text
.
├── gsa.py                    # Core GSA implementation
├── graph_utils.py            # Graph construction and geodesic distances
├── weighting.py              # Sample weight computation
├── experiments/              # Experimental pipelines
├── datasets/                 # Dataset loaders
├── figures/
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from gsa import compute_gsa_weights

weights = compute_gsa_weights(
    id_embeddings,
    ood_embeddings,
    num_anchors=50,
    k_neighbors=10,
)
```

The resulting weights can be used with any supervised learning algorithm.

## Applications

GSA is model-agnostic and can be applied to:

- Out-of-distribution generalization
- Covariate shift correction
- Domain adaptation
- Scientific machine learning
- Materials property prediction
- Multimodal learning

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{bhattacharya2026gsa,
  title     = {Geometry-Aware Sample Reweighting for Out-of-Distribution Generalization},
  author    = {Bhattacharya, Abhiroop and Qian, Hangwei and Cloutier, Sylvain G and Tsang, Ivor},
  booktitle = {Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026}
}
```

## Acknowledgments

This work was completed during an internship at **A*STAR, Singapore**, under the guidance of **Dr. Hangwei Qian**.
