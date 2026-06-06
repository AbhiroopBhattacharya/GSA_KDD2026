# Appendix: Hyperparameter Tuning

This section documents the hyperparameters used for unsupervised pretraining, supervised fine-tuning, and geodesic-weighted fine-tuning in the JEPA-based tabular pipeline. All values reflect the settings used in the main experiments unless noted otherwise.

---

## A.1 Unsupervised Pretraining (`train_unsupervised`)

Unsupervised training learns representations via JEPA and reconstruction objectives on tabular (and optionally image/text) inputs. The following settings were used:

| Parameter | Value | Description |
|-----------|--------|-------------|
| **Batch size** | 128 | Set in `main()`; used for all dataloaders. |
| **Epochs** | 25 | Overridden in `main()`; function default is 20. |
| **Learning rate** | \(1\times10^{-4}\) | Overridden in `main()`; function default is \(5\times10^{-5}\). |
| **Weight decay** | \(1\times10^{-4}\) | AdamW regularization. |
| **Warmup** | 5% of total steps | Linear warmup; after warmup the learning rate is held constant at the base value. |
| **Scheduler** | LambdaLR (warmup then constant) | No decay after warmup. |
| **Gradient clipping** | 1.0 (max norm) | Applied to all trainable parameters. |

Trainable parameters include the context encoder and the masked-column predictor; the target encoder is updated via exponential moving average (EMA) when applicable.

---

## A.2 Supervised Fine-Tuning (`fine_tune_supervised_property_predictor`)

Supervised fine-tuning trains a property prediction head on top of frozen (or partially frozen) representations. The head consumes context encoder outputs (e.g. attention-aggregated features) and predicts a single scalar property.

| Parameter | Value | Description |
|-----------|--------|-------------|
| **Epochs** | 10 | Function default. |
| **Learning rate** | \(1\times10^{-4}\) | AdamW. |
| **Weight decay** | \(1\times10^{-4}\) | In the standard supervised path (with weights: \(1\times10^{-4}\)). |
| **Scheduler** | StepLR | `step_size=10`, `gamma=0.1`. |
| **Loss** | MSE | `nn.MSELoss()`. |
| **Early stopping** | Patience 15 | Validation loss. |
| **Gradient clipping** | 1.0 (max norm) | Applied to the property predictor parameters. |

**Property predictor architecture (standard / in-domain):**  
`SupervisedPropertyPredictor` with `vit_hidden_dim=64` or `128`, `hidden_dim=64` or `128`, and `dropout=0.2`–\(0.4\) depending on the experiment (in-domain vs. OOD vs. geodesic).

---

## A.3 Geodesic-Weighted Fine-Tuning (`fine_tune_supervised_property_predictor_geodesic`)

Geodesic fine-tuning uses the same representation model as above but trains the property predictor with **sample-wise importance weights** derived from geodesic distances on the embedding manifold. Only the property predictor is trained; context and target encoders and the masked-column predictor are frozen.

| Parameter | Value | Description |
|-----------|--------|-------------|
| **Epochs** | 15 | Overridden in `main()`; function default is 10. |
| **Learning rate** | \(1\times10^{-4}\) | AdamW. |
| **Weight decay** | \(1\times10^{-2}\) | Stronger regularization than standard supervised. |
| **Scheduler** | StepLR | `step_size=10`, `gamma=0.1`. |
| **Loss** | MSE (reduction='none') | Per-sample losses; combined as weighted mean: \(\sum_i w_i \ell_i / (\sum_i w_i + \epsilon)\). |
| **Early stopping** | Patience 10 | Validation loss. |
| **Gradient clipping** | 1.0 (max norm) | Property predictor only. |

**Property predictor (geodesic branch):**  
`SupervisedPropertyPredictor` with `vit_hidden_dim=128`, `hidden_dim=64`, `dropout=0.2`.

---

## A.4 Geodesic Weight Computation

Weights for geodesic fine-tuning are computed from embeddings of the in-domain (ID) training set and the evaluation (OOD) set. The pipeline:

1. **Embeddings:**  
   Tabular embeddings are obtained from the frozen unsupervised model for the ID training set and the OOD set.

2. **Graph and anchors:**  
   - **`compute_split_metrics`** (OOD diagnostics): `A=20` (number of neighbors for graph construction), `k_graph=15`, `anchor_method="farthest"`, `knn_for_local=15`, `mutual=True`, `orc_mode="fast"`, `seed=1234`.  
   - **`precompute_id_geometry`** is called on the ID embeddings to build the graph and anchor set.

3. **Weights and trust region:**  
   **`calc_weights_geodesic_softanchors`** returns per-sample weights and geodesic distances. A **trust quantile** is applied: distances above the 90th percentile of finite distances are discarded (`d_max_trust = np.quantile(..., 0.90)`). Only samples with finite distance, distance \(\le d_{\max\text{trust}}\), and a valid anchor index are used when aggregating weights for the ID set. These per-ID weights are then used as `weights_by_id_dict` in the geodesic fine-tuning loss.

4. **Loss:**  
   Each training sample is weighted by its ID’s weight; the training objective is the weighted mean of per-sample MSE losses.

---

## A.5 Summary Table

| Stage | Epochs | LR | Weight decay | Scheduler | Loss | Notes |
|-------|--------|-----|--------------|-----------|------|--------|
| Unsupervised | 25 | \(1\times10^{-4}\) | \(1\times10^{-4}\) | 5% warmup, then constant | JEPA + reconstruction | Grad clip 1.0 |
| Supervised | 10 | \(1\times10^{-4}\) | \(1\times10^{-4}\) | StepLR 10, \(\gamma=0.1\) | MSE | Patience 15 |
| Geodesic | 15 | \(1\times10^{-4}\) | \(1\times10^{-2}\) | StepLR 10, \(\gamma=0.1\) | Weighted MSE | Patience 10; weights from ID geometry, 90% trust quantile |

These choices were kept fixed across the reported experiments; no separate hyperparameter search was performed for the appendix results.
