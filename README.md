# Self-Pruning Neural Network — Tredence AI Engineering Case Study

A feed-forward network that learns to prune itself **during training** on CIFAR-10, using learnable sigmoid gates and L1 sparsity regularization.

---

## Structure

```
├── tredence_case_study.py     # PrunableLinear + training loop
├── tredence_case_study.ipynb  # Notebook with full outputs
├── gate_distribution.png      # Gate value distribution plot
└── REPORT.md                  # Analysis and results
```

---

## Quickstart

```bash
pip install torch torchvision matplotlib numpy
python tredence_case_study.py
```

CIFAR-10 downloads automatically.

---

## Results Summary

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|:---:|:---:|:---:|
| 0.005 | **58.77** | 99.58 |
| 0.01 | 57.68 | 99.97 |
| 0.05 | 58.18 | 100.00 |

See [`REPORT.md`](REPORT.md) for full analysis.
