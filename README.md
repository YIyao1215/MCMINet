# MCMINet

This repository provides the implementation of MCMINet for neoadjuvant therapy
(NAT) response prediction in hormone receptor-positive (HR⁺) breast cancer using
pretreatment dynamic contrast-enhanced MRI (DCE-MRI) and H&E-stained whole-slide
images (WSIs). The MRI branch encodes primary tumor, peritumoral region, and
axillary lymph node ROIs, while the WSI branch uses ResNet-18 patch features and
a graph attention network (GAT) for spatial modeling. Both modalities are mapped
to a shared 256-D representation space and combined by a multimodal classifier
for binary prediction. The model integrates intra-modal and cross-modal metric
learning with proxy-based representation learning.

## Installation

Python 3.10 or newer is required.

```bash
pip install -r requirements.txt
```

## Data preparation

Prepare a patient-level manifest with one row per patient containing:

- Pretreatment DCE-MRI series paths.
- Primary tumor and axillary lymph node mask paths.
- H&E-stained WSI and corresponding QuPath annotation paths.
- A binary response label.

Real patient data are not distributed with this repository.
Prepare the manifest using paths to your own data.

## Training

Training settings are defined in `configs/default.yaml`.

| Parameter | Setting |
|---|---|
| Optimizer | AdamW |
| Batch size | 8 |
| Maximum epochs | 100 |
| MRI pretrained backbone learning rate | 1e-5 |
| Newly initialized layers / GAT / classifier learning rate | 1e-4 |
| Proxy learning rate | 5e-5 |
| Weight decay | 1e-3; zero for biases, BatchNorm affine parameters, and proxies |

| Objective component | Temperature | Loss weight |
|---|---|---|
| Classification | — | 1.0 |
| Intra-modal | 0.10 | 0.20 |
| Cross-modal | 0.07 | 0.30 |
| Hybrid proxy | 0.07 | 0.30 |
| Modality-specific proxy | 0.10 | 0.20 |

The best checkpoint is selected by the highest internal-validation AUROC.

## Evaluation

Evaluation is performed at the patient level, with AUROC computed from raw logits.
The external test cohort is not used for training or model selection.
Classification thresholds are not optimized on the external test cohort.

## Citation

If you use this code, please cite the accompanying MCMINet manuscript.
