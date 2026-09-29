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

## Evaluation

Evaluation is performed at the patient level, with AUROC computed from raw logits.
The external test cohort is not used for training or model selection.
Classification thresholds are not optimized on the external test cohort.

## Citation

If you use this code, please cite the accompanying MCMINet manuscript.
