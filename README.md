# RP-Fuse

[![Paper DOI](https://img.shields.io/badge/DOI-10.3390%2Fapp162110497-blue)](https://doi.org/10.3390/app162110497)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23377120.svg)](https://doi.org/10.5281/zenodo.23377120)

Official code for the paper

> **RP-Fuse: Development of an Adaptive Dose-Gated Multimodal Model for
> Predicting Radiation Pneumonitis in Non-Small Cell Lung Cancer**
> C. B. Gemirter, E. K. Kıraklı, C. C. Özdilli, A. K. Çırak,
> E. E. Korkmaz, D. Goularas.
> *Applied Sciences* 2026, 16(21), 10497.
> [https://doi.org/10.3390/app162110497](https://doi.org/10.3390/app162110497)

**If you use this code, you must cite the paper** (see [Citation](#citation)).

RP-Fuse predicts grade 2 or higher radiation pneumonitis within six months of
thoracic radiotherapy from five modalities: the clinical record, planning CT,
planning PET, the 3D dose map and the 3D contours.

## Method

**Adaptive dose gate.** A frozen 3D ResNet-18 (MedicalNet) encodes the planning
CT and the dose map. A learned monotone dose-response curve, a mixture of four
sigmoids with 12 parameters, turns the dose of every healthy-lung voxel into a
weight, and the encoder feature maps are pooled with these weights. See
`gate.py` and `train_gate_ct.py`.

**PET-E.** Ten PET descriptors built only from intra-patient uptake ratios, so
they do not depend on scanner calibration.

**Competence-weighted fusion.** Each stream has its own expert trained
separately. A single self-attention layer assigns per-patient weights to the
expert logits, using out-of-fold expert predictions only. See
`train_rpfuse.py`.

Three configurations are reported:

| configuration | streams |
|---|---|
| RP-Fuse *Scalar* | clinical record, lung dosimetrics, PET-E |
| RP-Fuse *Gated* | clinical record, PET-E, adaptive dose gate |
| RP-Fuse *Ensemble* | mean of the Scalar and Gated probabilities |

## Results

5 x 5 nested cross-validation on 87 patients (30 events), pooled out-of-fold.

| model | AUC [95% CI] | PR-AUC | Brier skill |
|---|---|---|---|
| CT + dose, uniform pooling | 0.550 [0.43-0.67] | 0.380 | -0.038 |
| CT + dose, adaptive dose gate | 0.704 [0.59-0.82] | 0.506 | +0.060 |
| RP-Fuse *Scalar* | 0.781 [0.68-0.87] | 0.672 | +0.204 |
| RP-Fuse *Gated* | 0.781 [0.68-0.88] | 0.619 | +0.179 |
| RP-Fuse *Ensemble* | 0.822 [0.74-0.91] | 0.688 | +0.218 |

## Setup

    python3.12 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt

Download the MedicalNet `resnet_18_23dataset.pth` weights into
`medicalnet_weights/`, and set `RAW_DIR`, `INPUTS_DIR` and `OUTPUTS_DIR` in
`configs/config.yaml`.

## Data

The patient imaging and clinical data cannot be shared because of
institutional and privacy constraints. The anonymized feature matrices used to
train the models are available from the corresponding author on reasonable
request.

`preprocess_pipeline.py` expects the raw export under `RAW_DIR` (DICOM CT and
PET, Monaco dose and contour files, the clinical spreadsheet) and writes one
folder per patient under `INPUTS_DIR/Cohort`, together with
`master_clinical_data.xlsx` and the fold assignment in `unified_split`.
Derived feature tables go to `INPUTS_DIR/Derivated_Modals`, results to
`OUTPUTS_DIR/results_*`.

## Configs

Every script reads the config named by the `RPFUSE_CONFIG` environment
variable (default `config.yaml`). A config can name a parent under `EXTENDS`
and override only the keys it changes.

| file | purpose |
|---|---|
| `config.yaml` | base settings |
| `config_gate_ct.yaml` | adaptive dose gate on planning CT and dose map |
| `config_rpfuse_scalar.yaml` | RP-Fuse *Scalar* |
| `config_rpfuse_gated.yaml` | RP-Fuse *Gated* |

## Reproducing the paper

Preprocessing and derived features:

    python preprocess_pipeline.py
    python generate_dvh_curve.py
    python generate_dosimetrics.py
    python pet_radiomics.py
    python ct_radiomics_gated.py
    python generate_radiomics_tables.py

Single-modality benchmarks (Table 4):

    python train_tabular_compare.py
    python tools/petomics_row.py
    python train_ct.py
    python train_pet.py
    python train_dose.py

Adaptive dose gate (Table 5). The run exports the embeddings that the fusion
reads:

    RPFUSE_CONFIG=config_gate_ct.yaml python train_gate_ct.py

Fusion (Table 6):

    RPFUSE_CONFIG=config_rpfuse_scalar.yaml python train_rpfuse.py
    RPFUSE_CONFIG=config_rpfuse_gated.yaml python train_rpfuse.py
    python tools/ensemble.py

Reimplemented reference methods (Table 7):

    python train_sota.py
    python tools/lkb_baseline.py
    python tools/sota_table.py

Calibration and decision-curve analysis:

    python posthoc_analysis.py --results_dir <results dir>

## Repository layout

| path | content |
|---|---|
| `preprocess_pipeline.py` | registration, rasterization, DVH, cohort and split |
| `generate_*.py`, `pet_radiomics.py`, `ct_radiomics_gated.py` | derived feature tables |
| `gate.py`, `train_gate_ct.py`, `export_gated_ct.py` | adaptive dose gate and its qualitative export |
| `train_rpfuse.py` | stream experts and fusion |
| `train_tabular_compare.py`, `train_ct.py`, `train_pet.py`, `train_dose.py` | single-modality benchmarks |
| `train_sota.py`, `train_sota_cnn.py` | reference methods |
| `nested_cv.py`, `endpoint.py`, `util.py` | cross-validation, endpoint definition, config loading |
| `posthoc_analysis.py`, `tools/` | tables, calibration, decision curves, data checks |

## License

This code is released under the
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license:
free for academic and other non-commercial use with attribution. See
[LICENSE](LICENSE) for details.

## Citation

**Any use of this code requires citing the paper below.** Citation metadata is
also provided in [CITATION.cff](CITATION.cff), and the repository itself is
archived on Zenodo with the DOI
[10.5281/zenodo.23377120](https://doi.org/10.5281/zenodo.23377120).

    @article{gemirter2026rpfuse,
      title     = {RP-Fuse: Development of an Adaptive Dose-Gated Multimodal Model
                   for Predicting Radiation Pneumonitis in Non-Small Cell Lung Cancer},
      author    = {Gemirter, Cavide Balk{\i} and K{\i}rakl{\i}, Esra Korkmaz and
                   {\"O}zdilli, Cemre Ceren and {\c{C}}{\i}rak, Ali Kadri and
                   Korkmaz, Emin Erkan and Goularas, Dionysis},
      journal   = {Applied Sciences},
      volume    = {16},
      number    = {21},
      pages     = {10497},
      year      = {2026},
      publisher = {MDPI},
      doi       = {10.3390/app162110497},
      url       = {https://doi.org/10.3390/app162110497}
    }
