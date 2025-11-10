# RestingStateBasics

### Execution order
1. `01_fmriprep_SubjX.sh` → Run fMRIPrep preprocessing.
2. `02_mean_bold_by_network.py` → Extract mean activation per network (DMN, DAN).
3. `03_DMNandDAN_seed_connectivity.py` → Compute within-network seed-based connectivity.
4. `04_plot_seed_violins.py` → Generate violin plots for group comparisons.

Minimal and modular scripts for resting-state fMRI preprocessing, network activation mapping, seed-based connectivity, and group-level analysis with standardized violin-plot visualizations.

These scripts were designed for reproducible pipelines based on **fMRIPrep**, **Nilearn**, and the **Schaefer 7-network atlas**, and can be adapted to any BIDS-formatted dataset.

---

## Included Scripts

| Script | Description |
|--------|--------------|
| `fmriprep_SubjX.sh` | Minimal fMRIPrep preprocessing (anatomical + functional) with MNI alignment and basic confound regression. |
| `mean_bold_by_network.py` | Extracts mean activation (BOLD or PSC) within functional networks (DMN, DAN). |
| `DMNandDAN_seed_connectivity.py` | Computes Fisher-z–transformed seed-based correlations for DMN and DAN networks. |
| `plot_seed_violins.py` | Generates consistent, color-coded violin plots comparing groups (Welch’s *t*, Hedges’ *g*, and 95% CI). |

---

## Requirements

```bash
python >=3.8
pip install numpy pandas nibabel nilearn matplotlib scipy
