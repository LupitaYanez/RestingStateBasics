# ============================================================
# Example usage:
# ============================================================
# python DMNandDAN_seed_connectivity.py \
#   --fmriprep-root /data/fmriprep_output \
#   --demo participants.tsv \
#   --id-col MRI_ID --dx-col diagnosis \
#   --groupA PD --groupB AD \
#   --atlas /data/atlases/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz \
#   --lut /data/atlases/Schaefer2018_200Parcels_7Networks_order.lut \
#   --outprefix results_PD_vs_AD \
#   --min-good-tr 100 --fd-mean-thr 0.5 --fd-prop05-thr 0.45
#
# This script computes seed-based connectivity for DMN and DAN
# networks, using fMRIPrep preprocessed BOLD data and the
# Schaefer 200-parcel, 7-network atlas.
#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import ttest_ind

# ---------------- Utils ----------------
def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def welch_df(a, b):
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    num = (va/na + vb/nb) ** 2
    den = (va**2)/((na**2)*(na-1)) + (vb**2)/((nb**2)*(nb-1))
    return num/den

def hedges_g(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / (na+nb-2))
    g = (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan
    J = 1 - 3/(4*(na+nb)-9)
    return g*J

def mean_diff_ci(a, b, alpha=0.05):
    # Welch CI for mean difference using normal approx with SE from Welch
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(va/na + vb/nb)
    if se == 0: 
        return (np.nan, np.nan)
    df = welch_df(a, b)
    # use t critical from scipy if available; fallback to normal ~1.96
    try:
        from scipy.stats import t
        tcrit = t.ppf(1 - alpha/2, df)
    except Exception:
        tcrit = 1.96
    diff = np.mean(a) - np.mean(b)
    return (diff - tcrit*se, diff + tcrit*se)

def seed_connectivity_z(seed, ts_mat, idx_rest):
    # seed: (T,), ts_mat: (T x P), idx_rest: indices (0-based)
    if len(idx_rest) == 0:
        return np.nan
    r = []
    for i in idx_rest:
        ri = np.corrcoef(seed, ts_mat[:, i])[0, 1]
        if np.isfinite(ri):
            r.append(ri)
    if len(r) == 0:
        return np.nan
    return np.nanmean(fisher_z(np.array(r)))

# --------------- Main ------------------
def main():
    ap = argparse.ArgumentParser(description="Seed-based within-network connectivity for DMN (PCC) and DAN (IPS/SPL).")
    ap.add_argument("--fmriprep-root", required=True, help="Root of fMRIPrep derivatives.")
    ap.add_argument("--demo", required=True, help="Participants table (tsv/csv/txt).")
    ap.add_argument("--id-col", default="mri_id", help="Column with participant IDs (e.g., sub-XXXX).")
    ap.add_argument("--dx-col", default="diagnosis", help="Column with diagnosis labels.")
    ap.add_argument("--groupA", required=True, help="Group A name pattern (e.g., MDD, PD).")
    ap.add_argument("--groupB", required=True, help="Group B name pattern (e.g., HC, AD).")
    ap.add_argument("--atlas", required=True, help="Schaefer200 7-networks labels NIfTI.")
    ap.add_argument("--lut", required=True, help="Schaefer LUT with columns: ID R G B Label.")
    ap.add_argument("--outprefix", required=True, help="Output prefix for CSVs and logs.")
    # QC thresholds (tus valores)
    ap.add_argument("--min-good-tr", type=int, default=100)
    ap.add_argument("--fd-mean-thr", type=float, default=0.5)
    ap.add_argument("--fd-prop05-thr", type=float, default=0.45)
    args = ap.parse_args()

    fmriprep_root = Path(args.fmriprep_root)

    # ------- Load LUT and define seeds/networks -------
    lut = pd.read_csv(args.lut, sep=r"\s+", header=None,
                      names=["ID", "R", "G", "B", "Label"], dtype={"Label": str})
    # 1-based IDs in LUT -> 0-based for arrays
    dmn_all = lut[lut["Label"].str.contains("Default", case=False)]["ID"].astype(int).to_numpy() - 1
    dmn_pcc = lut[lut["Label"].str.contains("Default_pCunPCC", case=False)]["ID"].astype(int).to_numpy() - 1

    dan_all = lut[lut["Label"].str.contains("DorsAttn", case=False)]["ID"].astype(int).to_numpy() - 1
    dan_post = lut[lut["Label"].str.contains("DorsAttn_Post", case=False)]["ID"].astype(int).to_numpy() - 1
    # opcional FEF si lo quisieras después:
    # dan_fef = lut[lut["Label"].str.contains("DorsAttn_FEF", case=False)]["ID"].astype(int).to_numpy() - 1

    if len(dmn_pcc) == 0 or len(dmn_all) == 0 or len(dan_post) == 0 or len(dan_all) == 0:
        print("ERROR: No se detectaron índices para alguna semilla o red. Revisa la LUT y etiquetas.")
        print("DMN_all:", dmn_all, "DMN_PCC:", dmn_pcc, "DAN_all:", dan_all, "DAN_Post:", dan_post)
        sys.exit(1)

    print("IDs detectados:")
    print("  DMN seed (PCC):", dmn_pcc)
    print("  DMN all:", dmn_all[:8], "...", dmn_all[-8:])
    print("  DAN seed (Post/IPS-SPL):", dan_post)
    print("  DAN all:", dan_all[:8], "...", dan_all[-8:])

    # ------- Load participants -------
    # detect delimiter automatically (csv/tsv/txt)
    try:
        demo = pd.read_csv(args.demo, sep=None, engine="python")
    except Exception:
        demo = pd.read_csv(args.demo, sep=r"\s+")
    demo.columns = demo.columns.str.strip()
    # normalize columns
    id_col = args.id_col
    dx_col = args.dx_col
    if id_col not in demo.columns or dx_col not in demo.columns:
        print("Columnas en demo:", demo.columns.tolist())
        raise ValueError(f"No encuentro columnas '{id_col}' y/o '{dx_col}' en el archivo demo.")
    demo["participant_id"] = demo[id_col].astype(str).str.strip()

    # ------- Iterate subjects -------
    records = []
    qc_rows = []
    n_excluded = 0

    for _, row in demo.iterrows():
        sub = row["participant_id"]
        dx  = str(row[dx_col])

        func_dir = fmriprep_root / sub / "func"
        if not func_dir.exists():
            qc_rows.append({"sub": sub, "reason": "no_func_dir"})
            continue

        bolds = sorted(func_dir.glob("*task-rest*desc-preproc_bold.nii.gz"))
        if not bolds:
            qc_rows.append({"sub": sub, "reason": "no_preproc_bold"})
            continue

        # confounds
        conf_files = sorted(func_dir.glob("*task-rest*desc-confounds_timeseries.tsv"))
        if not conf_files:
            qc_rows.append({"sub": sub, "reason": "no_confounds"})
            continue
        conf = conf_files[0]

        # QC from confounds
        conf_df = pd.read_csv(conf, sep="\t")
        if "framewise_displacement" not in conf_df.columns:
            qc_rows.append({"sub": sub, "reason": "no_FD_column"})
            continue

        fd = conf_df["framewise_displacement"].astype(float).replace([np.inf, -np.inf], np.nan)
        fd = fd.fillna(fd.median() if np.isfinite(fd.median()) else 0.0)

        mean_fd = float(np.mean(fd))
        prop_fd05 = float(np.mean(fd > 0.5))
        good_tr = int(np.sum(fd <= 0.5))

        if mean_fd > args.fd_mean_thr or prop_fd05 > args.fd_prop05_thr:
            qc_rows.append({"sub": sub, "reason": f"motion_excess: meanFD={mean_fd:.3f}, propFD>0.5={prop_fd05:.2f}"})
            n_excluded += 1
            continue
        if good_tr < args.min_good_tr:
            qc_rows.append({"sub": sub, "reason": f"low_good_TR: goodTR={good_tr}"})
            n_excluded += 1
            continue

        # Load image & TR
        img = nib.load(str(bolds[0]))
        tr = img.header.get_zooms()[3] if len(img.header.get_zooms()) > 3 else 2.0

        # Time series by parcels
        masker = NiftiLabelsMasker(labels_img=args.atlas,
                                   standardize=True, detrend=True,
                                   low_pass=0.1, high_pass=0.01, t_r=tr)
        # clean confounds for nilearn
        conf_np = conf_df.replace([np.inf, -np.inf], np.nan).fillna(0).values
        ts = masker.fit_transform(img, confounds=conf_np)  # (T x P)

        P = ts.shape[1]
        # safety: clip indices if atlas mismatch (shouldn't happen with Schaefer200)
        max_idx = P - 1
        dmn_all_clip = np.array([i for i in dmn_all if 0 <= i <= max_idx], dtype=int)
        dmn_pcc_clip = np.array([i for i in dmn_pcc if 0 <= i <= max_idx], dtype=int)
        dan_all_clip = np.array([i for i in dan_all if 0 <= i <= max_idx], dtype=int)
        dan_post_clip = np.array([i for i in dan_post if 0 <= i <= max_idx], dtype=int)

        if len(dmn_pcc_clip) == 0 or len(dan_post_clip) == 0:
            qc_rows.append({"sub": sub, "reason": "seed_indices_empty"})
            continue

        seed_dmn = ts[:, dmn_pcc_clip].mean(axis=1)  # PCC seed
        seed_dan = ts[:, dan_post_clip].mean(axis=1) # IPS/SPL seed

        rest_dmn = np.setdiff1d(dmn_all_clip, dmn_pcc_clip)
        rest_dan = np.setdiff1d(dan_all_clip, dan_post_clip)

        DMN_seedZ = seed_connectivity_z(seed_dmn, ts, rest_dmn)
        DAN_seedZ = seed_connectivity_z(seed_dan, ts, rest_dan)

        records.append({
            "sub": sub, "diagnosis": dx,
            "DMN_seedZ": DMN_seedZ, "DAN_seedZ": DAN_seedZ,
            "meanFD": mean_fd, "propFD_gt05": prop_fd05, "goodTR": good_tr
        })
        qc_rows.append({"sub": sub, "reason": "included"})

    # Save outputs
    res_df = pd.DataFrame(records)
    qc_df  = pd.DataFrame(qc_rows)
    res_path = f"{args.outprefix}_seed_connectivity.csv"
    qc_path  = f"{args.outprefix}_QC.csv"
    res_df.to_csv(res_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    print(f"Guardado: {res_path}")
    print(f"Guardado: {qc_path}")

    # Stats (Welch) for each metric
    if res_df.empty:
        print("Sin sujetos incluidos tras QC.")
        sys.exit(0)

    A_mask = res_df["diagnosis"].astype(str).str.upper().str.contains(args.groupA.upper(), na=False)
    B_mask = res_df["diagnosis"].astype(str).str.upper().str.contains(args.groupB.upper(), na=False)
    A = res_df[A_mask]
    B = res_df[B_mask]
    print(f"Tamaños tras QC: {args.groupA}={len(A)} | {args.groupB}={len(B)}")

    for metric in ["DMN_seedZ", "DAN_seedZ"]:
        a = A[metric].dropna().values
        b = B[metric].dropna().values
        if len(a) < 2 or len(b) < 2:
            print(f"{metric}: tamaños insuficientes para t-test.")
            continue
        t, p = ttest_ind(a, b, equal_var=False)
        dfw  = welch_df(a, b)
        g    = hedges_g(a, b)
        lo, hi = mean_diff_ci(a, b, alpha=0.05)
        print(f"{metric}: {args.groupA} mean={np.mean(a):.3f}±{np.std(a,ddof=1):.3f} "
              f"vs {args.groupB} mean={np.mean(b):.3f}±{np.std(b,ddof=1):.3f} "
              f"| t={t:.2f}, df={dfw:.1f}, p={p:.6g}, g={g:.2f}, 95%CI[{lo:.4f},{hi:.4f}]")

if __name__ == "__main__":
    main()
