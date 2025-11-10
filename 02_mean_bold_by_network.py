# ============================================================
# Example usage:
# ============================================================
# python mean_bold_by_network.py \
#   --fmriprep-root /data/fmriprep_output \
#   --demo participants.tsv \
#   --id-col MRI_ID --dx-col diagnosis \
#   --groupA PD --groupB AD \
#   --atlas /data/atlases/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz \
#   --lut /data/atlases/Schaefer2018_200Parcels_7Networks_order.lut \
#   --outprefix meanBOLD_PD_vs_AD \
#   --min-good-tr 100 --fd-mean-thr 0.5 --fd-prop05-thr 0.45
#
# Optionally include '--psc' to compute percent signal change
# relative to the global run mean.
#
# This script estimates mean activation per network (DMN/DAN)
# and performs group comparisons using Welch's t-tests.
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import ttest_ind, t
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# Paleta consistente global
# =========================
COLORS = {
    "PD":  "#5DADE2",  # azul
    "AD":  "#E67E22",  # naranja
    "MD":  "#AF7AC5",  # violeta
    "HC":  "#58D68D"   # verde
}
DEFAULT_A = "#5DADE2"
DEFAULT_B = "#E67E22"

# ============
# Utilidades
# ============
def scrub_mask(conf_df, fd_thr=0.5):
    """Máscara de frames 'buenos' con scrubbing ±1 TR alrededor de FD>fd_thr."""
    if "framewise_displacement" not in conf_df.columns:
        return np.ones(len(conf_df), dtype=bool)
    fd = pd.to_numeric(conf_df["framewise_displacement"], errors="coerce").fillna(0).to_numpy()
    bad = fd > fd_thr
    bad_prev = np.r_[False, bad[:-1]]
    bad_next = np.r_[bad[1:], False]
    keep = ~(bad | bad_prev | bad_next)
    return keep

def pooled_sd(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    return np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx + ny - 2))

def hedges_g_from_samples(x, y):
    """Hedges' g con corrección J."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    s_p = pooled_sd(x, y)
    if not np.isfinite(s_p) or s_p == 0: return np.nan
    d = (x.mean() - y.mean()) / s_p
    J = 1 - (3 / (4*(nx + ny) - 9))
    return d * J

def welch_df_from_samples(x, y):
    """df de Welch–Satterthwaite a partir de las muestras."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    num = (vx/nx + vy/ny)**2
    den = ((vx/nx)**2)/(nx-1) + ((vy/ny)**2)/(ny-1)
    return num / den

def ci95_welch_from_samples(x, y):
    """IC95% para la diferencia de medias (x - y) con Welch."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return (np.nan, np.nan, np.nan)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx/nx + vy/ny)
    if not np.isfinite(se) or se == 0: return (np.nan, np.nan, np.nan)
    df = welch_df_from_samples(x, y)
    q = t.ppf(0.975, df)
    diff = mx - my
    return diff - q*se, diff + q*se, df

def violin_two_groups(a_vals, b_vals, labelA, labelB, title, outfile,
                      ylabel="Mean (network)", seed=123):
    """Violín + puntos + boxplot con colores consistentes, formato editable (SVG/PDF/PNG)."""
    a = np.array(a_vals, float); a = a[np.isfinite(a)]
    b = np.array(b_vals, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        print(f"[SKIP FIG] {title}: uno de los grupos está vacío (sizes: {a.size}, {b.size}). No se genera figura.")
        return

    colorA = COLORS.get(labelA, DEFAULT_A)
    colorB = COLORS.get(labelB, DEFAULT_B)

    fig, ax = plt.subplots(figsize=(5.2, 5.8))
    vp = ax.violinplot([a, b], showmeans=False, showextrema=False, widths=0.9)
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor([colorA, colorB][i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.35)

    # puntos jitter
    rng = np.random.default_rng(seed)
    jitterA = 1 + (rng.random(len(a)) - 0.5)*0.15
    jitterB = 2 + (rng.random(len(b)) - 0.5)*0.15
    ax.scatter(jitterA, a, color=colorA, alpha=0.7, s=20, linewidths=0.3, edgecolors='black')
    ax.scatter(jitterB, b, color=colorB, alpha=0.7, s=20, linewidths=0.3, edgecolors='black')

    # boxplots encima
    ax.boxplot([a, b], positions=[1,2], widths=0.15, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor='none', color='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))

    ax.set_xticks([1,2]); ax.set_xticklabels([labelA, labelB], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    base = outfile.rsplit(".", 1)[0]
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base}.svg", format="svg", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figuras: {base}.png/.svg/.pdf")

# ========================
# Argumentos de ejecución
# ========================
ap = argparse.ArgumentParser(
    description="Mean BOLD por red (DMN/DAN) con Schaefer200/7N; QC + estadísticas + violines (colores consistentes)."
)
ap.add_argument("--fmriprep-root", required=True)
ap.add_argument("--demo", required=True, help=".csv o .xlsx con IDs y diagnóstico")
ap.add_argument("--id-col", required=True)
ap.add_argument("--dx-col", required=True)
ap.add_argument("--groupA", required=True, help="regex (p.ej. MD o PD)")
ap.add_argument("--groupB", required=True, help="regex (p.ej. HC o AD)")
ap.add_argument("--atlas", required=True, help="Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
ap.add_argument("--lut", required=True, help=".lut correspondiente")
ap.add_argument("--outprefix", default="MEANBOLD")
# QC (ajustables)
ap.add_argument("--min-good-tr", type=int, default=100)
ap.add_argument("--fd-mean-thr", type=float, default=0.5)
ap.add_argument("--fd-prop05-thr", type=float, default=0.45)
# Extracción
ap.add_argument("--detrend", action="store_true", help="detrend temporal")
ap.add_argument("--psc", action="store_true", help="Percent Signal Change respecto a baseline GLOBAL del run")
args = ap.parse_args()

root = Path(args.fmriprep_root)

# =========================
# LUT → índices (0-based)
# =========================
lut = pd.read_csv(args.lut, sep=r"\s+", header=None,
                  names=["ID","R","G","B","Label"], dtype={"Label":str})
dmn_ids = lut[lut.Label.str.contains("Default")]["ID"].astype(int).to_numpy() - 1
dan_ids = lut[lut.Label.str.contains("DorsAttn")]["ID"].astype(int).to_numpy() - 1
if len(dmn_ids)==0 or len(dan_ids)==0:
    raise SystemExit("No se detectaron DMN/DAN en la LUT. Revisa archivos/regex.")
print(f"DMN parcels: {len(dmn_ids)} | DAN parcels: {len(dan_ids)}")

# =================
# Participantes
# =================
if args.demo.lower().endswith(".xlsx"):
    demo = pd.read_excel(args.demo)
else:
    demo = pd.read_csv(args.demo, sep=None, engine="python")
demo.columns = demo.columns.str.strip().str.lower()
idcol = args.id_col.strip().lower()
dxcol = args.dx_col.strip().lower()
if idcol not in demo.columns or dxcol not in demo.columns:
    raise SystemExit(f"[ERROR] demo debe tener columnas '{idcol}' y '{dxcol}'. Tiene: {demo.columns.tolist()}")

demo["participant_id"] = demo[idcol].astype(str).str.strip()
demo["diagnosis"]     = demo[dxcol].astype(str).str.strip()

# ===========================
# Loop por sujetos + medidas
# ===========================
rows = []
for _, r in demo.iterrows():
    sub = r["participant_id"]; dx = r["diagnosis"]
    func_dir = root / sub / "func"
    if not func_dir.exists():
        print(f"[WARN] sin func/: {sub}"); continue
    bolds = sorted(func_dir.glob("*task-rest*desc-preproc_bold.nii.gz"))
    confs = sorted(func_dir.glob("*task-rest*desc-confounds_timeseries.tsv"))
    if not bolds or not confs:
        print(f"[WARN] sin BOLD/confounds: {sub}"); continue

    bold, conf = bolds[0], confs[0]
    conf_df = pd.read_csv(conf, sep="\t").replace([np.inf,-np.inf], np.nan).fillna(0)

    # --- QC movimiento ---
    if "framewise_displacement" in conf_df.columns:
        fd = pd.to_numeric(conf_df["framewise_displacement"], errors="coerce").fillna(0)
        mean_fd = float(fd.mean())
        prop_fd05 = float((fd > 0.5).mean())
    else:
        mean_fd, prop_fd05 = np.nan, 0.0

    if (np.isfinite(mean_fd) and mean_fd > args.fd_mean_thr) or (np.isfinite(prop_fd05) and prop_fd05 > args.fd_prop05_thr):
        print(f"[EXCLUDE] {sub} por movimiento: meanFD={mean_fd:.3f}, propFD>0.5={prop_fd05:.2f}")
        continue

    keep = scrub_mask(conf_df, fd_thr=0.5)
    if keep.sum() < args.min_good_tr:
        print(f"[EXCLUDE] {sub}: TR buenos={int(keep.sum())} < {args.min_good_tr}")
        continue

    # confounds básicos (opcionales)
    base_cols = []
    for c in conf_df.columns:
        if "trans" in c or "rot" in c or "motion" in c or c in ["white_matter","csf","global_signal"]:
            base_cols.append(c)
    conf_use = conf_df[base_cols] if base_cols else None

    img = nib.load(str(bold))
    tr = img.header.get_zooms()[3]

    masker = NiftiLabelsMasker(
        labels_img=str(args.atlas),
        standardize=False,          # no z-score parcel; PSC usa baseline global
        detrend=args.detrend,
        low_pass=None, high_pass=None,
        t_r=tr
    )

    ts_all = masker.fit_transform(str(bold), confounds=None if conf_use is None else conf_use.values)
    ts = ts_all[keep, :]  # aplicar scrubbing

    # --- PSC opcional usando baseline GLOBAL del run ---
    if args.psc:
        m_global = float(np.nanmean(ts))
        if not np.isfinite(m_global) or m_global == 0:
            ts = np.full_like(ts, np.nan, dtype=float)
        else:
            ts = 100.0 * (ts - m_global) / m_global

    # --- activación media por red ---
    mean_bold_dmn = float(np.nanmean(ts[:, dmn_ids])) if len(dmn_ids) else np.nan
    mean_bold_dan = float(np.nanmean(ts[:, dan_ids])) if len(dan_ids) else np.nan

    rows.append(dict(
        sub=sub, diagnosis=dx,
        mean_DMN=mean_bold_dmn,
        mean_DAN=mean_bold_dan
    ))

# ================
# Guardar CSV
# ================
out_csv = f"{args.outprefix}_meanBOLD_perNetwork.csv"
df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print(f"\nGuardado: {out_csv}")
print(df.head())

# ==============================
# Estadística entre grupos
# ==============================
def sel(regex):
    return df[df["diagnosis"].str.upper().str.contains(regex.upper(), na=False)]

A = sel(args.groupA); B = sel(args.groupB)
print(f"\nTamaños tras QC: {args.groupA}={len(A)} | {args.groupB}={len(B)}")

def report(col, label):
    x, y = A[col].values, B[col].values
    # Welch t-test (Scipy)
    tstat, pval = ttest_ind(x, y, equal_var=False, nan_policy="omit")
    # df y CI95 de Welch a partir de muestras
    lo, hi, df_w = ci95_welch_from_samples(x, y)
    # Hedges' g
    g = hedges_g_from_samples(x, y)
    # Medias/SD (informativas)
    meanA, sdA = np.nanmean(x), np.nanstd(x, ddof=1)
    meanB, sdB = np.nanmean(y), np.nanstd(y, ddof=1)
    nA = int(np.isfinite(x).sum()); nB = int(np.isfinite(y).sum())

    print(f"{label}: {args.groupA} mean={meanA:.4f}±{sdA:.4f} vs {args.groupB} mean={meanB:.4f}±{sdB:.4f} | "
          f"t={tstat:.2f}, df={df_w:.1f}, p={pval:.4g}, g={g:.2f}, 95%CI[{lo:.4f},{hi:.4f}]")
    return dict(
        network=label,
        groupA=args.groupA, nA=nA, meanA=float(meanA), sdA=float(sdA),
        groupB=args.groupB, nB=nB, meanB=float(meanB), sdB=float(sdB),
        t=float(tstat), df=float(df_w), p=float(pval), g=float(g),
        ci95_low=float(lo), ci95_high=float(hi),
        metric=("PSC_global(%)" if args.psc else "mean_BOLD(a.u.)")
    )

summary_rows = [report("mean_DMN", "DMN"),
                report("mean_DAN", "DAN")]

summary_csv = f"{args.outprefix}_stats_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
print(f"Resumen estadístico guardado en: {summary_csv}")

# =========
# Figuras
# =========
labels = (args.groupA, args.groupB)
ylabel = "Mean PSC (%)" if args.psc else "Mean BOLD (a.u.)"

violin_two_groups(A["mean_DMN"], B["mean_DMN"],
                  labels[0], labels[1],
                  "Mean activation in DMN" + (" (PSC)" if args.psc else ""),
                  f"{args.outprefix}_violin_meanBOLD_DMN.svg",
                  ylabel=ylabel)

violin_two_groups(A["mean_DAN"], B["mean_DAN"],
                  labels[0], labels[1],
                  "Mean activation in DAN" + (" (PSC)" if args.psc else ""),
                  f"{args.outprefix}_violin_meanBOLD_DAN.svg",
                  ylabel=ylabel)
