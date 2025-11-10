# ============================================================
# Example usage:
# ============================================================
# python plot_seed_violins.py \
#   --csv /data/results/PD_vs_AD_seed_connectivity.csv \
#   --groupA PD --groupB AD \
#   --dx-col diagnosis \
#   --outprefix plots_PD_vs_AD
#
# This script generates violin plots for seed-based connectivity
# values (e.g., DMN_seedZ and DAN_seedZ), using consistent colors
# across groups and export formats (PNG, SVG, PDF).
#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# ---------- utilidades estadísticas ----------
def sem(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.std(x, ddof=1)/np.sqrt(len(x)) if len(x) > 1 else np.nan

def welch_t_df(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)];       b = b[~np.isnan(b)]
    t, p = ttest_ind(a, b, equal_var=False)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    num = (s1/n1 + s2/n2)**2
    den = (s1**2/((n1**2)*(n1-1))) + (s2**2/((n2**2)*(n2-1)))
    df = num/den if den > 0 else (n1+n2-2)
    return t, df, p

def hedges_g(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)];       b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: return np.nan
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
    if not np.isfinite(sp) or sp <= 0: return np.nan
    d = (np.mean(a) - np.mean(b)) / sp
    J = 1 - 3/(4*(n1+n2)-9) if (n1+n2) > 2 else 1.0
    return d * J

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    raise ValueError(f"No encontré ninguna de estas columnas: {candidates}. Disponibles: {list(df.columns)}")

# ---------- colores consistentes ----------
COLORS = {
    "PD": "#5DADE2",   # azul
    "AD": "#E67E22",   # naranja
    "MD": "#AF7AC5",  # violeta
    "HC": "#58D68D"    # verde
}

# ---------- graficador ----------
def save_violin(dataA, dataB, groupA, groupB, ylabel, title, outprefix, filename):
    colorA = COLORS.get(groupA, "#5DADE2")
    colorB = COLORS.get(groupB, "#E67E22")
    A = np.asarray(dataA, float); B = np.asarray(dataB, float)
    A = A[~np.isnan(A)]; B = B[~np.isnan(B)]
    nA, nB = len(A), len(B)
    meanA, meanB = np.nanmean(A), np.nanmean(B)
    semA, semB = sem(A), sem(B)

    t, dfw, p = welch_t_df(A, B)
    g = hedges_g(A, B)

    fig, ax = plt.subplots(figsize=(6,6))
    vp = ax.violinplot([A, B], showmeans=False, showextrema=False, widths=0.9)
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor([colorA, colorB][i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.35)

    # puntos (ligero jitter)
    rng = np.random.default_rng(123)
    jitterA = 1 + (rng.random(nA) - 0.5)*0.15
    jitterB = 2 + (rng.random(nB) - 0.5)*0.15
    ax.scatter(jitterA, A, color=colorA, s=28, alpha=0.8)
    ax.scatter(jitterB, B, color=colorB, s=28, alpha=0.8)

    # boxplots encima
    ax.boxplot([A, B], positions=[1,2], widths=0.18, patch_artist=True,
               boxprops=dict(facecolor='none', color='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))

    # medias ± SEM
    ax.errorbar([1,2], [meanA, meanB], yerr=[semA, semB],
                fmt='o', color='black', lw=2, capsize=5, zorder=10)

    ax.set_xticks([1,2]); ax.set_xticklabels([groupA, groupB], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # texto de stats
    txt = (f"Welch t={t:.2f}, df={dfw:.1f}, p={p:.4g}, g={g:.2f}\n"
           f"{groupA}: n={nA}, mean={meanA:.3f} ± SEM {semA:.3f}\n"
           f"{groupB}: n={nB}, mean={meanB:.3f} ± SEM {semB:.3f}")
    fig.text(0.02, 0.02, txt, fontsize=9)

    plt.tight_layout()
    base = f"{outprefix}_{filename}"
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base}.svg", format="svg", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {base}.png/.svg/.pdf")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Violin plots para seed-based connectivity (DMN/DAN) por grupos.")
    ap.add_argument("--csv", required=True, help="CSV por sujeto con columnas de diagnóstico y seedZ.")
    ap.add_argument("--groupA", required=True, help="Etiqueta grupo A (p.ej., PD o MDD).")
    ap.add_argument("--groupB", required=True, help="Etiqueta grupo B (p.ej., AD o HC).")
    ap.add_argument("--dx-col", default="diagnosis", help="Nombre de la columna de diagnóstico.")
    ap.add_argument("--outprefix", required=True, help="Prefijo de salida.")
    ap.add_argument("--dmn-cols", nargs="+",
                    default=["DMN_seedZ","DMN_seed_z","seed_DMN_z","dmn_seed_z","dmn_seedZ"],
                    help="Candidatos de columna para DMN.")
    ap.add_argument("--dan-cols", nargs="+",
                    default=["DAN_seedZ","DAN_seed_z","seed_DAN_z","dan_seed_z","dan_seedZ"],
                    help="Candidatos de columna para DAN.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]

    dmn_col = pick_col(df, args.dmn_cols)
    dan_col = pick_col(df, args.dan_cols)

    # Subconjuntos por grupo
    maskA = df[args.dx_col].astype(str).str.upper() == args.groupA.upper()
    maskB = df[args.dx_col].astype(str).str.upper() == args.groupB.upper()

    # Graficar DMN
    save_violin(
        df.loc[maskA, dmn_col].values,
        df.loc[maskB, dmn_col].values,
        args.groupA, args.groupB,
        ylabel=f"{dmn_col} (Fisher z)",
        title="Seed-based connectivity (DMN: PCC → DMN)",
        outprefix=args.outprefix,
        filename="DMN_seed_violin"
    )
    # Graficar DAN
    save_violin(
        df.loc[maskA, dan_col].values,
        df.loc[maskB, dan_col].values,
        args.groupA, args.groupB,
        ylabel=f"{dan_col} (Fisher z)",
        title="Seed-based connectivity (DAN: IPS/SPL → DAN)",
        outprefix=args.outprefix,
        filename="DAN_seed_violin"
    )

if __name__ == "__main__":
    main()
