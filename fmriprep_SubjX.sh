# ============================================================
# Example usage:
# ============================================================
# bash fmriprep_SubjX.sh
#
# This script runs fMRIPrep for a single subject, assuming
# BIDS-compliant data and standard FreeSurfer reconstruction.
# Edit the following variables before running:
#
# BIDS_DIR=/data/BIDS
# OUT_DIR=/data/derivatives/fmriprep
# WORK_DIR=/data/work
# SUBJ=sub-01
#
# Example:
#   bash fmriprep_SubjX.sh
#
# After preprocessing, derivatives (including confounds and
# preprocessed BOLD) will be stored in $OUT_DIR.
#!/bin/bash

# === Configuration ===
BIDS_DIR=$HOME/Documents/BRIDGE/SRPBS/SRPBS_OPEN/data
OUT_DIR=$HOME/Documents/BRIDGE/SRPBS/SRPBS_OPEN/derivatives
FS_LICENSE=$HOME/Documents/GitLabMayo/CCEPs_rs-fMRI/license.txt
THREADS=8
MEM_MB=30000

# === folder ===
mkdir -p "$OUT_DIR"

# === Subj by subj ===
for sub in "$BIDS_DIR"/sub-*; do
    sub_id=$(basename "$sub")
    echo "Starting with $sub_id..."

    anat_dir="$sub/anat"

    # Detecta si hay T1w o T2w
    if ls "$anat_dir"/*T1w.nii* 1> /dev/null 2>&1; then
        echo "T1w for $sub_id"
        ANAT_OPTION=""
    elif ls "$anat_dir"/*T2w.nii* 1> /dev/null 2>&1; then
        echo "T2w for $sub_id"
        ANAT_OPTION="--anat-modality T2w"
    else
        echo "No found T1w or T2w for $sub_id"
        continue
    fi

    # Ejecuta fMRIPrep
    docker run --rm -ti \
        -v "$BIDS_DIR":/data:ro \
        -v "$OUT_DIR":/out \
        -v "$FS_LICENSE":/fs_license.txt \
        nipreps/fmriprep:24.1.1 \
        /data /out participant \
        --participant-label "${sub_id#sub-}" \
        --output-spaces MNI152NLin2009cAsym fsaverage \
        --fs-license-file /fs_license.txt \
        --nthreads $THREADS \
        --mem_mb $MEM_MB \
        --fs-no-reconall \
        $ANAT_OPTION

    echo "End $sub_id"
    echo ""
done

echo "The end."

