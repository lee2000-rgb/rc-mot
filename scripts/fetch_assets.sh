#!/bin/bash
# Fetch what git does not carry from the lab asset server, RCMOT_ASSETS=<user>@<host>:<path>/assets:
#   models/                 YOLO11 (.pt), OSNet ReID, DA-V2 Small (depth cue), VGGT-Omega 1B (work monitor / recon, 4.3 GB)
#   third_party/vggt-omega  VGGT-Omega code (work monitor / recon)
#   third_party/MOTDD       MOTDD original (reference only; not imported)
#   data/labels/            GT segments (gt.json; meta.bag = where each recording was on the dev PC)
# usage: scripts/fetch_assets.sh [--no-vggt]
#   --no-vggt  skip the 4.3 GB VGGT-Omega weights (safety alarm only)
# Recordings (ROS 2 bags) are not copied: see README, section 데이터.
set -euo pipefail
SRC=${RCMOT_ASSETS:?set RCMOT_ASSETS=<user>@<host>:<path>/assets (ask the lab admin)}
cd "$(dirname "$0")/.."
NOVGGT=0
for a in "$@"; do case $a in --no-vggt) NOVGGT=1;; *) sed -n 2,10p "$0"; exit 1;; esac; done
mkdir -p models third_party data
EX=(); [ $NOVGGT = 1 ] && EX=(--exclude=/vggt/)
rsync -a --info=progress2 "${EX[@]}" "$SRC/models/" models/
rsync -a "$SRC/third_party/MOTDD" third_party/
[ $NOVGGT = 1 ] || rsync -a "$SRC/third_party/vggt-omega" third_party/
rsync -a "$SRC/labels/" data/labels/
echo "done: $(du -sh models | cut -f1) models"
