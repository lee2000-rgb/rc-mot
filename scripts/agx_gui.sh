#!/bin/bash
# Launch the RC-MOT GUI on the AGX desktop (:1) with the TensorRT engines (docs/18). Run on the AGX:
#   bash ~/rc_mot_orin/scripts/agx_gui.sh [--profile hdx_live_mid | --profile boryeong_halt-short] [more rcmot_app args]
# The profile box in the GUI then switches between bag replays and the LIVE: profiles (configs/datasets.yaml).
cd "$(dirname "$0")/.." || exit 1
source env_agx.sh
export DISPLAY=:1 XAUTHORITY=$HOME/.Xauthority
# clocks: jetson_clocks.service pins them at boot; re-apply here without a password (sudoers.d/20-jetson-clocks)
if sudo -n jetson_clocks --show 2>/dev/null | grep -q "GPU MinFreq=306000000"; then
    sudo -n nvpmodel -m 0 >/dev/null 2>&1; sudo -n jetson_clocks && echo "jetson_clocks applied"
fi
pkill -f "[r]cmot_app.py" 2>/dev/null; sleep 1
ARGS=("$@"); [ $# -eq 0 ] && ARGS=(--profile hdx_live_mid)
nohup python3 scripts/rcmot_app.py "${ARGS[@]}" --set detector.weights=models/yolo11s-seg.engine \
    --set reid.weights=models/reid/osnet_x0_25_msmt17.engine > /tmp/rcmot_gui.log 2>&1 &
echo "pid $! log /tmp/rcmot_gui.log"
