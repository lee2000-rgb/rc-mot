#!/bin/bash
# RC-MOT alarm GUI as an "algorithm" of the HDX GUI (docs/20): foreground, so the HDX ProcessSupervisor's
# `bash -c "<SOURCE_CMD> && bash .../hdx_algo_rcmot.sh [flags]"` IS this process (exec) and its Stop (SIGTERM to the
# process group) closes the window cleanly (rcmot_app.py handles SIGTERM -> Qt quit). Flags from the HDX GUI row:
#   --publish            publish /rcmot/* topics (checkbox "ROS 2 topic publish"); without it the publisher is created
#                        OFF (--no-publish: 1 Hz /rcmot/status with publishing=false, PUB button turns it on)
#   --profile <name>     another live profile (default hdx_live_site)
# Terminal use (background, log file) stays scripts/agx_gui.sh.
cd "$(dirname "$0")/.." || exit 1
source env_agx.sh
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
if sudo -n jetson_clocks --show 2>/dev/null | grep -q "GPU MinFreq=306000000"; then
    sudo -n nvpmodel -m 0 >/dev/null 2>&1; sudo -n jetson_clocks && echo "jetson_clocks applied"
fi
pkill -f "[r]cmot_app.py" 2>/dev/null && sleep 1          # a copy started from a terminal
ARGS=("$@")
case " $* " in *" --profile "*|*" --bag "*|*" --live "*) ;; *) ARGS=(--profile hdx_live_site "${ARGS[@]}");; esac
case " $* " in *" --publish "*|*" --no-publish "*) ;; *) ARGS+=(--no-publish);; esac
exec python3 -u scripts/rcmot_app.py "${ARGS[@]}" --set detector.weights=models/yolo11s-seg.engine \
    --set reid.weights=models/reid/osnet_x0_25_msmt17.engine
