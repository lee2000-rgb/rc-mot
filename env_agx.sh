# RC-MOT on the AGX Orin: source this in the same shell command as the program (docs/18).
export ROS_DOMAIN_ID=12                                   # the HDX rig's DDS domain (sensor_driver/hdx/gui/run_hdx_gui.sh)
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/agx/04.hdx_gui/hdx_ws/config/fastdds_agx_lan_only.xml
export PATH=$HOME/.local/bin:/usr/local/cuda-12.6/bin:$PATH
NV=$HOME/.local/lib/python3.10/site-packages/nvidia                       # pip wheels ONLY for the libs JetPack lacks:
LDX=""; for d in cuda_cupti cusparselt cudss cu12 nvjitlink; do [ -d $NV/$d/lib ] && LDX="$LDX$NV/$d/lib:"; done   # (cublas/nvrtc from the pip wheels break cuBLAS on Jetson)
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/extras/CUPTI/lib64:$LDX$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
[ -f /home/agx/04.hdx_gui/hdx_ws/install/setup.bash ] && source /home/agx/04.hdx_gui/hdx_ws/install/setup.bash
export PYTHONPATH=$HOME/rc_mot_orin/src:$PYTHONPATH
export DISPLAY=${DISPLAY:-:1}
