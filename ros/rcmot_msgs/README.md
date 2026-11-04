# rcmot_msgs

ROS 2 interface package of the RC-MOT alarm (docs/20). Canonical copy lives here (`ros/rcmot_msgs`);
on the AGX it is built inside the HDX GUI workspace: `/home/agx/04.hdx_gui/hdx_ws/src/rcmot_msgs`
(`colcon build --packages-select rcmot_msgs`). `src/rcmot/sensors/bag_source.py` registers these .msg files
with `rosbags` so bags holding `/rcmot/*` topics can be read on a PC without ROS.
