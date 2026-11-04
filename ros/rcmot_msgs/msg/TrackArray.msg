# All displayed tracks of one processed frame (docs/20). stamp = radar scan time, frame_id = base_link_level.
std_msgs/Header header
Track[] tracks
geometry_msgs/Vector3 ego_velocity  # [m/s], base_link_level
bool ego_valid
