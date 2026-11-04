# Pipeline status, 1 Hz (docs/20): what ran with which settings, for bags recorded by the HDX GUI.
std_msgs/Header header
string profile
string detector
string calib
float32 fps                         # processed frames per second (last second)
float32 det_ms                      # detector time, median of the last second
float32 total_ms                    # pipeline time, median of the last second
float32 cam_lag_s                   # newest radar stamp - newest image stamp
uint32 frames_dropped               # radar scans skipped since start
bool publishing
