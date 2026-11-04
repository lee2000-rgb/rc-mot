# One tracked object of the RC-MOT pipeline (docs/20), in frame base_link_level (header of TrackArray).
int32 id
string cls                          # worker | machine | ...
uint8 STATE_TENTATIVE=0
uint8 STATE_CONFIRMED=1
uint8 STATE_LOST=2
uint8 STATE_RADAR_ONLY=3
uint8 state
float32 score
float32 exist                       # existence score (tracker S4)
geometry_msgs/Point position        # [m]
geometry_msgs/Vector3 velocity      # absolute velocity, ego motion removed [m/s]
float64[9] position_covariance      # row-major 3x3, all zero when unknown
bool is_static                      # standstill (ZUPT) mode: velocity held at zero
float32 range                       # horizontal distance [m]
uint8 zone                          # 0 none 1 caution 2 warning 3 danger (by range alone, no hysteresis)
string source                       # radar | ground_ray | roi_research | radar_only
uint16 n_radar
float32 age                         # [s]
float32 time_since_update           # [s]
int32[4] box2d                      # camera box x1 y1 x2 y2 [px], -1 when none
