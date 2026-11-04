# Alarm decision of one processed frame after hysteresis (docs/18, docs/20).
std_msgs/Header header
uint8 LEVEL_NONE=0
uint8 LEVEL_CAUTION=1
uint8 LEVEL_WARNING=2
uint8 LEVEL_DANGER=3
uint8 level
int32 nearest_id                    # -1 when no track counts
float32 nearest_range               # [m], NaN when none
float32[3] zones                    # danger, warning, caution boundaries [m]
bool sound_enabled                  # the alarm button of the RC-MOT GUI
bool confirmed_only                 # only camera-confirmed tracks count
