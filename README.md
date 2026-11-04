# RC-MOT

4D 레이더 + 카메라로 건설기계 주변 작업자를 찾고 추적해 거리 구역 경보를 내는 코드다. 같은 센서로 흙더미 작업량과 주행 중 3차원 재구성도 한다. 울산대학교 ISLab의 연구 코드다(아래 "라이선스").

| 기능 | 하는 일 | 핵심 코드 | 실행 |
|---|---|---|---|
| 안전관제 | 작업자 탐지·3D 위치·속도·ID, 10 / 20 / 30 m 구역 경보 | `rcmot.pipeline.Pipeline`, `radar/`, `detection/`, `fusion/`, `tracking/`, `alarm.py` | `scripts/rcmot_app.py`, `scripts/run_offline.py`, `scripts/eval_mot_gt.py` |
| 작업량 | 선회축 기준 2.5D 높이 지도, 흙더미 부피, 굴착·성토, 작업 사이클 | `rcmot.work.monitor`, `rcmot.work.depth_engine`(VGGT-Omega × 레이더 축척), `rcmot.world`(선회축 자세) | `scripts/rcmot_app.py --profile hdx1026_work_155530` → "작업" 탭 |
| 3차원 재구성 | 주행·선회하며 넓은 영역 점구름을 쌓음 | `rcmot.work.recon`(온라인), `scripts/drive_recon.py`(오프라인) | `rcmot_app.py --profile hdx1026_excA_140718` → "재구성" 탭 |

## 처리 흐름 (안전관제, 한 프레임)

```
레이더 스캔 ──► radar/scan_stats      필터(다중경로·약한 외딴 점), 자차 속도(도플러), 정지/이동 판정
영상 ────────► detection/yolo2d       YOLO11-seg 사람·장비 상자 + 마스크
             ► fusion/depth_stage     (선택, 기본 켬) DA-V2 Small 상대 깊이를 레이더 정지점으로 미터 맞춤, 3프레임마다
상자마다 ────► fusion/radar_camera    거리 확률분포 = 카메라 단서(키 사전·접지·폭·깊이 맵) × 상자 안 레이더 군집 배치 우도
               fusion/range_lik        (파라미터 configs/range_lik/full.json)
트랙 ────────► tracking/tracker       3D 칼만 + ByteTrack 2단계, 분포 × 트랙 예측으로 거리 선택, 도플러, 재탐색(recovery.py)
경보 ────────► alarm.py               확정·잠시 놓친·레이더만(카메라 1 s 이내) 트랙의 최근접 거리 → 구역 단계
```

## 폴더

```
src/rcmot/      패키지 (types.py: Det2D / RadarFrame / Det3D / TrackOut; pipeline.py: 한 프레임 처리)
  radar/        스캔 통계·자차 속도·이동체 군집·칸 기억(선택)
  detection/    YOLO (.pt / TensorRT .engine)
  fusion/       레이더-카메라 거리 분포, 깊이 맵 단계
  tracking/     추적기, 거리 선택, 재탐색
  reid/         OSNet 외형 특징
  work/         작업량 모니터, VGGT-Omega 깊이 엔진, 주행 재구성
  calib/        카메라-LiDAR, 레이더-LiDAR, 반사판, 현장 보정
  gui/          PyQt5 뷰어 (BEV·3D·트랙 카드·작업·재구성 탭)
  ros/          결과 발행 (/rcmot/tracks, /rcmot/alarm, ...)
  sensors/      ROS 2 bag 리더(ROS 설치 불필요), 실시간 구독
  labeling/     GT 형식, 평가 지표(HOTA, BEV, 경보 지표)
scripts/        실행·평가·보정·라벨 도구 (아래)
configs/        default.yaml (모든 키와 설명), 현장별 설정, datasets.yaml (GUI 프로필), calib/*.json, range_lik/
tests/          pytest
ros/rcmot_msgs/ ROS 2 메시지 패키지 (실시간 발행용)
assets/sounds/  경보음
third_party/depth_anything_v2/  DA-V2 망 코드 (Apache-2.0)
```

## 설치

Python 3.10, NVIDIA GPU(CUDA 11.8 휠). 개발 PC와 연구실 서버의 평가 환경이 같은 버전이다.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .                      # 패키지 rcmot (편집 모드: 코드가 저장소 루트의 configs/·models/를 찾는다)
RCMOT_ASSETS=<user>@<host>:<path>/assets scripts/fetch_assets.sh   # 모델·third_party·GT 라벨 (약 4.8 GB; 경보만 쓰면 --no-vggt)
pytest -q
```

- ROS 2 Humble은 실시간 입력(`--live`)과 결과 발행에만 필요하다. bag 재생은 `rosbags`로 읽어서 ROS가 없어도 된다.
- Jetson AGX Orin: `env_agx.sh`(ROS 도메인, CUDA 경로)를 먼저 source 한다. TensorRT 엔진은 `scripts/export_trt.sh`, `scripts/trt_build_onnx.py`로 Orin에서 만든다.

## git에 없는 것

| 무엇 | 어디에 | 받는 법 |
|---|---|---|
| `models/` (YOLO11, OSNet, DA-V2 Small, VGGT-Omega 1B) | 연구실 자산 서버 `<assets>/models` | `scripts/fetch_assets.sh` |
| `third_party/vggt-omega` (작업량·재구성), `third_party/MOTDD` (참고용, import 안 함) | 연구실 자산 서버 `<assets>/third_party` | 같음 |
| `data/labels/<구간>/gt.json` (GT 19구간 + 야간 1) | 연구실 자산 서버 `<assets>/labels` | 같음 |
| 녹화(ROS 2 bag) | 연구실 데이터 서버 (현장 원본, GT 구간 추출본 42 GB) | 아래 "데이터" |

자산 서버의 주소·경로(`RCMOT_ASSETS`)는 연구실 관리자에게 받는다. 모델은 공개 가중치(Ultralytics YOLO11, torchreid OSNet, Depth Anything V2 Small, VGGT-Omega)를 같은 이름으로 `models/`에 두어도 된다.

## 실행

```bash
# GUI: bag 재생(타임라인 탐색) — 프로필은 configs/datasets.yaml (bag·설정·보정·시작 시각)
python3 scripts/rcmot_app.py --profile boryeong_halt-short      # 목록은 configs/datasets.yaml
python3 scripts/rcmot_app.py --config configs/hdx_boryeong.yaml --bag <bag 폴더> [--calib <보정 json>] [--start 85]
python3 scripts/rcmot_app.py --profile hdx1026_work_155530     # 작업량 ("작업" 탭)
# 실시간 (ROS 2)
python3 scripts/rcmot_app.py --profile hdx_live_mid            # 또는 --config <설정> --live
# 화면 없이
python3 scripts/run_offline.py --bag <bag 폴더> --config <설정> --no-gui
# 설정 덮어쓰기 (GUI·평가 공통)
--set depth_map.enable=false --set cell_memory.enable=true
# 주행 재구성 (오프라인, LiDAR 비교 포함)
python3 scripts/drive_recon.py --profile hdx1026_excA_140718 --t0 0 --t1 120
```

`configs/datasets.yaml`의 bag 경로는 개발 PC 기준이다(`data/bags/...` 또는 `/media/islab/SAMSUNG/...`). 자기 PC 경로로 고치거나 같은 경로에 심볼릭 링크를 만든다. 파일이 없는 프로필은 GUI 목록에 나오지 않는다.

## 평가

```bash
# GT 한 구간: 2D HOTA / IDF1, BEV(2 m) 지표, 20·30 m 안 사람만 본 지표
python3 scripts/eval_mot_gt.py day_halt_short_085 --match bev2m --warmup 2.0 --out /tmp/eval.json --dump-pred /tmp/pred.json
python3 scripts/eval_suite.py run --tag base --segs conti,cf,val -j 2     # 여러 구간 묶음 (show --tags a,b 로 비교)
```

- 각 구간의 `gt.json` → `meta.bag`(녹화 경로), `meta.config`(설정)를 쓴다. 녹화가 그 경로에 있어야 한다.
- 비교 실험은 warmup 1.5 / 2.0 / 2.5 s 세 번을 돌려 구간별로 짝지어 본다(구간 부트스트랩). 한 번의 차이는 잡음일 수 있다.
- 거리 분포 파라미터: 배포용 `configs/range_lik/full.json`은 GT 19구간 전체로 맞춘 것이라 같은 구간 평가에서 낙관적이다(HOTA +0.008 정도). 공정한 비교는 구간 하나 빼기 파라미터 `--set fusion.range_lik_params=configs/range_lik/loso/<구간>.json`로, **비교하는 두 조건에 같은 방식**을 쓴다.

## 데이터

| 녹화 | 센서 | 내용 |
|---|---|---|
| 보령 실증장 2025-11 (`hdx_boryeong_pg`, conti) | Bosch / Continental ARS548 4D 레이더, 카메라, Ouster OS1-32 | 정지(가까운·중간 거리), 주행, 선회 |
| 보령 현장 2026-10-26 (`hdx_20261026_*`) | Bosch 레이더 ×3, OAK-D, Ouster, IMU, RTK | 삼각대, 굴착기 A(안전관제 거치, 선회), 굴착기 B(작업량) |
| 애리조나 (`construction_field`, `xpanner03/05`) | 레이더, 카메라, Ouster | 진동 큰 장비, 여러 작업자 |

설정 파일: `hdx_boryeong.yaml`, `hdx_boryeong_conti.yaml`, `hdx_20261026.yaml`, `xpanner_arizona.yaml`, `agmo.yaml`, 실시간 `hdx_agx_live.yaml`. 보정은 녹화 세션마다 `configs/calib/*.json`.

## 기본값 (2026-11-04)

- 거리: 거리 확률분포 방식(`fusion.range_model: lik`). 트랙은 분포와 자기 예측을 곱해 거리를 고른다. 트랙별 카메라 편향은 끔(`tracker.lik_bias: false`).
- 레이더 전처리: 점 신뢰도 α는 쓰지 않음(`scan_stats.alpha_source: none`). 약하고 외딴 점은 스캔 전체 RCS z < −2이면서 1.5 m 안에 이웃이 없는 점(`drop_source: rcs_scan`), 방위 모호성 짝은 RCS가 낮은 쪽을 버림(`ambig_source: rcs`), 지면 아래 다중경로 제거. 거리 구간별 SNR 통계는 쓰지 않는다.
- 영상 깊이 단서: 켬(`depth_map.enable: true`, DA-V2 Small 518×924, 3프레임마다, 25 m까지). 실시간 설정(`hdx_agx_live`, `conti_live`)과 작업량·재구성 프로필은 끔(Orin 시간 측정 전 / VGGT-Omega GPU 메모리). Orin에서 켤 때는 `every: 6`.
- 칸 기억(정지 중 배경 학습): 꺼짐(`cell_memory.enable`). 쐐기 대체: 꺼짐.
- 경보 계수 규칙(코드 기본값): 확정 트랙, 놓친 지 0.5 s 이내 트랙, 카메라가 마지막으로 본 뒤 1 s 이내의 레이더만 트랙(`alarm.confirmed_only`, `lost_grace_s`, `radar_only_grace_s`).
- ReID: 같은 작업복 대응(평균 빼기 + 부위 색, `reid.crowd_mean`, `reid.part_colour`) 켬.

설정 키 설명은 `configs/default.yaml` 주석(코드에만 있는 기본값 목록 포함)과 각 `...Params` 데이터클래스 주석에 있다.

## 함께 개발하기

- `main`에 직접 올리지 않고 브랜치 → Pull Request → 검토 후 병합한다.
- 병합 전 `pytest -q` 통과. 추적·경보 동작을 바꾸면 GT 구간 결과(위 "평가")를 PR에 붙인다.
- 모델·녹화·결과 파일(`*.pt`, `*.mcap`, `*.npz` 등)은 커밋하지 않는다(`.gitignore`). 큰 파일은 연구실 자산 서버에 둔다.
- 설정 기본값을 바꿀 때는 `configs/default.yaml` 주석에 근거(실험·수치)를 같이 적는다.

## 라이선스

이 저장소의 코드에는 아직 오픈소스 라이선스를 지정하지 않았다(저작권은 저작자에게 있고, 허락 없는 재사용·재배포는 할 수 없다). 받아 쓰는 외부 구성요소는 각자의 라이선스를 따른다.

| 구성요소 | 라이선스 | 비고 |
|---|---|---|
| Ultralytics YOLO11 | AGPL-3.0 | 배포·제품화 시 소스 공개 의무 또는 상용 라이선스 |
| VGGT-Omega (코드·가중치) | FAIR Noncommercial Research License | 비상업 연구만. 이 저장소에 포함하지 않음 |
| Depth Anything V2 Small | Apache-2.0 | `third_party/depth_anything_v2` |
| OSNet (torchreid 모델 정의·가중치) | MIT | `src/rcmot/reid/osnet.py` |
| MOTDD (울산대 최제환, IW-FCV 2026) | 저작자 허락 하에 사용 | 이 저장소에 포함하지 않음. 재탐색(PGR)·버퍼 IoU 아이디어를 다시 구현했다(`src/rcmot/tracking/recovery.py`) |
