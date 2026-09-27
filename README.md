# WIUT Hackathon 2026 — CV Track

Fixed-camera traffic event detection & accident risk anticipation.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2Bcu126-red)
![Tests](https://img.shields.io/badge/Tests-732%20passed-green)

## Quickstart (Docker)
```bash
docker build -t wiut-smart-traffic .
docker run --rm --gpus all -v /data/test:/data/test wiut-smart-traffic
```

## Quickstart (Local)
```bash
pip install -r requirements-torch.txt -r requirements.txt
python run_submission.py --videos samples --out predictions.json --team <name>
python evaluate.py --pred predictions.json --gt ground_truth.json
```

## Test Commands
```bash
pytest tests -q                    # 732 tests
pytest tests -k "integration" -v   # integration only
pytest tests/test_solution_format.py -v  # format check
```

## Architecture
```
Video → Pipeline (sampled decode)
       ├── Detector (YOLO11x + ByteTrack) → Trajectories
       ├── Scene Geometry (scene_config.json) → Lanes, crosswalks, zones
       ├── EventManager (14 providers) → Events [start, end, label]
       └── RiskEstimator (TTC + braking + pedestrian) → P(accident|5s)
```

## Output Format
```json
{
  "team": "team-name",
  "videos": {
    "test_001.mp4": {
      "events": [[12.4, 18.9, "accident"], ...],
      "risk": [[0.00, 0.01], [0.04, 0.01], ...]
    }
  }
}
```

## ⚠️ Critical: Weights & Git LFS
`weights/yolo11x.pt` (109 MB) is stored in Git LFS. **Clone with `git lfs install` or run `bash weights/download.sh` before offline evaluation.** A plain clone yields a 130-byte pointer — `torch.load` fails silently, producing zero detections and `Score_A = Score_B = 0` while `evaluate.py --validate-only` still reports `VALID`.

## Event Classes (14)
Exact labels required by the harness (`solution.CLASSES`):
```python
["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
 "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
 "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]
```

## Constraints
- **Time budget**: 3× video duration (Part A + B combined); over-budget = empty submission
- **Determinism**: No RNG in inference path — identical output on same machine by construction
- **Offline grading**: No internet at evaluation; Docker image must be pre-built online
- **RiskEstimator.step()**: Called per-frame by harness; must not open video; internal stride OK

## Scoring
- **Score_A**: Mean F1 across 14 classes × tIoU thresholds {0.3, 0.5, 0.7} (greedy IoU matching)
- **Score_B** (accident only): 0.4·AP + 0.4·F1_alarm + 0.2·mTTA/W (H=5s, W=10s, θ=0.5)
- **Model score**: M = 0.7·Score_A + 0.3·Score_B (M = Score_A if no accidents in test set)

## Key Files
- `solution.py` — `CLASSES`, `detect_events()`, `RiskEstimator`
- `src/events/manager.py` — 14 event providers registry
- `src/scene/geometry.py` — calibrated scene layout
- `scene_config.json` — shared geometry for all videos
- `weights/yolo11x.pt` — included (109 MB, Git LFS)

## Environment (Verified)
| Package | Version |
|---------|---------|
| torch | 2.14.0+cu126 |
| torchvision | 0.29.0+cu126 |
| numpy | 2.5.3 |
| opencv-python | 5.0.0.93 |
| ultralytics | 8.4.160 |

## Licences
- YOLO11x / ultralytics: **AGPL-3.0**
- torch / torchvision: **BSD-3-Clause**
- numpy: **BSD-3-Clause**
- opencv-python: **Apache-2.0**
- Sample clips: Organizers' task data

## Repo Structure
```
.
├── solution.py           # ONLY file to implement
├── run_submission.py     # Organizers' harness (do not modify)
├── evaluate.py           # Format check + metric (do not modify)
├── src/                  # Implementation
│   ├── pipeline/         # Video decoding, sampling
│   ├── detection/        # YOLO + ByteTrack wrapper
│   ├── tracking/         # Trajectories, motion, interaction
│   ├── scene/            # Geometry, calibration
│   ├── events/           # 14 rule-based providers
│   ├── risk/             # Part B estimator
│   └── postprocessing/   # Segment merge, blip removal
├── tests/                # 732 tests
├── weights/              # yolo11x.pt (LFS) + download.sh
├── examples/             # ground_truth.json, predictions.json
└── Dockerfile            # Offline-safe image
```