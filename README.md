# BEHAVIOR Challenge Dataset Pipeline

A dataset pipeline for the [BEHAVIOR-1K](https://behavior.stanford.edu/) robot manipulation challenge. Downloads episodes from HuggingFace, samples a budget-constrained subset, encodes video frames with a V-JEPA2 vision transformer, and writes [MDS shards](https://docs.mosaicml.com/projects/streaming/en/stable/) for downstream training.

## Pipeline Overview

```
HuggingFace BEHAVIOR-1K
        │
        ▼
┌───────────────────┐
│  base_dataset.py  │  Sample episodes by task up to a time budget → base_dataset/manifest.json +  raw MP4/parquet
└────────┬──────────┘
         │
         ▼
┌────────────────────────────┐
│  encode_behavior_dataset.py│  Encode frames with V-JEPA2 → MDS shards (tokens + actions + states)
└────────┬───────────────────┘
         │
         ▼
  HuggingFace MDS repo/ local
```

## Setup

```bash
cd ..
git clone https://github.com/LOTO-H-JEPA/vjepa2_BEHAVIOR.git
cd BEHAVIOR_challenge_dataset
```

```bash
pip install -e ../vjepa2_BEHAVIOR
pip install -r requirements.txt
```

A HuggingFace token with access to the BEHAVIOR-1K dataset is required:

```bash
export HF_TOKEN=<your_token>
```

## Running the Pipeline

### Step 1 — Download and sample episodes

```bash
python base_dataset.py
```

Reads `configs/base_dataset.yaml`, samples episodes up to the configured time budget, and writes raw files plus `base_dataset/manifest.json`.

### Step 2 — Encode with V-JEPA2

Configure the config you are using for the correct output target (`config.output`). Either set `hf_repo_id` to stream shards directly to a HuggingFace dataset repo, or set `local_dir` to write shards to a local directory. Adjust `batch_size` and `fps_clips_per_second` to match your GPU memory and desired sampling rate.

```bash
# V-JEPA2 (HuggingFace backend, 256px)
python encode_behavior_dataset.py --fname configs/behavior-vjepa2-vitg16-256px-16f.yaml

# V-JEPA 2.1 (native backend, 384px)
python encode_behavior_dataset.py --fname configs/behavior-vjepa21-vitg16-384px-16f.yaml
```

Encodes all episodes in the manifest and streams MDS shards to HuggingFace concurrently.

### Step 3 — Validate shards

```bash
jupyter notebook test_preencoded_dataset.ipynb
```

Runs schema, episode completeness, action/state alignment, and token shape checks.

## Configuration

### `configs/base_dataset.yaml` — episode sampling

| Key | Description |
|---|---|
| `camera_view_type` | `"head"` or `"multi"` (head + left/right wrist cameras) |
| `dataset_size` | Time budget in hours |
| `seed` | Random seed for deterministic sampling |
| `eval_tasks` | Tasks fully included regardless of budget |
| `exclude_eval_tasks` | Exclude eval tasks from the sampled split |
| `base_dataset_destination` | Output directory for raw episodes |

### `configs/behavior-vjepa2-vitg16-256px-16f.yaml` — encoding (HF backend)

| Key | Description |
|---|---|
| `model.hf_repo` | HuggingFace model repo (e.g. `facebook/vjepa2-vitg-fpc64-256`) |
| `data.fps` | Frame sampling rate |
| `data.crop_size` | Spatial crop resolution |
| `data.camera_view` | `"head"` or `"multi"` |
| `output.hf_repo_id` | Destination HF dataset repo for MDS shards |
| `output.max_shard_bytes` | Max shard size (default 1 GiB) |

### `configs/behavior-vjepa21-vitg16-384px-16f.yaml` — encoding (native backend)

Same structure as above but uses `model.backend: native_vjepa21` and `model.model_name` to select the architecture (`vjepa2_1_vit_base_384`, `vjepa2_1_vit_large_384`, `vjepa2_1_vit_giant_384`, `vjepa2_1_vit_gigantic_384`).

## MDS Shard Schema

Each shard row corresponds to one timestep (one frame) and contains tokens for all camera views together:

| Column | Type | Description |
|---|---|---|
| `tokens_head` | `ndarray [tokens_per_frame, D]` | V-JEPA2 tokens for head camera at this frame |
| `tokens_left` | `ndarray [tokens_per_frame, D]` | V-JEPA2 tokens for left wrist camera at this frame |
| `tokens_right` | `ndarray [tokens_per_frame, D]` | V-JEPA2 tokens for right wrist camera at this frame |
| `actions` | `ndarray [fstp * action_dim]` | Raw actions executed from this frame to the next |
| `states` | `ndarray [133]` | Proprioceptive state at this frame (133-dim curated subset) |
| `cam_rel_poses` | `ndarray [21]` | Camera poses at this frame (3 cameras × pos[3] + quat[4]) |
| `frame_index` | `int` | Source video frame index within the episode |
| `episode_idx` | `int` | Episode index |
| `sample_idx` | `int` | Sample index within the dataset |
| `step_pos` | `int` | Timestep position within the episode (0-indexed) |
| `episode_len` | `int` | Total number of timesteps in this episode |

## Proprioceptive State (`PROPRIO_DIM = 133`)

The 133-dim state vector is a curated subset of the raw 256-dim `observation.state` from BEHAVIOR-1K, selecting only sensor-observable quantities suitable for real-robot transfer:

- `[6:28]` — controllable joint positions (torso + arms + grippers)
- `[34:56]` — joint position sine encodings
- `[62:84]` — joint position cosine encodings
- `[84:112]` — joint velocities (all joints)
- `[152:158]` — linear and angular velocity (IMU)
- `[186:197]` — left end-effector pose + gripper state
- `[225:244]` — right end-effector pose + gripper state + trunk
- `[253:256]` — base encoder velocity

Simulator-only global state (absolute position, accumulated odometry) is excluded.

## Architecture

**`base_dataset.py` / `base_dataset_utils.py`** — `BaseDataset` loads task/episode metadata from the remote BEHAVIOR-1K HuggingFace repo, samples episodes per task up to the time budget, downloads parquet (actions/states), MP4 (video), and JSON (meta) files locally, then writes `manifest.json`.

**`behavior.py`** — `BehaviorVideoDataset` is a PyTorch `Dataset` that reads the manifest and yields fixed-length 8-frame clips: loads frames via Decord, extracts the 133-dim proprioceptive state, and returns `{video, actions, states}`. `BehaviorEpisodePreencoder` wraps this with a DataLoader, runs the encoder over full episodes, accumulates tokenized tubelets, writes MDS shards via `MDSWriter`, and concurrently uploads completed shards to HuggingFace via a background `_ShardUploader` thread.

**`encode_behavior_dataset.py`** — `HFVJEPA2Encoder` loads V-JEPA2 via HuggingFace `AutoModel`. `NativeVJEPA21Encoder` loads V-JEPA 2.1 via the native repo factory. Both expose an `encode_frames(video_tensor) → tokens` interface.

## License

See [LICENSE](LICENSE).

## Citations

If you use this pipeline or the associated work, please cite:

```bibtex
@misc{Quast2026,
  title={Short Horizon Planning with V-JEPA-2 AC on BEHAVIOR-1K},
  author={Quast, Julian},
  year={2026},
}

@article{Assran2025VJEPA2,
  title={V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mido and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and Arnaud, Sergio and Gejji, Abha and Martin, Ada and Hogan, Francois Robert and Dugas, Daniel and Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025},
}

@article{MurLabadia2026VJEPA21,
  title={V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning},
  author={Mur-Labadia, Lorenzo and Muckley, Matthew and Bar, Amir and Assran, Mido and Sinha, Koustuv and Rabbat, Mike and LeCun, Yann and Ballas, Nicolas and Bardes, Adrien},
  journal={arXiv preprint arXiv:2603.14482},
  year={2026},
}

@inproceedings{Li2022BEHAVIOR1K,
  title={BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
  author={Li, Chengshu and Zhang, Ruohan and Wong, Josiah and Gokmen, Cem and Srivastava, Sanjana and Mart{\'i}n-Mart{\'i}n, Roberto and Wang, Chen and Levine, Gabrael and Ai, Wensi and Martinez, Benjamin and others},
  booktitle={6th Conference on Robot Learning (CoRL)},
  year={2022},
}
```
