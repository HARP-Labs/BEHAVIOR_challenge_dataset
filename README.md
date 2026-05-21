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

Configure the config you are using for the correct output target (`config.output`). Either set `hf_repo_id` to stream shards directly to a HuggingFace dataset repo, or set `local_output_dir` to write shards to a local directory. Adjust `batch_size` and `fps` to match your GPU memory and desired sampling rate.

```bash
# V-JEPA2 (HuggingFace backend, 256px)
python encode_behavior_dataset.py --fname configs/behavior-vjepa2-vitg16-256px-16f.yaml

# V-JEPA 2.1 (native backend, 384px)
python encode_behavior_dataset.py --fname configs/behavior-vjepa21-vitg16-384px-16f.yaml
```

Encodes all episodes in the manifest.

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

**`meta`**

| Key | Default | Description |
|---|---|---|
| `meta.dtype` | `bfloat16` | Encoder I/O dtype (`float32`, `float16`, `bfloat16`) |
| `meta.compile` | `true` | `torch.compile` the encoder (`reduce-overhead` mode) |

**`model`**

| Key | Default | Description |
|---|---|---|
| `model.backend` | `hf` | `hf` (HuggingFace `AutoModel`) or `native_vjepa21` (native repo factory) |
| `model.hf_repo` | `facebook/vjepa2-vitg-fpc64-256` | HF model repo (used when `backend: hf`) |
| `model.temporal_patch_size` | `2` | Tubelet size in frames; auto-detected from model config, this is the fallback |

**`data`**

| Key | Default | Description |
|---|---|---|
| `data.datasets` | — | List of `manifest.json` paths produced by `base_dataset.py` |
| `data.dataset_fpcs` | — | Frames-per-clip for each manifest (must be divisible by `temporal_patch_size`) |
| `data.fps` | — | Frame sampling rate (fps) for video decoding |
| `data.crop_size` | `256` | Spatial crop resolution in pixels |
| `data.camera_view` | `multi` | `"head"` or `"multi"` (head + left\_wrist + right\_wrist) |
| `data.action_dim` | `23` | Action dimension per step |
| `data.batch_size` | `4` | DataLoader batch size (windows per batch) |
| `data.num_workers` | `8` | DataLoader worker processes |
| `data.persistent_workers` | `true` | Keep workers alive between batches |
| `data.prefetch_factor` | `8` | Batches prefetched per worker |
| `data.pin_mem` | `true` | Pinned memory for faster host→GPU transfer |
| `data.cache_parquet` | `true` | Cache decoded parquet action/state tables in memory |
| `data.cache_video_readers` | `false` | Cache Decord `VideoReader` objects across clips |
| `data.cache_max_entries` | `12` | Max cached video readers (LRU, when `cache_video_readers: true`) |

**`data_aug`**

| Key | Default | Description |
|---|---|---|
| `data_aug.horizontal_flip` | `false` | Random horizontal flip |
| `data_aug.random_resize_aspect_ratio` | `[0.75, 1.35]` | Aspect ratio jitter range |
| `data_aug.random_resize_scale` | `[1.777, 1.777]` | Scale jitter range (fixed = no jitter) |
| `data_aug.reprob` | `0.0` | Random erasing probability |
| `data_aug.auto_augment` | `false` | AutoAugment policy |
| `data_aug.motion_shift` | `false` | Temporal motion shift augmentation |

**`output`**

| Key | Default | Description |
|---|---|---|
| `output.local_output_dir` | — | Local directory for MDS shards; leave blank for HF-only mode (shards written to a tmpdir and deleted after upload) |
| `output.hf_repo_id` | — | Destination HuggingFace dataset repo for MDS shards |
| `output.hf_path_prefix` | `""` | Path prefix inside the HF repo (e.g. `shards_256px_vit_16_g`) |
| `output.max_shard_bytes` | `1073741824` | Max shard file size in bytes (default 1 GiB) |
| `output.commit_batch_size` | `20` | Shards bundled per HF `create_commit` call; keeps total commits well below the 128/hour API limit |
| `output.num_upload_workers` | `1` | Concurrent HF commit threads; values > 1 hide API round-trip latency when uploads are concurrency-bound |
| `output.max_pending_shards` | — | Pause encoding when this many shards are waiting on disk (e.g. `40` ≈ 40 GiB at 1 GiB/shard); leave unset to disable backpressure |

### `configs/behavior-vjepa21-vitg16-384px-16f.yaml` — encoding (native backend)

Same structure as above. Key differences:

| Key | Value |
|---|---|
| `model.backend` | `native_vjepa21` |
| `model.model_name` | Architecture to load: `vjepa2_1_vit_base_384`, `vjepa2_1_vit_large_384`, `vjepa2_1_vit_giant_384`, `vjepa2_1_vit_gigantic_384` |
| `data.crop_size` | `384` |

## MDS Shard Schema

Each shard row corresponds to one timestep (one frame) and contains tokens for all camera views together:

| Column | Type | Description |
|---|---|---|
| `tokens_head` | `ndarray [tokens_per_frame, D]` | V-JEPA2 tokens for head camera at this frame |
| `tokens_left_wrist` | `ndarray [tokens_per_frame, D]` | V-JEPA2 tokens for left wrist camera at this frame (present when `camera_view: multi`) |
| `tokens_right_wrist` | `ndarray [tokens_per_frame, D]` | V-JEPA2 tokens for right wrist camera at this frame (present when `camera_view: multi`) |
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

**`behavior.py`** — `BehaviorVideoDataset` is a PyTorch `Dataset` that reads the manifest and yields fixed-length clips: loads frames via Decord, extracts the 133-dim proprioceptive state, and returns `{video, actions, states}`. `BehaviorEpisodePreencoder` wraps this with a DataLoader, runs the encoder over full episodes, accumulates tokenized tubelets, writes MDS shards via `MDSWriter`, and concurrently uploads completed shards to HuggingFace via `_ShardUploader` (a background polling thread backed by a `ThreadPoolExecutor` for parallel commits). Backpressure via `max_pending_shards` pauses encoding when uploads fall behind to prevent disk exhaustion.

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
