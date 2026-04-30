from __future__ import annotations

import random
from typing import Any

from .dataset_utils import (
    load_from_huggingface,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    load_yaml_config,
    logger,
)


class BaseDataset:
    """
    Base class for dataset handling. Provides common functionality for loading and managing datasets.
    """

    def __init__(self, use_hub_download: bool = False, token: str | None = None, **kwargs):
        """
        Initialize the BaseDataset using configuration from configs/dataset.yaml.
        Args:
            use_hub_download (bool, optional): Whether to use huggingface_hub for file download. Default is False.
            token (str, optional): Hugging Face authentication token.
            **kwargs: Additional arguments for dataset loading.
        """
        config = load_yaml_config("dataset.yaml")
        self.dataset_cfg = config.get("dataset", {})

        # Explicitly set each attribute from config (no fallbacks)
        self.camera_view_type = self.dataset_cfg["camera_view_type"]
        self.dataset_size = self.dataset_cfg["dataset_size"]
        self.seed = self.dataset_cfg["seed"]
        self.eval_tasks = self.dataset_cfg["eval_tasks"]
        self.exclude_eval_tasks = self.dataset_cfg["exclude_eval_tasks"]
        self.include_eval_tasks_fully = self.dataset_cfg["include_eval_tasks_fully"]
        self.obs_resolution = self.dataset_cfg["obs_resolution"]
        self.fps = self.dataset_cfg["fps"]
        self.shard_size = self.dataset_cfg["shard_size"]
        self.base_dataset_destination = self.dataset_cfg["base_dataset_destination"]
        self.encoded_dataset_destination = self.dataset_cfg["encoded_dataset_destination"]
        self.encoded_dataset_destination_path = self.dataset_cfg[
            "encoded_dataset_destination_path"
        ]
        self.augmentation = self.dataset_cfg["augmentation"]
        self.encode_dataset = self.dataset_cfg["encode_dataset"]

        self.repo_id = "behavior-1k/2025-challenge-demos"
        self.use_hub_download = use_hub_download
        self.token = token
        self.kwargs = kwargs
        self.logger = logger

        config_preview = {
            "camera": self.camera_view_type,
            "size": self.dataset_size,
            "seed": self.seed,
            "eval_tasks": len(self.eval_tasks),
            "exclude_eval_tasks": self.exclude_eval_tasks,
            "include_eval_tasks_fully": self.include_eval_tasks_fully,
            "obs_resolution": self.obs_resolution,
            "fps": self.fps,
            "shard_size": self.shard_size,
            "base_dst": self.base_dataset_destination,
            "encoded_dst": self.encoded_dataset_destination,
            "encode_dataset": self.encode_dataset,
        }
        self.logger.info(f"BaseDataset config: {config_preview}")

        # Build base dataset state immediately after initialization.
        self.build_base_dataset()

    def build_base_dataset(self) -> dict[str, Any]:
        """
        Load required metadata files from the repository /meta folder.

        Returns:
            dict[str, Any]: dictionary containing parsed file contents keyed by filename.
        """
        required_meta_files = {
            "info": "meta/info.json",
            "tasks": "meta/tasks.jsonl",
            "episodes": "meta/episodes.jsonl",
        }
        loaded_meta: dict[str, Any] = {"info": None, "tasks": None, "episodes": None}

        self.logger.info("Building base dataset: loading required metadata files from /meta.")

        for key, file_path in required_meta_files.items():
            try:
                local_path = load_from_huggingface(
                    self.repo_id,
                    file_path=file_path,
                    use_hub_download=True,
                    token=self.token,
                    **self.kwargs,
                )

                if file_path.endswith(".json"):
                    parsed = load_json_file(local_path)
                else:
                    parsed = load_jsonl_file(local_path)

                loaded_meta[key] = parsed
            except Exception as exc:
                self.logger.warning(f"Missing or unreadable: {file_path} ({exc})")

        self.info = loaded_meta["info"]
        self.tasks = loaded_meta["tasks"]
        self.episodes = loaded_meta["episodes"]

        found_count = len([v for v in loaded_meta.values() if v is not None])
        total_count = len(required_meta_files)
        self.logger.info(
            f"Base dataset metadata loaded: {found_count}/{total_count} files found "
            f"(info={self.info is not None}, tasks={self.tasks is not None}, episodes={self.episodes is not None})"
        )

        if found_count == total_count:
            self.selected_meta = self._build_selected_meta()
            self._log_metadata_preview()
        else:
            self.selected_meta = None

        return loaded_meta

    def _build_selected_meta(self) -> dict[str, Any]:
        if self.exclude_eval_tasks and self.include_eval_tasks_fully:
            raise ValueError(
                "Invalid config: exclude_eval_tasks and include_eval_tasks_fully cannot both be True."
            )

        rng = random.Random(self.seed)
        target_hours = float(self.dataset_size)
        eval_task_set = set(self.eval_tasks)

        episodes_by_task: dict[str, list[dict[str, Any]]] = {}
        for episode in self.episodes:
            task_name = self._episode_task_name(episode)
            episodes_by_task.setdefault(task_name, []).append(episode)

        selected: list[dict[str, Any]] = []
        selected_hours = 0.0

        if not self.exclude_eval_tasks and self.include_eval_tasks_fully:
            for eval_task in eval_task_set:
                eval_eps = episodes_by_task.get(eval_task, [])
                selected.extend(eval_eps)
                selected_hours += sum(self._episode_duration_hours(ep) for ep in eval_eps)

            if selected_hours > target_hours:
                raise ValueError(
                    f"Eval task episodes already exceed dataset_size: {selected_hours:.3f}h > {target_hours:.3f}h."
                )

            pool = [
                ep
                for task_name, task_eps in episodes_by_task.items()
                if task_name not in eval_task_set
                for ep in task_eps
            ]
        else:
            pool = [
                ep
                for task_name, task_eps in episodes_by_task.items()
                if (task_name not in eval_task_set or not self.exclude_eval_tasks)
                for ep in task_eps
            ]

        rng.shuffle(pool)
        while pool and selected_hours < target_hours:
            ep = pool.pop()
            selected.append(ep)
            selected_hours += self._episode_duration_hours(ep)

        selected_meta = {
            "target_hours": target_hours,
            "selected_hours": selected_hours,
            "num_selected_episodes": len(selected),
            "exclude_eval_tasks": self.exclude_eval_tasks,
            "include_eval_tasks_fully": self.include_eval_tasks_fully,
            "eval_tasks": list(eval_task_set),
            "episodes": [
                {
                    "task": self._episode_task_name(ep),
                    "duration_hours": self._episode_duration_hours(ep),
                    "video_file": ep.get("video_file") or ep.get("video_path"),
                    "data_parquet_file": ep.get("data_parquet_file") or ep.get("data_parquet_path"),
                    "episode_file": ep.get("episode_file") or ep.get("episode_path"),
                    "raw": ep,
                }
                for ep in selected
            ],
        }
        self.logger.info(
            f"Selected {len(selected)} episodes totaling {selected_hours:.3f}h "
            f"(target={target_hours:.3f}h)."
        )
        save_json_file("output/meta.json", selected_meta)
        self.logger.info("Saved selected metadata to output/meta.json")
        return selected_meta

    def _log_metadata_preview(self) -> None:
        tasks_count = len(self.tasks) if isinstance(self.tasks, list) else 0
        episodes_count = len(self.episodes) if isinstance(self.episodes, list) else 0
        selected_count = (
            len(self.selected_meta.get("episodes", []))
            if isinstance(self.selected_meta, dict)
            else 0
        )
        self.logger.info(
            f"Metadata preview: info_keys={list(self.info.keys())[:8] if isinstance(self.info, dict) else []}, "
            f"tasks={tasks_count}, episodes={episodes_count}, selected={selected_count}"
        )

    def _episode_task_name(self, episode: dict[str, Any]) -> str:
        return (
            episode.get("task")
            or episode.get("task_name")
            or episode.get("task_id")
            or "unknown_task"
        )

    def _episode_duration_hours(self, episode: dict[str, Any]) -> float:
        if "duration_hours" in episode:
            return float(episode["duration_hours"])
        if "duration_seconds" in episode:
            return float(episode["duration_seconds"]) / 3600.0
        if "num_frames" in episode and self.fps:
            return float(episode["num_frames"]) / float(self.fps) / 3600.0
        return 0.0
