from __future__ import annotations

import os
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _to_plain_config(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {"value": str(config)}


def _find_params(tree: Any) -> Optional[Any]:
    if isinstance(tree, dict):
        if "params" in tree:
            return tree["params"]
        for value in tree.values():
            nested = _find_params(value)
            if nested is not None:
                return nested
    return None


def _is_orbax_checkpoint_dir(path: Path) -> bool:
    return (path / "_CHECKPOINT_METADATA").exists() or (path / "manifest.ocdbt").exists()


def _load_orbax(path: Path) -> Tuple[Any, Dict[str, Any]]:
    try:
        import orbax.checkpoint as ocp
    except ImportError as exc:
        raise RuntimeError(
            "Orbax is required to load this checkpoint. Install with `pip install orbax-checkpoint`."
        ) from exc

    resolved = path
    if (resolved / "models").is_dir():
        resolved = resolved / "models"

    errors = []

    # Try checkpoint manager API first (directory containing step subdirs).
    try:
        manager = ocp.CheckpointManager(str(resolved), ocp.PyTreeCheckpointer())
        step = manager.latest_step()
        if step is not None:
            restored = manager.restore(step)
            params = _find_params(restored)
            if params is None and isinstance(restored, dict):
                params = restored
            if params is None:
                raise RuntimeError("Could not locate `params` in Orbax checkpoint tree.")
            return params, {"backend": "orbax", "mode": "manager", "step": int(step)}
    except Exception as exc:  # pragma: no cover - version/runtime dependent
        errors.append(f"manager restore failed: {exc}")

    # Try direct restore against the given directory.
    checkpointer = None
    if hasattr(ocp, "PyTreeCheckpointer"):
        checkpointer = ocp.PyTreeCheckpointer()
    elif hasattr(ocp, "StandardCheckpointer"):
        checkpointer = ocp.StandardCheckpointer()

    if checkpointer is None:
        errors.append("no compatible Orbax checkpointer class found")
    else:
        try:
            restored = checkpointer.restore(str(resolved))
            params = _find_params(restored)
            if params is None and isinstance(restored, dict):
                params = restored
            if params is None:
                raise RuntimeError("Could not locate `params` in Orbax checkpoint tree.")
            return params, {"backend": "orbax", "mode": "direct", "step": None}
        except Exception as exc:  # pragma: no cover - version/runtime dependent
            errors.append(f"direct restore failed: {exc}")

    raise RuntimeError(
        "Unable to restore Orbax checkpoint from "
        f"{resolved}. Details: {' | '.join(errors)}"
    )


def _load_pickle(path: Path) -> Tuple[Any, Dict[str, Any]]:
    with path.open("rb") as file:
        payload = pickle.load(file)

    if isinstance(payload, dict) and "params" in payload:
        step = payload.get("step")
        return payload["params"], {"backend": "pickle", "step": step}

    if isinstance(payload, dict):
        params = _find_params(payload)
        if params is not None:
            return params, {"backend": "pickle", "step": payload.get("step")}

    return payload, {"backend": "pickle", "step": None}


def load_model_params(checkpoint_path: str) -> Tuple[Any, Dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if path.is_file() and path.suffix == ".pkl":
        return _load_pickle(path)

    if path.is_dir() and _is_orbax_checkpoint_dir(path):
        return _load_orbax(path)

    if path.is_dir() and (path / "models").is_dir():
        return _load_orbax(path)

    if path.is_file():
        # Last-resort pickle load for unknown extensions.
        return _load_pickle(path)

    raise RuntimeError(
        "Unsupported checkpoint path format. Expected a `.pkl` file or Orbax checkpoint directory. "
        f"Got: {path}"
    )


def save_pickle_checkpoint(path: str, params: Any, step: int, config: Any = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "step": int(step),
        "config": _to_plain_config(config),
    }
    with target.open("wb") as file:
        pickle.dump(payload, file)


def save_orbax_checkpoint(path: str, params: Any, step: int, config: Any = None) -> None:
    try:
        import orbax.checkpoint as ocp
    except ImportError as exc:
        raise RuntimeError(
            "Orbax is required to save Orbax checkpoints. Install with `pip install orbax-checkpoint`."
        ) from exc

    target_dir = Path(path)
    target_dir.mkdir(parents=True, exist_ok=True)

    item = {
        "params": params,
        "step": int(step),
        "config": _to_plain_config(config),
    }

    # Use the most stable API path across Orbax versions.
    if hasattr(ocp, "PyTreeCheckpointer"):
        checkpointer = ocp.PyTreeCheckpointer()
    elif hasattr(ocp, "StandardCheckpointer"):
        checkpointer = ocp.StandardCheckpointer()
    else:
        raise RuntimeError("No compatible Orbax checkpointer class found.")

    # Save under per-step subdirectories for easy resumption.
    step_dir = target_dir / str(step)
    if step_dir.exists():
        # Remove stale directory to avoid Orbax complaining on overwrite.
        # This is limited to the step path we own.
        import shutil

        shutil.rmtree(step_dir)

    checkpointer.save(str(step_dir), item)


def maybe_save_checkpoint(
    directory: str,
    params: Any,
    step: int,
    config: Any = None,
    prefer_orbax: bool = True,
) -> str:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    if prefer_orbax:
        try:
            save_orbax_checkpoint(str(target_dir), params, step, config=config)
            return str(target_dir / str(step))
        except Exception:
            pass

    pkl_path = target_dir / f"checkpoint_{step}.pkl"
    save_pickle_checkpoint(str(pkl_path), params, step, config=config)
    return str(pkl_path)
