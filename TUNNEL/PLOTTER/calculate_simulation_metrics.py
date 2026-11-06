







from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


OUTPUT_NAME = "simulation_metrics.xlsx"
ACTION_REWARDS = {
    "do_nothing": 30.0,
    "major_rehab": -100.0,
    "restrict_operations": 27.0,
    "inspection": 10.0,
}


def _has_simulation_sheet(path: Path) -> bool:
    if path.name == OUTPUT_NAME or path.name.startswith("~$"):
        return False
    try:
        with pd.ExcelFile(path) as workbook:
            return "Simulation" in workbook.sheet_names
    except Exception:
        return False


def _workbooks(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*.xlsx") if _has_simulation_sheet(path))


def _find_folders(location: Path) -> list[Path]:
    if location.is_file():
        if not _has_simulation_sheet(location):
            raise ValueError(f" Simulation {location}")
        return [location.parent]

    if _workbooks(location):
        return [location]

    children = sorted(path for path in location.glob("health_*")
                      if path.is_dir() and _workbooks(path))
    if children:
        return children

    
    candidates = (location / "simulation_results", location.parent / "simulation_results")
    for candidate in candidates:
        if _workbooks(candidate):
            return [candidate]
        nested = sorted(path for path in candidate.glob("health_*")
                        if path.is_dir() and _workbooks(path)) if candidate.is_dir() else []
        if nested:
            return nested
    raise FileNotFoundError(f" Simulation  .xlsx {location}")


def _damage_values(simulation: pd.DataFrame) -> np.ndarray:
    
    if "p_state_num" in simulation:
        values = pd.to_numeric(simulation["p_state_num"], errors="coerce").to_numpy(dtype=float)
    elif "p_state" in simulation:
        states = simulation["p_state"].astype("string").str.split(";")
        values = pd.to_numeric(states.str[1], errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError("Simulation  p_state_num  p_state")

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Simulation ")
    
    scale = 100.0 if np.nanmax(finite) <= 1.0 + 1e-12 else 1.0
    return values * scale


def _executed_action_mask(simulation: pd.DataFrame) -> np.ndarray:
    
    if "time" not in simulation:
        raise ValueError("Simulation  time ")
    time = pd.to_numeric(simulation["time"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(time)
    if "action_executed" in simulation:
        executed = simulation["action_executed"].fillna(False).astype(bool).to_numpy()
        return finite & executed
    if not finite.any():
        raise ValueError("Simulation  time ")
    return finite & (time < np.nanmax(time) - 1e-12)


def _action_totals(simulation: pd.DataFrame) -> float:
    
    if "current_action" not in simulation:
        raise ValueError("Simulation  current_action")
    mask = _executed_action_mask(simulation)
    actions = simulation["current_action"].astype("string").to_numpy()
    inspection = np.zeros(len(simulation), dtype=bool)
    if "inspection_performed" in simulation:
        inspection = simulation["inspection_performed"].fillna(False).astype(bool).to_numpy()
    elif "inspection_cost" in simulation:
        inspection = pd.to_numeric(simulation["inspection_cost"], errors="coerce").to_numpy() > 0
    elif "inspection_reward" in simulation:
        inspection = pd.to_numeric(simulation["inspection_reward"], errors="coerce").fillna(0).to_numpy() != 0

    total_reward = 0.0
    unknown = set()
    for index in np.flatnonzero(mask):
        action = actions[index]
        if pd.isna(action) or action == "<NA>":
            continue
        
        
        if action == "do_nothing" and inspection[index]:
            action = "inspection"
        if action not in ACTION_REWARDS:
            unknown.add(str(action))
            continue
        total_reward += ACTION_REWARDS[action]
    if unknown:
        raise ValueError("" + ", ".join(sorted(unknown)))

    return total_reward


def calculate_workbook(path: Path) -> dict:
    simulation = pd.read_excel(path, sheet_name="Simulation")
    damage = _damage_values(simulation)
    valid_damage = damage[np.isfinite(damage)]
    if valid_damage.size == 0:
        raise ValueError("Simulation ")

    total_reward = _action_totals(simulation)
    executed_steps = int(np.count_nonzero(_executed_action_mask(simulation)))
    average_reward = total_reward / executed_steps if executed_steps > 0 else np.nan

    return {
        "source_file": path.name,
        "mean_damage_percent": float(np.mean(valid_damage)),
        "max_damage_percent": float(np.max(valid_damage)),
        "damage_above_60_percent_count": int(np.count_nonzero(valid_damage > 60.0)),
        "total_action_reward": total_reward,
        "mean_action_reward": float(average_reward),
    }


def process_folder(folder: Path, workbook: Path | None = None) -> Path:
    paths = [workbook] if workbook is not None else _workbooks(folder)
    if not paths:
        raise FileNotFoundError(f" Simulation  .xlsx {folder}")
    records = []
    for path in paths:
        try:
            records.append(calculate_workbook(path))
        except Exception as error:
            warnings.warn(f" {path.name}: {error}", RuntimeWarning)
    if not records:
        raise ValueError(f"{folder}")

    summary = pd.DataFrame(records)
    output = folder / OUTPUT_NAME
    with pd.ExcelWriter(output) as writer:
        summary.to_excel(writer, sheet_name="Simulation_Metrics", index=False)
    print(f"\n{folder}")
    print(summary.to_string(index=False))
    print(f"{output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path,
                        help="health_*")
    args = parser.parse_args()
    location = args.folder.expanduser().resolve() if args.folder else Path(__file__).resolve().parent
    folders = _find_folders(location)
    selected_workbook = location if location.is_file() else None
    for folder in folders:
        process_folder(folder, selected_workbook)


if __name__ == "__main__":
    main()
