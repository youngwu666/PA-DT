







from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook


OUTPUT_NAME = "uncertainty_metrics.xlsx"
SIMULATION_SHEET = "Simulation"
PROBABILITY_SHEET = "Simulation_State_Prob"


def _read_rows(sheet):
    rows = sheet.iter_rows(values_only=True)
    headers = next(rows)
    return headers, list(rows)


def _find_result_workbook(folder):
    candidates = [
        path for path in folder.glob("*.xlsx")
        if path.name != OUTPUT_NAME and not path.name.startswith("~$")
    ]
    valid = []

    for path in candidates:
        try:
            workbook = load_workbook(path, read_only=True, data_only=True)
            if SIMULATION_SHEET not in workbook.sheetnames or PROBABILITY_SHEET not in workbook.sheetnames:
                workbook.close()
                continue
            simulation_headers, _ = _read_rows(workbook[SIMULATION_SHEET])
            probability_headers, _ = _read_rows(workbook[PROBABILITY_SHEET])
            workbook.close()
            if {"time", "p_state", "Lower", "Upper"}.issubset(simulation_headers) \
                    and "time" in probability_headers:
                valid.append(path)
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            f"No result workbook with '{SIMULATION_SHEET}' and "
            f"'{PROBABILITY_SHEET}' was found in: {folder}"
        )

    
    return max(valid, key=lambda path: path.stat().st_mtime)


def _as_float(value, field, time):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} at time {time}: {value!r}") from error
    if not np.isfinite(number):
        raise ValueError(f"Non-finite {field} at time {time}: {value!r}")
    return number


def calculate_metrics(workbook_path):
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    simulation_headers, simulation_rows = _read_rows(workbook[SIMULATION_SHEET])
    probability_headers, probability_rows = _read_rows(workbook[PROBABILITY_SHEET])
    workbook.close()

    simulation_column = {name: index for index, name in enumerate(simulation_headers)}
    probability_column = {name: index for index, name in enumerate(probability_headers)}
    state_headers = [name for name in probability_headers if name != "time"]
    if not state_headers:
        raise ValueError("No discrete-state probability columns were found.")

    state_values = np.asarray([float(name) for name in state_headers], dtype=float)
    probability_by_time = {
        row[probability_column["time"]]: np.asarray(
            [float(row[probability_column[name]]) for name in state_headers], dtype=float
        )
        for row in probability_rows
        if row[probability_column["time"]] is not None
    }

    per_step = []
    for row in simulation_rows:
        time = row[simulation_column["time"]]
        if time is None:
            continue
        if time not in probability_by_time:
            raise ValueError(f"No posterior probability row matches simulation time {time}.")

        true_state = _as_float(row[simulation_column["p_state"]], "true discrete state", time)
        lower = _as_float(row[simulation_column["Lower"]], "lower interval bound", time)
        upper = _as_float(row[simulation_column["Upper"]], "upper interval bound", time)
        if lower > upper:
            raise ValueError(f"Lower interval bound exceeds upper bound at time {time}.")

        probabilities = probability_by_time[time]
        if (not np.isfinite(probabilities).all()) or np.any(probabilities < 0):
            raise ValueError(f"Invalid posterior probability vector at time {time}.")
        probability_sum = probabilities.sum()
        if probability_sum <= 0:
            raise ValueError(f"Posterior probability vector sums to zero at time {time}.")
        probabilities = probabilities / probability_sum

        matching_state = np.flatnonzero(np.isclose(state_values, true_state, atol=1e-9, rtol=0))
        if matching_state.size != 1:
            raise ValueError(
                f"True state {true_state} at time {time} does not match exactly one "
                "state column in Simulation_State_Prob."
            )

        one_hot = np.zeros(len(state_values), dtype=float)
        one_hot[matching_state[0]] = 1.0
        covered = lower - 1e-12 <= true_state <= upper + 1e-12
        brier_score = float(np.sum((probabilities - one_hot) ** 2))
        per_step.append({
            "time": time,
            "true_discrete_state": true_state,
            "lower_95": lower,
            "upper_95": upper,
            "interval_width": upper - lower,
            "covered": int(covered),
            "brier_score": brier_score,
        })

    if not per_step:
        raise ValueError("No simulation time steps were found.")

    n_steps = len(per_step)
    picp = float(np.mean([row["covered"] for row in per_step]))
    mpiw = float(np.mean([row["interval_width"] for row in per_step]))
    brier = float(np.mean([row["brier_score"] for row in per_step]))
    return {
        "n_time_steps": n_steps,
        "nominal_interval": "95% equal-tailed credible interval",
        "PICP": picp,
        "covered_steps": int(sum(row["covered"] for row in per_step)),
        "MPIW": mpiw,
        "MPIW_unit": "remaining-wall-thickness ratio (dimensionless)",
        "Brier_score": brier,
        "Brier_definition": "Mean over time of the sum over states of (posterior probability - one-hot truth)^2",
    }, per_step


def save_results(output_path, summary, per_step):
    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"
    summary_sheet.append(["Metric", "Value"])
    for name, value in summary.items():
        summary_sheet.append([name, value])

    details_sheet = workbook.create_sheet("Per_time_step")
    headers = list(per_step[0].keys())
    details_sheet.append(headers)
    for row in per_step:
        details_sheet.append([row[name] for name in headers])

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            width = min(max(len(str(cell.value or "")) for cell in column) + 2, 70)
            sheet.column_dimensions[column[0].column_letter].width = width

    workbook.save(output_path)


def main():
    folder = Path(__file__).resolve().parent
    result_workbook = _find_result_workbook(folder)
    summary, per_step = calculate_metrics(result_workbook)
    output_path = folder / OUTPUT_NAME
    save_results(output_path, summary, per_step)

    print(f"Input workbook: {result_workbook.name}")
    print(f"Evaluation: simulation phase only ({summary['n_time_steps']} time steps)")
    print(f"PICP (95%): {summary['PICP']:.4f} ({summary['covered_steps']}/{summary['n_time_steps']})")
    print(f"MPIW: {summary['MPIW']:.6f} {summary['MPIW_unit']}")
    print(f"Brier score: {summary['Brier_score']:.6f}")
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()
