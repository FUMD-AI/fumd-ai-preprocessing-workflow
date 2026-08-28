"""
=============================================================================
FUMD-AI Preprocessing Workflow -- shared library: migration labeling logic
=============================================================================
Used by: notebooks/step_6_label_cell_migrations.ipynb

Author(s):
  - Cristina Bernad (ORCID: 0000-0001-9537-415X)
  - Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855)
  - Katja Gilly (ORCID: 0000-0002-8985-0639)

Copyright:    (c) 2026 Cristina Bernad, Sonja Filiposka, Katja Gilly
Repository:   https://github.com/FUMD-AI/fumd-ai-preprocessing-workflow
Version:      1.1.4
Funding:      This work has been funded by the FUMD-AI project, an EOSC GRAVITY -
              Inter Project with Grant Number 25-EOSC-GRV-INTER-013.

-----------------------------------------------------------------------------
Licence
Unless otherwise indicated:

  * Source code in this notebook is licensed under the MIT License.

  * Explanatory text and original figures are licensed under Creative
    Commons Attribution 4.0 International (CC BY 4.0). Input datasets
    retain the licences stated in their corresponding metadata or
    source records.

SPDX-License-Identifier: MIT
-----------------------------------------------------------------------------

Cell-migration (handover) trajectory reconstruction and labeling.

This module contains the final, consolidated version of logic that was
previously reworked five times directly inside the "Vehicles_Combined_Migration"
notebooks (v1-v5). Only the last, most complete version (v5) is kept here;
it is the one that:

  * classifies every serving-cell change into an explicit case
    (normal handover, ping-pong, rapid A->B->C burst, handover with no
    prior stable history), and
  * freezes the "destination" label to the next *stable* cell rather than
    to whatever cell is momentarily reported during a ping-pong burst.

Two public functions are exposed:

  build_trajectories(df, ...)   -> per-vehicle runs of stable serving cells
  annotate_migrations(df, runs) -> the input df + migration/destination
                                     columns, plus an events log
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def build_trajectories(
    df: pd.DataFrame,
    time_col: str = "t",
    veh_col: str = "veh_id",
    cell_col: str = "servingCell",
) -> pd.DataFrame:
    """
    Summarize each vehicle's trajectory as a sequence of "runs": contiguous
    stretches of time during which the vehicle stayed on the same serving
    cell.

    Returns a DataFrame with columns:
        [veh_id, run_id, start_time, end_time, servingCell, duration]
    """
    d = df.sort_values([veh_col, time_col]).reset_index(drop=True).copy()

    # A run ends whenever the serving cell changes for a given vehicle.
    d["change"] = d[cell_col] != d.groupby(veh_col)[cell_col].shift()
    d["run_id"] = d.groupby(veh_col)["change"].cumsum()

    runs = (
        d.groupby([veh_col, "run_id"], as_index=False).agg(
            start_time=(time_col, "first"),
            end_time=(time_col, "last"),
            **{cell_col: (cell_col, "first")},
        )
    )
    runs["duration"] = runs["end_time"] - runs["start_time"]
    return runs


def annotate_migrations(
    df_sorted: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    window_s: float,
    tol_s: float = 0.1,
    time_col: str = "t",
    veh_col: str = "veh_id",
    cell_col: str = "servingCell",
    announce_policy: str = "strict",
    handle_bursts: bool = True,
    destination_mode: str = "freeze_stable",
    log_all_cases: bool = True,
):
    """
    Label every row of `df_sorted` with:

      migration (Int8):
          0 - steady state (no upcoming handover within `window_s`)
          1 - first row of the pre-handover warning window
          2 - remaining rows of the pre-handover warning window

      destination (Int64):
          the serving cell the vehicle will actually settle on. Inside a
          warning window this is the upcoming stable cell; outside a
          window it is frozen to the current/next stable cell so that
          short-lived ping-pong blips are not reported as destinations
          (destination_mode="freeze_stable"), or simply mirrors the raw
          servingCell value (destination_mode="serving").

    Parameters
    ----------
    df_sorted : per-sample dataset (one row per vehicle per timestep),
        must contain veh_col, time_col, cell_col.
    runs : output of `build_trajectories`.
    window_s : how many seconds ahead of a confirmed handover to start
        flagging migration=1/2. This is the main tunable parameter -
        the original study swept 2, 3, 4 and 5 seconds.
    tol_s : tolerance subtracted from window_s when deciding whether a
        run counts as "stable" (a run is stable if duration >= window_s - tol_s).
    announce_policy : "strict" (recommended) looks at the previous and
        current run duration before declaring a handover; "lookback"
        only requires the destination run to be stable.
    handle_bursts : when True (strict policy only), also detects and
        classifies handovers that are preceded by one or more short
        ("unstable") runs, instead of only detecting stable->stable
        transitions.
    destination_mode : "freeze_stable" or "serving", see above.
    log_all_cases : if True, also returns a second events table with
        every detected case (including non-actionable ones like
        ping-pong) tagged for downstream analysis/visualization.

    Returns
    -------
    (out, events, events_all)
        out         : df_sorted + ["migration", "destination"]
        events      : actionable handover events actually applied to `out`
        events_all  : every detected case, for diagnostics/plots
                      (empty DataFrame if log_all_cases=False)

    Event cases logged in events_all["case"]
    -----------------------------------------
    C1_handover_normal          stable -> stable, direct transition
    C2a_ABC                     stable -> (short runs only) -> different stable
    C2b_pingpong                stable -> (short runs only) -> same stable
                                 (not a real handover)
    C3_handover_sin_historico   first stable run reached after only short
                                runs since the start of the recording
                                (no earlier stable "from" cell to compare to)
    C4_no_estable                entering a short/unstable run right after a
                                stable one (kept for diagnostics only)
    """
    out = df_sorted.sort_values([veh_col, time_col], kind="mergesort").reset_index(drop=True).copy()
    if not pd.api.types.is_integer_dtype(out[cell_col]):
        out[cell_col] = pd.to_numeric(out[cell_col], errors="coerce").astype("Int64")
    out["migration"] = pd.Series(0, index=out.index, dtype="Int8")
    out["destination"] = out[cell_col].astype("Int64")

    r = runs.sort_values([veh_col, "run_id"], kind="mergesort").copy()
    needed = {"start_time", "end_time", "servingCell", "duration", "run_id", veh_col}
    missing = needed - set(r.columns)
    if missing:
        raise ValueError(f"'runs' is missing required columns: {missing}")

    thr = max(0.0, float(window_s) - float(tol_s))

    events = []       # actionable handovers only (used to label `out`)
    events_all = []   # every detected case (diagnostics)

    def _log_case(veh, t_change, from_cell, to_cell, case, policy=None, note=None):
        if not log_all_cases:
            return
        events_all.append({
            veh_col: veh,
            "t_change": float(t_change),
            "from_cell": int(from_cell) if pd.notna(from_cell) else pd.NA,
            "to_cell": int(to_cell) if pd.notna(to_cell) else pd.NA,
            "case": case,
            "policy_src": policy,
            "thr": float(thr),
            "window_s": float(window_s),
            "tol_s": float(tol_s),
            "note": note,
        })

    if announce_policy == "strict":
        # (a) classic strict handovers: both the previous and the new run
        # are long enough to count as "stable".
        r["prev_duration"] = r.groupby(veh_col, sort=False)["duration"].shift()
        r["prev_cell"] = r.groupby(veh_col, sort=False)["servingCell"].shift()
        cand = r[r["prev_duration"].notna()].copy()
        valid = cand[(cand["prev_duration"] >= thr) & (cand["duration"] >= thr)].copy()

        for _, row in valid.iterrows():
            prev_cell = int(row["prev_cell"])
            to_cell = int(row["servingCell"])
            t_change = float(row["start_time"])
            veh = row[veh_col]
            if prev_cell != to_cell:
                _log_case(veh, t_change, prev_cell, to_cell, "C1_handover_normal", policy="strict")
                events.append({veh_col: veh, "t_change": t_change, "from_cell": prev_cell,
                                "to_cell": to_cell, "policy": "strict"})

        if handle_bursts:
            # (b) stable -> (only short runs) -> stable: either a delayed
            # handover (different cell) or a ping-pong (same cell).
            for veh, grp in r.groupby(veh_col, sort=False):
                grp = grp.reset_index(drop=True)
                prev_stable_idx = None
                for i in range(len(grp)):
                    dur_i = grp.loc[i, "duration"]
                    cell_i = int(grp.loc[i, "servingCell"])

                    if prev_stable_idx is not None and pd.notna(dur_i) and dur_i < thr:
                        prev_cell = int(grp.loc[prev_stable_idx, "servingCell"])
                        _log_case(veh, float(grp.loc[i, "start_time"]), prev_cell, cell_i,
                                   "C4_no_estable", policy="strict_burst_scan",
                                   note="enter_short_after_stable")

                    if pd.notna(dur_i) and dur_i >= thr:
                        if prev_stable_idx is not None:
                            inter = grp.loc[prev_stable_idx + 1:i - 1] if i - 1 >= prev_stable_idx + 1 else None
                            has_inter = (inter is not None) and (len(inter) > 0)
                            only_short = has_inter and bool((inter["duration"] < thr).all())
                            prev_cell = int(grp.loc[prev_stable_idx, "servingCell"])

                            if only_short:
                                t_change = float(grp.loc[i, "start_time"])
                                if cell_i != prev_cell:
                                    _log_case(veh, t_change, prev_cell, cell_i, "C2a_ABC", policy="strict_burst")
                                    events.append({veh_col: veh, "t_change": t_change, "from_cell": prev_cell,
                                                    "to_cell": cell_i, "policy": "strict_burst"})
                                else:
                                    _log_case(veh, t_change, prev_cell, cell_i, "C2b_pingpong",
                                               policy="strict_burst", note="stable->shorts->same_stable")
                        prev_stable_idx = i

            # (c) first stable run of the recording, reached only via short
            # runs from the very start (no earlier stable run to compare to).
            for veh, grp in r.groupby(veh_col, sort=False):
                grp = grp.reset_index(drop=True)
                stable_idx = next(
                    (i for i in range(len(grp)) if pd.notna(grp.loc[i, "duration"]) and grp.loc[i, "duration"] >= thr),
                    None,
                )
                if stable_idx is None or stable_idx == 0:
                    continue
                before = grp.loc[0:stable_idx - 1]
                if len(before) and bool((before["duration"] < thr).all()):
                    to_cell = int(grp.loc[stable_idx, "servingCell"])
                    from_cell = int(grp.loc[0, "servingCell"])
                    t_change = float(grp.loc[stable_idx, "start_time"])
                    if to_cell != from_cell:
                        _log_case(veh, t_change, from_cell, to_cell, "C3_handover_sin_historico",
                                   policy="strict_startburst")
                        events.append({veh_col: veh, "t_change": t_change, "from_cell": from_cell,
                                        "to_cell": to_cell, "policy": "strict_startburst"})
                    else:
                        _log_case(veh, t_change, from_cell, to_cell, "C2b_pingpong",
                                   policy="strict_startburst", note="start_shorts->same_stable")

    elif announce_policy == "lookback":
        stable = r[r["duration"] >= thr].copy()
        stable["prev_run_id"] = stable.groupby(veh_col, sort=False)["run_id"].shift()
        stable["from_cell"] = stable.groupby(veh_col, sort=False)["servingCell"].shift()
        valid = stable[stable["prev_run_id"].notna()].copy()
        for _, row in valid.iterrows():
            from_cell = int(row["from_cell"])
            to_cell = int(row["servingCell"])
            t_change = float(row["start_time"])
            veh = row[veh_col]
            if from_cell != to_cell:
                _log_case(veh, t_change, from_cell, to_cell, "C1_handover_normal", policy="lookback")
                events.append({veh_col: veh, "t_change": t_change, "from_cell": from_cell,
                                "to_cell": to_cell, "policy": "lookback"})
    else:
        raise ValueError("announce_policy must be 'strict' or 'lookback'.")

    events_df = (
        pd.DataFrame(events).sort_values([veh_col, "t_change"], kind="mergesort").reset_index(drop=True)
        if events else pd.DataFrame(columns=[veh_col, "t_change", "from_cell", "to_cell", "policy"])
    )
    events_all_df = (
        pd.DataFrame(events_all).sort_values([veh_col, "t_change"], kind="mergesort").reset_index(drop=True)
        if (log_all_cases and events_all)
        else pd.DataFrame(columns=[veh_col, "t_change", "from_cell", "to_cell", "case", "policy_src", "thr",
                                     "window_s", "tol_s", "note"])
    )

    # -------- apply the pre-handover warning windows --------
    for veh, sub in events_df.groupby(veh_col):
        mask_veh = out[veh_col].eq(veh)
        idx_veh = out.index[mask_veh]
        if len(idx_veh) == 0:
            continue
        t_veh = out.loc[idx_veh, time_col].to_numpy()

        for _, ev in sub.iterrows():
            t_change = float(ev["t_change"])
            to_cell = int(ev["to_cell"])
            in_win = (t_veh > (t_change - window_s)) & (t_veh <= t_change)
            if not in_win.any():
                continue
            win_idx = idx_veh[in_win]
            out.loc[win_idx, "destination"] = pd.Series(to_cell, index=win_idx, dtype="Int64")
            out.loc[win_idx, "migration"] = pd.Series(2, index=win_idx, dtype="Int8")
            out.at[win_idx[0], "migration"] = 1

    # -------- resolve destination outside of warning windows --------
    if destination_mode == "freeze_stable":
        st = r[r["duration"] >= thr][[veh_col, "start_time", "end_time", "servingCell"]]

        out["__short_run__"] = False
        for veh, grp in r.groupby(veh_col, sort=False):
            mask_veh = out[veh_col].eq(veh)
            idx_veh = out.index[mask_veh]
            if len(idx_veh) == 0:
                continue
            t_veh = out.loc[idx_veh, time_col].to_numpy()
            for _, row in r[(r[veh_col] == veh) & (r["duration"] < thr)].iterrows():
                in_short = (t_veh >= row["start_time"]) & (t_veh <= row["end_time"])
                if in_short.any():
                    out.loc[idx_veh[in_short], "__short_run__"] = True

        for veh, subdf in out.groupby(veh_col, sort=False):
            idx = subdf.index
            t = subdf[time_col].to_numpy()
            st_veh = st[st[veh_col] == veh]

            base_fwd = pd.Series(pd.array([pd.NA] * len(idx), dtype="Int64"), index=idx)
            for _, row in st_veh.iterrows():
                in_stable = (t >= row["start_time"]) & (t <= row["end_time"])
                if in_stable.any():
                    base_fwd.loc[idx[in_stable]] = int(row["servingCell"])
            base_fwd = base_fwd.ffill()

            base_bwd = pd.Series(pd.array([pd.NA] * len(idx), dtype="Int64"), index=idx)
            for _, row in st_veh.iterrows():
                in_stable = (t >= row["start_time"]) & (t <= row["end_time"])
                if in_stable.any():
                    base_bwd.loc[idx[in_stable]] = int(row["servingCell"])
            base_bwd = base_bwd.bfill()

            cand = (out.loc[idx, "migration"] == 0) & (out.loc[idx, "__short_run__"])
            if cand.any():
                val = base_fwd.loc[idx[cand]].fillna(base_bwd.loc[idx[cand]]).fillna(out.loc[idx[cand], cell_col])
                out.loc[idx[cand], "destination"] = val.astype("Int64")

        out.drop(columns="__short_run__", inplace=True)

    elif destination_mode != "serving":
        raise ValueError("destination_mode must be 'freeze_stable' or 'serving'.")

    out["migration"] = out["migration"].astype("Int8")
    out["destination"] = out["destination"].astype("Int64")
    return out, events_df, events_all_df
