import itertools, numbers, os

import pandas as pd


def _norm(vals):
    # Round numbers, so values read back from the CSV match the ones in the grid
    return tuple(round(float(v), 8) if isinstance(v, numbers.Real) and not isinstance(v, bool) else v
                 for v in vals)


def grid_search(run_fn, grid, csv_path, on_row=None):
    # run_fn(cfg) -> (row dict, fitted model). Skips the configurations already in the CSV,
    # so an interrupted search resumes where it stopped. Returns ALL rows of the CSV.
    # NOTE: avoid None in a grid (it becomes NaN in the CSV and the run would be repeated).
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    rows = pd.read_csv(csv_path).to_dict("records") if os.path.exists(csv_path) else []
    done = {_norm(r.get(k) for k in grid) for r in rows}

    for vals in itertools.product(*grid.values()):
        if _norm(vals) in done:
            continue
        cfg = dict(zip(grid.keys(), vals))
        row, model = run_fn(cfg)
        rows.append(row)
        done.add(_norm(vals))
        pd.DataFrame(rows).to_csv(csv_path, index=False)   # saved after every run
        if on_row:
            on_row(row, model)
        print(row)
    return pd.DataFrame(rows)


def best_row(df, metric="val_acc", **where):
    # Best row of the results table, optionally restricted (e.g. wd=0.0)
    d = df
    for k, v in where.items():
        d = d[d[k] == v]
    return d.loc[d[metric].idxmax()]


def keep_best(save_fn, csv_path=None, metric="val_acc"):
    # Callback for grid_search: calls save_fn(row, model) only when the metric improves.
    # With csv_path, runs already in the CSV count too (no overwrite after a resume).
    best = {"v": float("-inf")}
    if csv_path and os.path.exists(csv_path):
        best["v"] = pd.read_csv(csv_path)[metric].max()

    def cb(row, model):
        if row[metric] > best["v"]:
            best["v"] = row[metric]
            save_fn(row, model)
    return cb
