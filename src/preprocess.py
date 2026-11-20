import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAKDB_URL = "https://raw.githubusercontent.com/KIOS-Research/LeakDB/master/CCWI-WDSA2018/Benchmarks/Hanoi_CMH.zip"


def download(scenario="Scenario-10", data_dir=ROOT / "data" / "leakdb"):
    if (data_dir / scenario).exists():
        return data_dir / scenario
    with urllib.request.urlopen(LEAKDB_URL) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))
    archive.extractall(data_dir, members=[m for m in archive.namelist() if m.startswith(scenario + "/")])
    return data_dir / scenario


def read_series(path, column="Value"):
    return pd.read_csv(path, index_col="Timestamp", parse_dates=True)[column]


def read_pipes(inp_file):
    section = inp_file.read_text().split("[PIPES]")[1].split("[")[0]
    rows = [line.split() for line in section.splitlines()]
    return {r[0]: (r[1], r[2]) for r in rows if len(r) > 2 and not r[0].startswith(";")}


def read_coordinates(inp_file):
    section = inp_file.read_text().split("[COORDINATES]")[1].split("[")[0]
    rows = [line.split() for line in section.splitlines()]
    return {r[0]: (float(r[1]), float(r[2])) for r in rows if len(r) == 3 and not r[0].startswith(";")}


def leak_info(scenario_dir):
    leaks = []
    for f in (scenario_dir / "Leaks").glob("Leak_*_info.csv"):
        info = dict(line.split(" , ") for line in f.read_text().strip().splitlines()[1:])
        leaks.append({"node": info["Leak Node"], "type": info["Leak Type"],
                      "start": pd.Timestamp(info["Leak Start"]), "end": pd.Timestamp(info["Leak End"])})
    return sorted(leaks, key=lambda l: l["start"])


def leak_nodes(scenario_dir):
    return [l["node"] for l in leak_info(scenario_dir)]


def hops_from(sources, pipes):
    neighbours = {}
    for a, b in pipes.values():
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    dist, frontier = {n: 0 for n in sources}, list(sources)
    while frontier:
        nxt = []
        for n in frontier:
            for m in neighbours[n] - dist.keys():
                dist[m] = dist[n] + 1
                nxt.append(m)
        frontier = nxt
    return dist


def sensors_near_leaks(scenario_dir, hops):
    pipes = read_pipes(next(scenario_dir.glob("*.inp")))
    dist = hops_from(leak_nodes(scenario_dir), pipes)
    nodes = {n for n, d in dist.items() if d <= hops}
    links = {p for p, (a, b) in pipes.items() if a in nodes and b in nodes}
    return [f"node_{n}_pressure" for n in nodes] + [f"pipe_{p}_flow" for p in links]


def read_sensors(scenario_dir):
    sensors = {}
    for folder, prefix in [("Pressures", "node_{}_pressure"), ("Flows", "pipe_{}_flow")]:
        for f in sorted((scenario_dir / folder).glob("*.csv"), key=lambda p: int(p.stem.split("_")[1])):
            sensors[prefix.format(f.stem.split("_")[1])] = read_series(f)
    return pd.DataFrame(sensors)


def build_table(scenario_dir, window="6h", hops=2):
    X = read_sensors(scenario_dir)
    X = X.loc[:, X.std() > 0]  # drop constant sensors (e.g. reservoir node)
    if hops >= 0:
        keep = set(sensors_near_leaks(scenario_dir, hops))
        X = X[[c for c in X.columns if c in keep]]

    labels = read_series(scenario_dir / "Labels.csv", "Label")
    table = X.resample(window).mean()
    table["leakage"] = labels.resample(window).max().astype(int)
    table.index.name = "timestamp"
    return table.reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="Scenario-10")
    parser.add_argument("--window", default="6h", help="pandas resample rule, e.g. 1h, 6h, 1D")
    parser.add_argument("--hops", default=2, type=int, help="keep sensors within this many hops of a leak; -1 keeps all")
    parser.add_argument("--data-dir", default=ROOT / "data" / "leakdb", type=Path)
    parser.add_argument("--out", default=ROOT / "data" / "leakdb_preprocessed.csv", type=Path)
    args = parser.parse_args()

    table = build_table(download(args.scenario, args.data_dir), args.window, args.hops)
    table.to_csv(args.out, index=False)
    print(f"{args.scenario} @ {args.window}: {table.shape[0]} rows, {table.shape[1] - 2} sensors, "
          f"leakage rate {table['leakage'].mean():.2f} -> {args.out}")
