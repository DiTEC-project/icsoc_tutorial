import itertools

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

import preprocess


def _installed(*families):
    available = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in families if f in available]

# thesis palette
RED, RED_EDGE = "#BE503E", "#963C30"
DARK, DARK_EDGE = "#5E5A57", "#3B3836"
BROWN, PINK, BEIGE, GRID, BADGE = "#9C7B5C", "#E9C7C0", "#F4EEEA", "#C2B2AB", "#EEEEEE"
FADED = "#6C6A68"  # readable grey for de-emphasised text (>= 4.5:1 contrast)
SANS = _installed("Liberation Sans", "Arial", "DejaVu Sans")
SERIF = _installed("Liberation Serif", "Times New Roman", "DejaVu Serif")

TABLE_CSS = [
    {"selector": "", "props": f"border-collapse:collapse; border:2px solid #000; font-family:'Liberation Sans',Arial,sans-serif; font-size:14px; margin:8px 0"},
    {"selector": "th", "props": f"background:{PINK}; color:#000; font-weight:bold; padding:7px 14px; border:1px solid {GRID}; text-align:center"},
    {"selector": "td", "props": f"background:{BEIGE}; color:#000; padding:6px 14px; border:1px solid {GRID}; text-align:center"},
]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "font.family": "sans-serif", "font.sans-serif": SANS, "font.size": 11,
        "text.color": DARK_EDGE, "axes.labelcolor": DARK_EDGE, "xtick.color": DARK_EDGE, "ytick.color": DARK_EDGE,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titlepad": 14, "legend.frameon": False,
    })


def title(ax, text):
    ax.set_title(text, fontfamily=SERIF, fontsize=15, color="#000")


def month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def show(df, left=(), row_css=None):
    fmt = {c: "{:.1f}" for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    if "timestamp" in df.columns:
        fmt["timestamp"] = lambda t: f"{t:%d %b %H:%M}"
    styler = df.style.hide(axis="index").set_table_styles(TABLE_CSS).format(fmt)
    if left:
        styler = styler.set_table_styles({c: [{"selector": "td", "props": "text-align:left"}] for c in left}, overwrite=False)
    if row_css:
        styler = styler.set_table_styles({i: [{"selector": "td", "props": css}] for i, css in row_css.items()},
                                         axis=1, overwrite=False)
    return styler


# ---------- data ----------

def discretize(table, n_bins=3):
    sensors = [c for c in table.columns if c not in ("timestamp", "leakage")]
    out = pd.DataFrame({c: pd.cut(table[c], n_bins, labels=["low", "mid", "high"]).astype(str) for c in sensors})
    out["leakage"] = table["leakage"].map({1: "yes", 0: "no"})
    return out


def fires(antecedents, data):
    return np.all([data[a["feature"]].values == a["value"] for a in antecedents], axis=0)


def rule_text(antecedents):
    return " AND ".join(f"{a['feature']} = {a['value']}" for a in antecedents)


def rules_table(rules, n=None):
    df = pd.DataFrame({
        "IF": [rule_text(r["antecedents"]) for r in rules],
        "THEN": [f"{r['consequent']['feature']} = {r['consequent']['value']}" for r in rules],
        "support": [r["support"] for r in rules],
        "confidence": [r["confidence"] for r in rules],
    })
    df["n"] = [len(r["antecedents"]) for r in rules]
    df = df.sort_values(["n", "support", "confidence"], ascending=[True, False, False]).drop(columns="n").reset_index(drop=True)
    df = df.head(n) if n else df
    return show(df, left=["IF", "THEN"]).format({"support": "{:.0%}", "confidence": "{:.0%}"})


def rule_sensors(rules):
    return {a["feature"] for r in rules for a in r["antecedents"]}


# ---------- Part 2: rule explosion ----------

def count_possible_rules(data, max_len=3):
    n_values = data.nunique().values
    counts, total = [], 0
    for k in range(1, max_len + 1):
        for ant in itertools.combinations(range(len(n_values)), k):
            total += np.prod(n_values[list(ant)]) * (n_values.sum() - n_values[list(ant)].sum())
        counts.append(int(total))
    return counts


def mine_all_rules(data, max_len=2):
    onehot = pd.get_dummies(data, prefix_sep="=").astype(bool)
    X, items = onehot.values, list(onehot.columns)
    feature = [i.split("=")[0] for i in items]
    base = X.mean(0)
    rows = []
    for k in range(1, max_len + 1):
        for ant in itertools.combinations(range(len(items)), k):
            if len({feature[a] for a in ant}) < k:
                continue
            mask = X[:, ant].all(1)
            if not mask.any():
                continue
            conf = X[mask].mean(0)
            for c in np.nonzero(conf)[0]:
                if feature[c] not in {feature[a] for a in ant}:
                    rows.append((" AND ".join(items[a] for a in ant), items[c], mask.mean() * conf[c], conf[c], base[c]))
    return pd.DataFrame(rows, columns=["IF", "THEN", "support", "confidence", "base_rate"])


def plot_rule_explosion(possible, found):
    labels = ["1 condition", "up to 2 conditions", "up to 3 conditions", "up to 2 conditions\nand confidence ≥ 80%"]
    values = possible + [found]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, color=[DARK, DARK, DARK, RED], edgecolor=[DARK_EDGE] * 3 + [RED_EDGE], height=0.58)
    for yi, v in zip(y, values):
        ax.text(v * 1.15, yi, f"{v:,}", va="center", fontsize=12, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlim(1, max(values) * 15)
    ax.set_xlabel("number of rules (log scale)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    title(ax, "Possible rules for 19 columns")
    plt.show()


def explain_examples(data, examples):
    rows = []
    for ant, cons, why in examples:
        ants = [{"feature": f, "value": v} for f, v in ant.items()]
        m = fires(ants, data)
        hit = data[cons[0]].values == cons[1]
        rows.append({"Rule": f"{rule_text(ants)}  →  {cons[0]} = {cons[1]}", "Support": (m & hit).mean(),
                     "Confidence": hit[m].mean(), "Base rate": hit.mean(), "Why": why})
    df = pd.DataFrame(rows)
    row_css = {i: (f"font-weight:bold; color:#000; background:#fff; border-top:2.5px solid {RED}; border-bottom:2.5px solid {RED}"
                   if e[2].startswith("far above") else f"color:{FADED}") for i, e in enumerate(examples)}
    return (show(df, left=["Rule"], row_css=row_css)
            .format({"Support": "{:.0%}", "Confidence": "{:.0%}", "Base rate": "{:.0%}"})
            .set_table_styles({"Why": [{"selector": "td", "props": "font-style:italic"}]}, overwrite=False))


# ---------- network ----------

def _network(ax, scenario_dir, sensors, highlight=()):
    inp = next(scenario_dir.glob("*.inp"))
    pipes, xy = preprocess.read_pipes(inp), preprocess.read_coordinates(inp)
    sensors, highlight = set(sensors), set(highlight)

    for p, (a, b) in pipes.items():
        (x1, y1), (x2, y2) = xy[a], xy[b]
        name = f"pipe_{p}_flow"
        if name in highlight:
            ax.plot([x1, x2], [y1, y2], color=PINK, lw=13, solid_capstyle="round", zorder=1)
        used = name in sensors
        ax.plot([x1, x2], [y1, y2], color=DARK if used else GRID, lw=3.5 if used else 1.6, zorder=2)

    for n, (x, y) in xy.items():
        name = f"node_{n}_pressure"
        if name in highlight:
            ax.scatter(x, y, s=480, color=PINK, zorder=3)
        if n == "1":
            ax.scatter(x, y, s=170, marker="s", color=BROWN, edgecolor=DARK_EDGE, zorder=4)
            ax.annotate("reservoir", (x, y), xytext=(0, -20), textcoords="offset points", ha="center", fontsize=10)
        elif name in sensors:
            ax.scatter(x, y, s=95, color=DARK, edgecolor=DARK_EDGE, zorder=4)
            ax.annotate(n, (x, y), xytext=(-9, 7), textcoords="offset points", ha="right", fontsize=10, fontweight="bold")
        else:
            ax.scatter(x, y, s=28, color=GRID, zorder=4)

    for leak in preprocess.leak_info(scenario_dir):
        x, y = xy[leak["node"]]
        days = (leak["end"] - leak["start"]).total_seconds() / 86400
        duration = f"{days:.0f} days" if days >= 10 else f"{days:.1f} days"
        ax.scatter(x, y, s=700, marker="*", color=RED, edgecolor=RED_EDGE, linewidth=1, zorder=5)
        right_half = x > np.median([p[0] for p in xy.values()])
        ax.annotate(f"leak at node {leak['node']}\n{leak['start']:%d %b} – {leak['end']:%d %b}  ({duration})",
                    (x, y), xytext=(26, 0) if right_half else (0, -34), textcoords="offset points",
                    ha="left" if right_half else "center", va="center" if right_half else "top",
                    fontsize=10, color="#000", linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.5", fc="#fff", ec=RED, lw=1.5), zorder=6)
    ax.set_aspect("equal")
    ax.margins(0.06)
    ax.axis("off")


def _network_legend(fig, highlight, ax=None):
    handles = [Line2D([], [], marker="*", ls="", ms=18, color=RED, mec=RED_EDGE, label="leak location"),
               Line2D([], [], marker="o", ls="", ms=9, color=DARK, label="pressure sensor"),
               Line2D([], [], color=DARK, lw=3.5, label="flow sensor (pipe)"),
               Line2D([], [], marker="o", ms=5, color=GRID, lw=1.6, label="not used")]
    if highlight:
        handles.append(Line2D([], [], marker="o", ls="", ms=16, color=PINK, label="used in a leak rule"))
    kw = dict(handles=handles, ncol=len(handles), fontsize=10.5, handletextpad=0.6, columnspacing=2.2)
    if ax is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0), **kw)
    else:
        fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), **kw)


def draw_network(scenario_dir, sensors, highlight=(), title_text="Hanoi water network (LeakDB, Scenario 10)"):
    fig, ax = plt.subplots(figsize=(9, 7))
    _network(ax, scenario_dir, sensors, highlight)
    title(ax, title_text)
    _network_legend(fig, highlight, ax=ax)
    plt.show()


def compare_networks(scenario_dir, sensors, rule_sets):
    fig, axes = plt.subplots(1, len(rule_sets), figsize=(8 * len(rule_sets), 6.4))
    for ax, (name, rules) in zip(axes, rule_sets.items()):
        _network(ax, scenario_dir, sensors, rule_sensors(rules))
        title(ax, name)
    _network_legend(fig, True)
    plt.show()


# ---------- time series ----------

def shade_leaks(ax, scenario_dir):
    for i, leak in enumerate(preprocess.leak_info(scenario_dir)):
        ax.axvspan(leak["start"], leak["end"], color=RED, alpha=0.12, lw=0, label="leak active" if i == 0 else None)


UNITS = {"pressure": "m", "flow": "m³/h"}


def plot_aggregation(raw, table, sensors, scenario_dir, start="2017-05-08", end="2017-06-05"):
    sensors = [sensors] if isinstance(sensors, str) else sensors
    fig, axes = plt.subplots(len(sensors), 1, figsize=(11, 2.9 * len(sensors)), sharex=True, squeeze=False)
    for i, (ax, sensor) in enumerate(zip(axes[:, 0], sensors)):
        r = raw[sensor].loc[start:end]
        t = table.set_index("timestamp")[sensor].loc[start:end]
        ax.plot(r.index, r.values, color=GRID, lw=1.2, label="raw reading (every 30 min)")
        ax.step(t.index, t.values, where="post", color=DARK, lw=2.2, label="6-hour average (= one table row)")
        shade_leaks(ax, scenario_dir)
        ax.set_xlim(r.index[0], r.index[-1])
        ax.set_ylabel(UNITS[sensor.rsplit("_", 1)[1]])
        title(ax, sensor)
        if i == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.18), ncol=3, fontsize=10.5)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout(h_pad=2.5)
    plt.show()


def _two_lane(ax, t, top, bottom, labels, top_color=DARK, bottom_color=RED):
    ax.fill_between(t, 1.15, 1.15 + 0.75 * top, step="post", color=top_color, lw=0)
    ax.fill_between(t, 0, 0.75 * bottom, step="post", color=bottom_color, lw=0)
    ax.axhline(1.15, color=GRID, lw=0.8)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_yticks([0.375, 1.525], labels[::-1])
    ax.set_ylim(-0.1, 2.0)
    ax.tick_params(axis="y", length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)


def plot_rule_timeline(data, timestamps, rule):
    ant = fires(rule["antecedents"], data)
    hit = data[rule["consequent"]["feature"]].values == rule["consequent"]["value"]
    t = pd.to_datetime(timestamps).values
    fig, ax = plt.subplots(figsize=(11, 2.4))
    _two_lane(ax, t, ant, hit, [f"IF  {rule_text(rule['antecedents'])}",
                                f"THEN  {rule['consequent']['feature']} = {rule['consequent']['value']}"])
    ax.set_xlim(t[0], t[-1])
    month_axis(ax)
    title(ax, f"support {(ant & hit).mean():.0%}      confidence {hit[ant].mean():.0%}")
    plt.show()


def plot_bottleneck(results):
    sizes = [str(k) for k in results]
    counts = [len(r) for r in results.values()]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(sizes, counts, color=DARK, edgecolor=DARK_EDGE, width=0.5)
    for x, c in zip(sizes, counts):
        ax.text(x, c + max(counts) * 0.03, str(c), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.set_xlabel("bottleneck size (neurons)")
    ax.set_ylabel("number of leak rules")
    title(ax, "Smaller bottleneck → fewer rules")
    plt.show()


def plot_split(timestamps, test_mask, table):
    t = pd.to_datetime(timestamps).values
    fig, ax = plt.subplots(figsize=(11, 2.2))
    _two_lane(ax, t, test_mask, table["leakage"].values == 1, ["test weeks", "leak active"], top_color=BROWN)
    ax.set_xlim(t[0], t[-1])
    month_axis(ax)
    title(ax, "Learn on 3 of every 4 weeks, test on the 4th")
    plt.show()


def plot_train_vs_test(rules, test_data):
    y = test_data["leakage"].values == "yes"
    points = {}
    for r in rules:
        m = fires(r["antecedents"], test_data)
        if m.any():
            key = (round(r["confidence"], 2), round(y[m].mean(), 2))
            points[key] = points.get(key, 0) + 1
    xs, ys, ns = zip(*[(a, b, n) for (a, b), n in points.items()])
    lo = np.floor((min(xs + ys) - 0.03) * 20) / 20
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([lo, 1], [lo, 1], color=GRID, ls="--", lw=1.2, zorder=1)
    ax.scatter(xs, ys, s=[60 + 25 * n for n in ns], color=DARK, alpha=0.8, edgecolor="#fff", lw=1, zorder=2)
    for x, yy, n in zip(xs, ys, ns):
        if n > 1:
            ax.annotate(f"{n} rules", (x, yy), xytext=(-14, 0), textcoords="offset points", ha="right", va="center", fontsize=10)
    ax.text(1.0, lo + 0.01, "dashed line: equally confident\non training and test weeks", color=FADED, fontsize=10,
            ha="right", va="bottom", linespacing=1.4)
    ax.set_xlim(lo, 1.02)
    ax.set_ylim(lo, 1.02)
    ax.set_xticks(np.arange(lo, 1.001, 0.05))
    ax.set_yticks(np.arange(lo, 1.001, 0.05))
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("confidence on training weeks")
    ax.set_ylabel("confidence on test weeks")
    title(ax, f"{sum(ns)} leak rules on unseen weeks")
    plt.show()


def plot_alarms(timestamps, test_data, alarm, year_start="2017-01-01", year_end="2017-12-31"):
    t = pd.to_datetime(timestamps).values
    y = test_data["leakage"].values == "yes"
    fig, ax = plt.subplots(figsize=(11, 2.4))
    width = pd.Timedelta("6h")
    ax.bar(t, 0.75 * alarm, bottom=1.15, width=width, align="edge", color=DARK)
    ax.bar(t, 0.75 * y, width=width, align="edge", color=RED)
    ax.axhline(1.15, color=GRID, lw=0.8)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_yticks([0.375, 1.525], ["leak active", "a rule fired"])
    ax.set_ylim(-0.1, 2.0)
    ax.tick_params(axis="y", length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(pd.Timestamp(year_start), pd.Timestamp(year_end))
    month_axis(ax)
    title(ax, f"Test weeks:  {y[alarm].mean():.0%} of alarms are real leaks,  "
              f"{alarm[y].mean():.0%} of leak windows raise an alarm")
    plt.show()
