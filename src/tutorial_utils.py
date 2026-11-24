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


def show(df, left=(), row_css=None, numbered=True):
    df = df.copy()
    if numbered:
        df.insert(0, "#", df.index + 1 if pd.api.types.is_integer_dtype(df.index) else range(1, len(df) + 1))
    fmt = {c: "{:.1f}" for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    if "timestamp" in df.columns:
        fmt["timestamp"] = lambda t: f"{t:%d %b %H:%M}"
    styler = (df.style.hide(axis="index").set_table_styles(TABLE_CSS).format(fmt)
              .set_table_styles({"#": [{"selector": "td", "props": f"color:{FADED}"}]} if "#" in df else {},
                                overwrite=False))
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


def sort_rules(rules):
    return sorted(rules, key=lambda r: (len(r["antecedents"]), -r["support"], -r["confidence"]))


def rule_at(rules, row):
    return sort_rules(rules)[row - 1]


def rules_table(rules, n=None):
    rules = sort_rules(rules)
    df = pd.DataFrame({
        "IF": [rule_text(r["antecedents"]) for r in rules],
        "THEN": [f"{r['consequent']['feature']} = {r['consequent']['value']}" for r in rules],
        "support": [r["support"] for r in rules],
        "confidence": [r["confidence"] for r in rules],
    })
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


def explain_examples(data, examples, highlight=None):
    rows = []
    for ant, cons, why in examples:
        ants = [{"feature": f, "value": v} for f, v in ant.items()]
        m = fires(ants, data)
        hit = data[cons[0]].values == cons[1]
        rows.append({"Rule": f"{rule_text(ants)}  →  {cons[0]} = {cons[1]}", "Support": (m & hit).mean(),
                     "Confidence": hit[m].mean(), "Base rate": hit.mean(), "Why": why})
    df = pd.DataFrame(rows)
    row_css = {i: (f"font-weight:bold; color:#000; background:#fff; border-top:2.5px solid {RED}; border-bottom:2.5px solid {RED}"
                   if i + 1 == highlight else f"color:{FADED}") for i in range(len(examples))}
    return (show(df, left=["Rule"], row_css=row_css)
            .format({"Support": "{:.0%}", "Confidence": "{:.0%}", "Base rate": "{:.0%}"})
            .set_table_styles({"Why": [{"selector": "td", "props": "font-style:italic"}]}, overwrite=False))


# ---------- network ----------

NODE_LABEL_RIGHT = {"11"}  # Hanoi: keeps n11 clear of the p12 label


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
        if used:
            horizontal = abs(x2 - x1) > abs(y2 - y1)
            ax.annotate(f"p{p}", ((x1 + x2) / 2, (y1 + y2) / 2), xytext=(0, -9) if horizontal else (9, 0),
                        textcoords="offset points", ha="center" if horizontal else "left",
                        va="top" if horizontal else "center", color=BROWN, fontsize=9.5, fontweight="bold", zorder=6)

    for n, (x, y) in xy.items():
        name = f"node_{n}_pressure"
        if name in highlight:
            ax.scatter(x, y, s=480, color=PINK, zorder=3)
        if n == "1":
            ax.scatter(x, y, s=170, marker="s", color=BROWN, edgecolor=DARK_EDGE, zorder=4)
            ax.annotate("reservoir", (x, y), xytext=(0, -20), textcoords="offset points", ha="center", fontsize=10)
        elif name in sensors:
            ax.scatter(x, y, s=95, color=DARK, edgecolor=DARK_EDGE, zorder=4)
            right = n in NODE_LABEL_RIGHT
            ax.annotate(f"n{n}", (x, y), xytext=(10, 0) if right else (-9, 7), textcoords="offset points",
                        ha="left" if right else "right", va="center" if right else "baseline", fontsize=10,
                        fontweight="bold")
        else:
            ax.scatter(x, y, s=28, color=GRID, zorder=4)

    for leak in preprocess.leak_info(scenario_dir):
        x, y = xy[leak["node"]]
        days = (leak["end"] - leak["start"]).total_seconds() / 86400
        duration = f"{days:.0f} days" if days >= 10 else f"{days:.1f} days"
        ax.scatter(x, y, s=420, facecolor="none", edgecolor=RED, linewidth=3, zorder=5)
        right_half = x > np.median([p[0] for p in xy.values()])
        ax.annotate(f"leak at junction {leak['node']}\n{leak['start']:%d %b} – {leak['end']:%d %b}  ({duration})",
                    (x, y), xytext=(28, 4) if right_half else (0, -34), textcoords="offset points",
                    ha="left" if right_half else "center", va="bottom" if right_half else "top",
                    fontsize=10, color="#000", linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.5", fc="#fff", ec=RED, lw=1.5), zorder=6)
    ax.set_aspect("equal")
    ax.margins(0.06)
    ax.axis("off")


def _network_legend(fig, highlight, ax=None):
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=DARK, label="n28 = pressure sensor at junction 28"),
               Line2D([], [], color=DARK, lw=3.5, label="p29 = flow sensor in pipe 29"),
               Line2D([], [], marker="o", ls="", ms=14, mfc="none", mec=RED, mew=2.5, label="junction with a leak"),
               Line2D([], [], marker="o", ms=5, color=GRID, lw=1.6, label="junction / pipe without sensor"),
               Line2D([], [], marker="s", ls="", ms=10, color=BROWN, mec=DARK_EDGE, label="reservoir (water source)")]
    if highlight:
        handles.append(Line2D([], [], marker="o", ls="", ms=16, color=PINK, label="used in a leak rule"))
    kw = dict(handles=handles, ncol=3, fontsize=10.5, handletextpad=0.6, columnspacing=2.0, labelspacing=0.9)
    if ax is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0), **kw)
    else:
        fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), **kw)


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


ALARM_HTML = r"""
<div id="ta-__UID__" style="position:relative; width:960px; font-family:'Liberation Sans',Arial,sans-serif; color:#3B3836">
  <div class="ta-head" style="font-family:'Liberation Serif','Times New Roman',serif; font-size:19px; color:#000; margin:4px 0 10px; white-space:pre"></div>
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px">
    <button class="ta-play" style="width:84px; padding:5px 0; font-size:14px; border:1.5px solid #5E5A57; border-radius:6px; background:#F4EEEA; cursor:pointer">&#9654; play</button>
    <input class="ta-slider" type="range" min="0" value="0" style="flex:1; accent-color:#9C7B5C">
    <select class="ta-speed" style="font-size:13px; padding:3px">
      <option value="300">slow</option><option value="120" selected>normal</option><option value="40">fast</option>
    </select>
  </div>
  <svg class="ta-svg" width="960" height="175" shape-rendering="crispEdges"></svg>
  <div class="ta-status" style="font-size:15px; font-weight:bold; margin:2px 0 0 110px; min-height:22px; white-space:pre"></div>
  <div style="font-size:13px; color:#6C6A68; margin:4px 0 0 110px">Click a dark alarm bar to see every rule that raised it.</div>
  <div class="ta-pop" style="display:none; position:absolute; left:110px; top:60px; width:760px; max-height:330px; overflow-y:auto;
       background:#fff; border:2px solid #000; border-radius:8px; box-shadow:0 6px 20px rgba(0,0,0,.25); padding:14px 18px; z-index:10"></div>
</div>
<script>
(function () {
  var D = __DATA__;
  var root = document.getElementById("ta-__UID__");
  var svg = root.querySelector(".ta-svg"), head = root.querySelector(".ta-head"), status = root.querySelector(".ta-status");
  var pop = root.querySelector(".ta-pop"), slider = root.querySelector(".ta-slider"), play = root.querySelector(".ta-play");
  var speed = root.querySelector(".ta-speed");
  var C = {dark: "#5E5A57", red: "#BE503E", grid: "#C2B2AB", brown: "#9C7B5C", faded: "#6C6A68"};
  var NS = "http://www.w3.org/2000/svg", n = D.windows.length, L = 110, PW = 960 - L - 20, bw = PW / n;
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e);
    return e;
  }
  function txt(x, y, s, attrs) { var t = el("text", Object.assign({x: x, y: y, "font-size": 13, fill: "#3B3836"}, attrs || {})); t.textContent = s; return t; }
  function num(v) { return v.toLocaleString("en-US", {minimumFractionDigits: 1, maximumFractionDigits: 1}); }
  function pct(v) { return Math.round(v * 100) + "%"; }

  txt(L - 10, 45, "a rule fired", {"text-anchor": "end"});
  txt(L - 10, 110, "leak active", {"text-anchor": "end"});
  el("line", {x1: L, x2: L + PW, y1: 70, y2: 70, stroke: C.grid});
  el("line", {x1: L, x2: L + PW, y1: 135, y2: 135, stroke: C.grid});
  D.weeks.forEach(function (w) {
    if (w[0] > 0) el("line", {x1: L + w[0] * bw, x2: L + w[0] * bw, y1: 15, y2: 135, stroke: C.grid, "stroke-dasharray": "2,3"});
    if (w[1]) txt(L + w[0] * bw, 152, w[1], {"font-size": 11, "text-anchor": "middle"});
  });
  txt(L + PW / 2, 170, "test weeks only, placed side by side", {"font-size": 12, fill: C.faded, "text-anchor": "middle"});
  var alarmBars = [], leakBars = [], targets = [];
  D.windows.forEach(function (w, i) {
    var a = el("rect", {x: L + i * bw, y: 20, width: bw + 0.6, height: 48, fill: C.dark, visibility: "hidden", cursor: "pointer"});
    var hit = el("rect", {x: L + (i + 0.5) * bw - Math.max(bw, 8) / 2, y: 16, width: Math.max(bw, 8), height: 56,
                          fill: "transparent", visibility: "hidden", cursor: "pointer"});
    if (w.fired.length) {
      el("title", {}, hit).textContent = w.t + ", " + w.fired.length + " rule" + (w.fired.length > 1 ? "s" : "") + " fired, click for details";
      hit.addEventListener("click", function () { open(i); });
    }
    alarmBars.push(a); targets.push(hit);
    leakBars.push(el("rect", {x: L + i * bw, y: 85, width: bw + 0.6, height: 48, fill: C.red, visibility: "hidden"}));
  });
  var cursor = el("line", {x1: L, x2: L, y1: 12, y2: 137, stroke: C.brown, "stroke-width": 2});

  function verdict(w) {
    if (w.fired.length) return w.leak ? ["correct alarm: a leak is active", "#3B3836"] : ["false alarm: no leak", C.red];
    return w.leak ? ["missed leak", C.red] : ["all quiet", C.faded];
  }
  function render(pos) {
    pos = Math.max(0, Math.min(pos, n));
    var ok = 0, bad = 0, miss = 0;
    for (var i = 0; i < n; i++) {
      var w = D.windows[i], shown = i < pos;
      alarmBars[i].setAttribute("visibility", shown && w.fired.length ? "visible" : "hidden");
      targets[i].setAttribute("visibility", shown && w.fired.length ? "visible" : "hidden");
      leakBars[i].setAttribute("visibility", shown && w.leak ? "visible" : "hidden");
      if (shown) { if (w.fired.length && w.leak) ok++; else if (w.fired.length) bad++; else if (w.leak) miss++; }
    }
    var x = L + pos * bw;
    cursor.setAttribute("x1", x); cursor.setAttribute("x2", x);
    var cur = D.windows[Math.max(pos - 1, 0)];
    head.textContent = (pos ? cur.t : "start") + "      correct alarms " + ok + "  ·  false alarms " + bad + "  ·  missed leak windows " + miss;
    var v = pos ? verdict(cur) : ["", ""];
    status.textContent = pos ? cur.t.slice(-5) + "   " + v[0] + (cur.fired.length ? "  (" + cur.fired.length + " rule" + (cur.fired.length > 1 ? "s" : "") + " fired)" : "") : "";
    status.style.color = v[1];
    slider.value = pos;
  }
  function open(i) {
    var w = D.windows[i], v = verdict(w), h = [];
    h.push('<div style="display:flex; justify-content:space-between; align-items:baseline">' +
           '<div style="font-size:16px; font-weight:bold; color:' + v[1] + '">' + w.t + ' &nbsp; ' + v[0] + '</div>' +
           '<span class="ta-close" style="cursor:pointer; font-size:22px; color:#3B3836">&times;</span></div>');
    h.push('<div style="font-size:13px; color:' + C.faded + '; margin:2px 0 10px">' + w.fired.length +
           ' rule' + (w.fired.length > 1 ? 's' : '') + ' fired. Support and confidence are measured on the weeks the rules were learned from.</div>');
    w.fired.forEach(function (r, k) {
      var R = D.rules[r];
      h.push('<div style="padding:8px 0; border-top:1px solid ' + C.grid + '">');
      h.push('<div style="font-size:14px; color:#000"><b>' + (k + 1) + '. IF</b> ' + R.if + ' <b>THEN</b> leakage = yes</div>');
      h.push('<div style="font-size:13px; margin:3px 0 4px 18px">support <b>' + pct(R.s) + '</b> &nbsp;·&nbsp; confidence <b>' + pct(R.c) + '</b></div>');
      R.conds.forEach(function (c) {
        h.push('<div style="font-size:13px; color:#3B3836; margin-left:18px">' + c[0] + ' reads <b>' + num(D.read[c[0]][i]) + ' ' + c[2] +
               '</b>, inside ‘' + c[1] + '’ (' + num(c[3]) + ' – ' + num(c[4]) + ' ' + c[2] + ')</div>');
      });
      h.push('</div>');
    });
    pop.innerHTML = h.join("");
    pop.style.display = "block";
    pop.querySelector(".ta-close").addEventListener("click", function () { pop.style.display = "none"; });
  }
  var pos = 0, timer = null;
  function stop() { clearInterval(timer); timer = null; play.innerHTML = "&#9654; play"; }
  function start() {
    if (pos >= n) pos = 0;
    play.innerHTML = "&#10074;&#10074; pause";
    timer = setInterval(function () { pos++; render(pos); if (pos >= n) stop(); }, +speed.value);
  }
  play.addEventListener("click", function () { timer ? stop() : start(); });
  speed.addEventListener("change", function () { if (timer) { stop(); start(); } });
  slider.max = n;
  slider.addEventListener("input", function () { pos = +slider.value; render(pos); });
  render(0);
  root.taOpen = open; root.taRender = function (p) { pos = Math.min(p, n); render(pos); };
})();
</script>
"""


def animate_alarms(table, data, test, rules):
    import json
    import uuid
    from IPython.display import HTML

    t = pd.to_datetime(table["timestamp"][test].values)
    raw = table[test].reset_index(drop=True)
    test_data = data[test].reset_index(drop=True)
    leak = test_data["leakage"].values == "yes"
    ranges = category_ranges(table)
    ordered = sort_rules(rules)
    hits = np.array([fires(r["antecedents"], test_data) for r in ordered])
    week_starts = [0] + [i for i in range(1, len(t)) if t[i] - t[i - 1] > pd.Timedelta("1D")]
    bounds = week_starts + [len(t)]
    payload = {
        "windows": [{"t": f"{t[i]:%d %b %H:%M}", "leak": bool(leak[i]), "fired": np.flatnonzero(hits[:, i]).tolist()}
                    for i in range(len(t))],
        "weeks": [[w, f"{t[w]:%d %b}" if nxt - w >= 12 else ""] for w, nxt in zip(week_starts, bounds[1:])],
        "rules": [{"if": rule_text(r["antecedents"]), "s": r["support"], "c": r["confidence"],
                   "conds": [[a["feature"], a["value"], UNITS[a["feature"].rsplit("_", 1)[1]],
                              *ranges[a["feature"]][a["value"]]] for a in r["antecedents"]]} for r in ordered],
        "read": {f: raw[f].round(1).tolist() for f in rule_sensors(ordered)},
    }
    html = ALARM_HTML.replace("__UID__", uuid.uuid4().hex[:8]).replace("__DATA__", json.dumps(payload))
    return HTML(html)


# ---------- checking alarms ----------

def category_ranges(table, n_bins=3):
    ranges = {}
    for c in table.columns:
        if c in ("timestamp", "leakage"):
            continue
        _, edges = pd.cut(table[c], n_bins, retbins=True)
        ranges[c] = {label: (edges[k], edges[k + 1]) for k, label in enumerate(["low", "mid", "high"])}
    return ranges
