"""Render a trained NEAT genome as a layered network graph (standalone HTML).

    python graph_genome.py models/best_genome.pkl [--schema N] [--out page.html] [--open]

The genome pickle stores only nodes and connections, not which I/O *schema* it
was trained on, so pass the same --schema you trained/play with (default 0);
input labels are read from that schema's ordered input_dictionary
(pickthelock.schemas / observations.FEATURE_MAP) so this keeps working as new
schemas are added. Disabled connection genes and any node with no enabled
connection are dropped. Remaining nodes are placed in layers by shortest
distance from the input layer (inputs first, outputs last), and each output's
exact activation-expression is derived from the live sub-graph.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pickle
import sys
import webbrowser
from collections import deque

# labels come from the game's schema/feature registry, so this file never has to
# hardcode the input vector (which changes as schemas are added).
from pickthelock.schemas import get_schema, SCHEMAS
from pickthelock.observations import NUM_OUTPUTS

ROOT = os.path.dirname(os.path.abspath(__file__))

# Per-schema output names (the genome only knows output indices 0..k-1). Falls
# back to "output N" for schemas not listed here.
OUTPUT_LABELS: dict[int, list[str]] = {
    0: ["target distance", "hold speed", "click"],
}


# --------------------------------------------------------------------------- #
# label helpers: feature key -> human / compact forms

_BAR_PROP = {
    "forward_distance": "fwd dist",
    "reverse_distance": "rev dist",
    "is_blue": "is blue",
    "width": "width",
}
_BAR_CODE = {
    "forward_distance": "fwd",
    "reverse_distance": "rev",
    "is_blue": "blue",
    "width": "w",
}
_GLOBAL_HUMAN = {
    "pick_in_hit_zone_boolean": "in zone",
    "time_remaining_percentage": "time left",
    "boost_multiplier_percentage": "boost mult",
    "penalty_factor_ratio": "penalty",
    "pick_disabled_boolean": "pick disabled",
    "spawn_interval_ratio": "spawn interval",
    "blue_chance_percentage": "blue chance",
    "current_speed_percentage": "speed",
}
_GLOBAL_CODE = {
    "pick_in_hit_zone_boolean": "in_zone",
    "time_remaining_percentage": "time_left",
    "boost_multiplier_percentage": "boost",
    "penalty_factor_ratio": "penalty",
    "pick_disabled_boolean": "pick_off",
    "spawn_interval_ratio": "spawn_int",
    "blue_chance_percentage": "blue_chance",
    "current_speed_percentage": "speed",
}


def _split_bar_key(key: str):
    """('bar_forward_distance_percentage_3') -> ('forward_distance', 3) or None."""
    if not key.startswith("bar_"):
        return None
    body, _, idx = key.rpartition("_")
    if not idx.isdigit():
        return None
    # body is e.g. 'bar_forward_distance_percentage'; strip 'bar_' and the format word
    inner = body[len("bar_"):].rsplit("_", 1)[0]
    return inner, int(idx)


def humanize(key: str) -> str:
    bar = _split_bar_key(key)
    if bar:
        prop, n = bar
        return f"bar {n} · {_BAR_PROP.get(prop, prop)}"
    return _GLOBAL_HUMAN.get(key, key)


def code_label(key: str) -> str:
    bar = _split_bar_key(key)
    if bar:
        prop, n = bar
        return f"bar{n}_{_BAR_CODE.get(prop, prop)}"
    return _GLOBAL_CODE.get(key, key)


# --------------------------------------------------------------------------- #
# number formatting for the derived expressions

def fmt_w(x: float) -> str:
    a = abs(x)
    return f"{a:.4f}" if a < 0.1 else f"{a:.2f}"


def fmt_bias(b: float) -> str:
    return f"{b:+.2f}".replace("+", "+").replace("-", "−")


# --------------------------------------------------------------------------- #

def load_genome(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def build_graph(genome, schema, schema_id: int, num_outputs: int,
                input_labels: dict[int, str]):
    """Return (columns, edges, formulas, meta) ready to serialize to the page."""
    output_keys = set(range(num_outputs))
    all_nodes = genome.nodes                      # int key -> node gene (hidden + outputs)
    hidden_keys = [k for k in all_nodes if k not in output_keys]

    enabled = [(i, o, c.weight) for (i, o), c in genome.connections.items() if c.enabled]

    # nodes actually incident to an enabled connection (drop the rest)
    incident: set[int] = set()
    for i, o, _ in enabled:
        incident.add(i)
        incident.add(o)

    drawn_inputs = sorted((k for k in incident if k < 0), reverse=True)   # -1, -2, ...
    drawn_hidden = [k for k in hidden_keys if k in incident]
    drawn_outputs = [k for k in sorted(output_keys) if k in incident]
    drawn = set(drawn_inputs) | set(drawn_hidden) | set(drawn_outputs)

    edges = [(i, o, w) for (i, o, w) in enabled if i in drawn and o in drawn]

    # incoming enabled edges per node (within the drawn graph)
    incoming: dict[int, list[tuple[int, float]]] = {k: [] for k in drawn}
    outgoing: dict[int, list[int]] = {k: [] for k in drawn}
    for i, o, w in edges:
        incoming[o].append((i, w))
        outgoing[i].append(o)

    # --- layering: shortest distance from the input layer -------------------
    # topological order via Kahn over the drawn DAG (NEAT feed-forward => DAG)
    indeg = {k: len(incoming[k]) for k in drawn}
    queue = deque(sorted(k for k in drawn if indeg[k] == 0))
    topo: list[int] = []
    while queue:
        n = queue.popleft()
        topo.append(n)
        for m in outgoing[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)

    layer: dict[int, int] = {}
    for n in topo:
        if n < 0:
            layer[n] = 0                          # inputs are the source layer
        else:
            preds = [layer[s] for s, _ in incoming[n] if s in layer]
            # min(pred)+1 == shortest distance from inputs; a bias-only hidden
            # node (no incoming) has no path from inputs, so it seeds layer 1.
            layer[n] = (min(preds) + 1) if preds else 1

    hidden_layers = [layer[k] for k in drawn_hidden]
    max_hidden = max(hidden_layers) if hidden_layers else 0
    output_col = max_hidden + 1                   # outputs always sit last

    # --- live set: nodes/edges on a path that reaches an output -------------
    live: set[int] = set()
    stack = list(drawn_outputs)
    live.update(drawn_outputs)
    while stack:
        n = stack.pop()
        for s, _ in incoming[n]:
            if s not in live:
                live.add(s)
                stack.append(s)

    def node_payload(key: int, kind: str) -> dict:
        if kind == "input":
            feat = input_labels[key]
            return {
                "key": str(key), "kind": kind,
                "label": humanize(feat),
                "sub": f"{feat}  ({key})",
                "bias": None, "live": key in live, "biasOnly": False,
            }
        node = all_nodes[key]
        bias_only = len(incoming[key]) == 0
        if kind == "output":
            idx = key
            names = OUTPUT_LABELS.get(schema_id, [])
            oname = names[idx] if idx < len(names) else f"output {idx}"
            label = f"{key} · {oname}"
            sub = f"output {key} · {oname}"
        else:
            label = str(key)
            sub = f"hidden node {key}" + (" · bias-only" if bias_only else "")
        return {
            "key": str(key), "kind": kind, "label": label, "sub": sub,
            "bias": round(node.bias, 4), "live": key in live, "biasOnly": bias_only,
            "act": getattr(node, "activation", "sigmoid"),
            "agg": getattr(node, "aggregation", "sum"),
        }

    # --- assemble columns ---------------------------------------------------
    columns = [{"kind": "input", "title": "inputs",
                "nodes": [node_payload(k, "input") for k in drawn_inputs]}]
    for L in range(1, max_hidden + 1):
        col_nodes = [node_payload(k, "hidden") for k in drawn_hidden if layer[k] == L]
        col_nodes.sort(key=lambda n: n["key"])
        title = "hidden · layer %d" % L if max_hidden > 1 else "hidden"
        columns.append({"kind": "hidden", "title": title, "nodes": col_nodes})
    columns.append({"kind": "output", "title": "outputs",
                    "nodes": [node_payload(k, "output") for k in drawn_outputs]})

    edge_payload = [{"from": str(i), "to": str(o), "w": round(w, 4),
                     "live": (o in live)} for i, o, w in edges]
    max_w = max((abs(w) for _, _, w in edges), default=1.0)

    # --- derived activation-expression per output ---------------------------
    formulas = _derive_formulas(drawn_outputs, all_nodes, incoming, input_labels,
                                schema_id)

    # --- counts for the header / footer -------------------------------------
    hidden_total = len(hidden_keys)
    activation_note = _activation_note(drawn_hidden + drawn_outputs, all_nodes)
    meta = {
        "fitness": getattr(genome, "fitness", None),
        "schema": schema_id,
        "geneCount": len(genome.connections),
        "enabledCount": len(enabled),
        "liveEdges": sum(1 for e in edge_payload if e["live"]),
        "inputsConnected": len(drawn_inputs), "inputsTotal": schema.num_inputs,
        "hiddenConnected": len(drawn_hidden), "hiddenTotal": hidden_total,
        "outputsConnected": len(drawn_outputs), "outputsTotal": num_outputs,
        "omittedGenes": len(genome.connections) - len(enabled),
        "omittedInputs": schema.num_inputs - len(drawn_inputs),
        "omittedHidden": hidden_total - len(drawn_hidden),
        "biasOnly": sorted(str(k) for k in drawn_hidden if len(incoming[k]) == 0),
        "maxWeight": round(max_w, 4),
        "activationNote": activation_note,
    }
    return columns, edge_payload, formulas, meta


def _derive_formulas(outputs, all_nodes, incoming, input_labels, schema_id):
    """Expand each output as a pretty-printed tree of act(bias + sum(w*child)).

    Returns one block of newline-separated lines per output: the bias and every
    signed term on its own line, and each nested activation indented one level
    deeper -- so the box grows vertically instead of scrolling sideways.
    """
    names = OUTPUT_LABELS.get(schema_id, [])
    INDENT = "  "

    def block(key: int, budget: list[int]) -> list[str]:
        """Lines for act(bias + ...); first line is the header, last is ')'."""
        node = all_nodes[key]
        act = getattr(node, "activation", "sigmoid")
        fn = "σ" if act == "sigmoid" else html.escape(act)
        preds = sorted(incoming[key], key=lambda p: -abs(p[1]))
        budget[0] -= len(preds)
        body = [fmt_bias(node.bias)]                # constant term leads
        for src, w in preds:
            prefix = ("−" if w < 0 else "+") + " " + fmt_w(w) + "·"
            if src < 0:                             # input leaf
                body.append(prefix + html.escape(code_label(input_labels[src])))
            elif budget[0] <= 0:                    # keep very large nets bounded
                body.append(prefix + '<span class="hid">n{}</span>'.format(src))
            else:                                   # nested activation: open, recurse
                inner = block(src, budget)
                body.append(prefix + inner[0])      # 'sign coeff.fn('
                body.extend(inner[1:])              # its body + closing paren
        # wrap: header, one-level-indented body, closing paren
        return [fn + "("] + [INDENT + ln for ln in body] + [")"]

    out = []
    for k in outputs:
        oname = names[k] if k < len(names) else "output {}".format(k)
        constant = len(incoming[k]) == 0
        out.append({
            "label": oname,
            "html": "\n".join(block(k, [60])),
            "note": "constant (no inputs reach it)" if constant else "",
        })
    return out


def _activation_note(node_keys, all_nodes) -> str:
    acts = {getattr(all_nodes[k], "activation", "sigmoid") for k in node_keys}
    aggs = {getattr(all_nodes[k], "aggregation", "sum") for k in node_keys}
    if acts == {"sigmoid"} and aggs == {"sum"}:
        return "All nodes use sigmoid activation and sum aggregation; biases are in the tooltips."
    return (f"Activations: {', '.join(sorted(acts))}. Aggregations: {', '.join(sorted(aggs))}. "
            "See tooltips for per-node bias, activation and aggregation.")


# --------------------------------------------------------------------------- #
# page template

def render_page(title: str, subtitle: str, data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return _TEMPLATE.replace("__TITLE__", html.escape(title)) \
                    .replace("__SUBTITLE__", html.escape(subtitle)) \
                    .replace("__DATA__", payload)


def main() -> int:
    ap = argparse.ArgumentParser(description="Graph a NEAT genome as a layered network (HTML).")
    ap.add_argument("model", help="path to a genome .pkl (e.g. models/best_genome.pkl)")
    ap.add_argument("--schema", type=int, default=0,
                    help="I/O schema the genome trained on (default 0); sets input labels")
    ap.add_argument("--out", default=None, help="output .html path (default: alongside the model)")
    ap.add_argument("--no-open", dest="open", action="store_false",
                    help="don't open the page in a browser (it opens by default)")
    ap.add_argument("--title", default=None, help="override the page title")
    args = ap.parse_args()

    if args.schema not in SCHEMAS:
        valid = ", ".join(str(k) for k in sorted(SCHEMAS))
        ap.error(f"--schema {args.schema} unknown; valid schemas: {valid}")
    if not os.path.isfile(args.model):
        ap.error(f"model not found: {args.model}")

    schema = get_schema(args.schema)
    num_outputs = schema.num_outputs
    # NEAT input key -k maps to input_dictionary[k-1]
    input_labels = {-(i + 1): key for i, key in enumerate(schema.input_dictionary)}

    genome = load_genome(args.model)

    # guard against a schema mismatch (genome references an input the schema lacks)
    referenced = [i for (i, _o) in genome.connections for i in (i,) if i < 0]
    if referenced:
        deepest = -min(referenced)
        if deepest > schema.num_inputs:
            ap.error(f"genome references input -{deepest} but schema {args.schema} only "
                     f"defines {schema.num_inputs} inputs; wrong --schema?")

    columns, edges, formulas, meta = build_graph(
        genome, schema, args.schema, num_outputs, input_labels)

    model_rel = os.path.relpath(args.model, ROOT).replace("\\", "/")
    title = args.title or f"Genome network — {os.path.basename(args.model)}"
    fit = meta["fitness"]
    fit_str = f"fitness {fit:,.2f} · " if isinstance(fit, (int, float)) else ""
    subtitle = (f"{model_rel} · schema {meta['schema']} · {fit_str}"
                "enabled connections only, unconnected nodes dropped · "
                "layer = shortest distance from the input layer")

    data = {"meta": meta, "columns": columns, "edges": edges, "formulas": formulas,
            "model": model_rel}
    page = render_page(title, subtitle, data)

    out = args.out or (os.path.splitext(args.model)[0] + ".html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)

    print(f"wrote {out}")
    print(f"  {meta['enabledCount']}/{meta['geneCount']} connections enabled, "
          f"{meta['liveEdges']} reach an output")
    print(f"  drawn: {meta['inputsConnected']} inputs, {meta['hiddenConnected']} hidden, "
          f"{meta['outputsConnected']} outputs")
    if args.open:
        webbrowser.open("file:///" + os.path.abspath(out).replace("\\", "/"))
    return 0


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
    --muted:#898781; --hairline:#e1e0d9; --border:rgba(11,11,11,.10);
    --pos:#2a78d6; --neg:#e34948; --node-fill:#fcfcfb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7;
      --muted:#898781; --hairline:#2c2c2a; --border:rgba(255,255,255,.10);
      --pos:#3987e5; --neg:#e66767; --node-fill:#1a1a19;
    }
  }
  :root[data-theme="dark"] {
    --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7;
    --muted:#898781; --hairline:#2c2c2a; --border:rgba(255,255,255,.10);
    --pos:#3987e5; --neg:#e66767; --node-fill:#1a1a19;
  }
  :root[data-theme="light"] {
    --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
    --muted:#898781; --hairline:#e1e0d9; --border:rgba(11,11,11,.10);
    --pos:#2a78d6; --neg:#e34948; --node-fill:#fcfcfb;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.5; margin: 0; padding: 32px 24px 48px;
  }
  .wrap { max-width: 1240px; margin: 0 auto; }
  .mono { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }
  header h1 { font-size: 1.35rem; font-weight: 650; letter-spacing: -.01em;
    text-wrap: balance; margin: 0; }
  header .sub { color: var(--ink-2); font-size: .9rem; margin-top: 4px; word-break: break-word; }
  .stats { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
  .chip { border: 1px solid var(--border); background: var(--surface);
    border-radius: 999px; padding: 3px 12px; font-size: .78rem; color: var(--ink-2);
    white-space: nowrap; }
  .chip b { color: var(--ink); font-weight: 600; }
  .card { background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; margin-top: 20px; overflow: hidden; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 20px; padding: 12px 18px;
    border-bottom: 1px solid var(--hairline); font-size: .78rem; color: var(--ink-2);
    align-items: center; }
  .legend .item { display: inline-flex; align-items: center; gap: 7px; }
  .legend svg { display: block; }
  .chart-scroll { overflow-x: auto; }
  #net { display: block; margin: 0 auto; }
  .colhead { font-size: 11px; fill: var(--muted); text-transform: uppercase;
    letter-spacing: .08em; }
  .iolabel { font-size: 12px; fill: var(--ink-2); }
  .iolabel.live { fill: var(--ink); font-weight: 600; }
  .nid { font-size: 9.5px; fill: var(--muted); }
  .iolabel, .nid { paint-order: stroke; stroke: var(--surface); stroke-width: 3px;
    stroke-linejoin: round; }
  #tip { position: fixed; pointer-events: none; background: var(--surface);
    color: var(--ink); border: 1px solid var(--border); border-radius: 6px;
    box-shadow: 0 2px 10px rgba(0,0,0,.18); padding: 6px 10px; font-size: .78rem;
    display: none; z-index: 10; max-width: 300px; }
  #tip .t2 { color: var(--ink-2); }
  .edge { transition: opacity 120ms ease; }
  .hit { stroke: transparent; fill: none; stroke-width: 12; cursor: pointer; }
  @media (prefers-reduced-motion: reduce) { .edge { transition: none; } }
  section.explain { margin-top: 26px; }
  section.explain h2 { font-size: .95rem; font-weight: 650; margin: 0 0 4px; }
  section.explain > p { color: var(--ink-2); font-size: .88rem; max-width: 70ch; margin: 0 0 14px; }
  .outs { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px; align-items: start; }
  .out { background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; }
  .out h3 { margin: 0; font-size: .85rem; font-weight: 650; }
  .out .formula { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
    font-size: .78rem; line-height: 1.5; color: var(--ink); margin: 8px 0 0;
    overflow-x: auto; white-space: pre; padding-bottom: 4px; }
  .out .formula .hid { color: var(--muted); }
  .out .note { font-size: .78rem; color: var(--muted); margin-top: 6px; }
  footer { margin-top: 26px; padding-top: 14px; border-top: 1px solid var(--hairline);
    color: var(--muted); font-size: .78rem; max-width: 80ch; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>__TITLE__</h1>
    <div class="sub">__SUBTITLE__</div>
    <div class="stats" id="stats"></div>
  </header>
  <div class="card">
    <div class="legend">
      <span class="item"><svg width="26" height="8"><line x1="1" y1="4" x2="25" y2="4" stroke="var(--pos)" stroke-width="3" stroke-linecap="round"/></svg>positive weight</span>
      <span class="item"><svg width="26" height="8"><line x1="1" y1="4" x2="25" y2="4" stroke="var(--neg)" stroke-width="3" stroke-linecap="round"/></svg>negative weight</span>
      <span class="item"><svg width="30" height="10"><line x1="1" y1="5" x2="13" y2="5" stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round"/><line x1="17" y1="5" x2="29" y2="5" stroke="var(--muted)" stroke-width="5" stroke-linecap="round"/></svg>thickness = |weight|</span>
      <span class="item"><svg width="26" height="8"><line x1="1" y1="4" x2="25" y2="4" stroke="var(--muted)" stroke-width="3" stroke-linecap="round" opacity="0.28"/></svg>faded = dead end (never reaches an output)</span>
      <span class="item"><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="none" stroke="var(--ink-2)" stroke-width="1.6" stroke-dasharray="2.5 2.5"/></svg>bias-only (no path from inputs)</span>
      <span class="item" style="color:var(--muted)">hover a node or edge for details</span>
    </div>
    <div class="chart-scroll"><svg id="net" role="img"></svg></div>
  </div>
  <section class="explain">
    <h2 id="exhead"></h2>
    <p id="exsub"></p>
    <div class="outs" id="outs"></div>
  </section>
  <footer id="foot"></footer>
</div>
<div id="tip"></div>
<script>
(function () {
  if (location.hash === "#light" || location.hash === "#dark")
    document.documentElement.dataset.theme = location.hash.slice(1);

  var DATA = __DATA__;
  var meta = DATA.meta, columns = DATA.columns, edges = DATA.edges;

  // ---- header chips ----
  function chip(html) { return '<span class="chip">' + html + '</span>'; }
  var chips = [];
  if (typeof meta.fitness === "number")
    chips.push(chip('fitness <b class="mono">' + meta.fitness.toLocaleString(undefined,{maximumFractionDigits:2}) + '</b>'));
  chips.push(chip('schema <b class="mono">' + meta.schema + '</b>'));
  chips.push(chip('connections <b class="mono">' + meta.enabledCount + '</b> enabled of <b class="mono">' + meta.geneCount + '</b>'));
  chips.push(chip('reach an output <b class="mono">' + meta.liveEdges + '</b>'));
  chips.push(chip('inputs <b class="mono">' + meta.inputsConnected + '</b> of <b class="mono">' + meta.inputsTotal + '</b>'));
  chips.push(chip('hidden <b class="mono">' + meta.hiddenConnected + '</b> of <b class="mono">' + meta.hiddenTotal + '</b>'));
  chips.push(chip('outputs <b class="mono">' + meta.outputsConnected + '</b>'));
  document.getElementById("stats").innerHTML = chips.join("");

  // ---- layout ----
  var ncols = columns.length;
  var colGap = 232, xInput = 200, rightPad = 210, topPad = 56, botPad = 26;
  var maxLen = columns.reduce(function (m, c) { return Math.max(m, c.nodes.length); }, 1);
  var W = xInput + (ncols - 1) * colGap + rightPad;
  var H = Math.max(520, topPad + botPad + maxLen * 46);
  var svg = document.getElementById("net");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("aria-label", "Layered NEAT genome graph: " + meta.inputsConnected +
    " inputs, " + meta.hiddenConnected + " hidden nodes, " + meta.outputsConnected +
    " outputs, " + meta.enabledCount + " weighted connections.");

  var pos = {};   // node key -> {x,y,col,kind,live}
  columns.forEach(function (col, ci) {
    var x = xInput + ci * colGap;
    var step = (H - topPad - botPad) / (col.nodes.length + 1);
    col.nodes.forEach(function (n, i) {
      pos[n.key] = { x: x, y: topPad + step * (i + 1), col: ci, kind: n.kind, live: n.live, n: n };
    });
  });

  var NS = "http://www.w3.org/2000/svg";
  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e); return e;
  }
  var tip = document.getElementById("tip");

  // column headers
  columns.forEach(function (col, ci) {
    var t = el("text", { x: xInput + ci * colGap, y: 26, "text-anchor": "middle", "class": "colhead" });
    t.textContent = col.title;
  });

  var edgeLayer = el("g", {}), nodeLayer = el("g", {}), hitLayer = el("g", {});

  function edgePath(a, b) {
    var r = 11;
    if (b.x > a.x) {
      var x1 = a.x + r, x2 = b.x - r, mx = (x1 + x2) / 2;
      return "M " + x1 + " " + a.y + " C " + mx + " " + a.y + ", " + mx + " " + b.y + ", " + x2 + " " + b.y;
    }
    // same column or backward: bulge out to the right so it stays visible
    var bend = 46 + Math.abs(a.x - b.x) * 0.5;
    return "M " + (a.x + r) + " " + a.y + " C " + (a.x + bend) + " " + a.y +
           ", " + (b.x + bend) + " " + b.y + ", " + (b.x + r) + " " + b.y;
  }
  var maxW = meta.maxWeight || 1;
  function wScale(w) { return 1.2 + 4.3 * Math.sqrt(Math.abs(w) / maxW); }
  function fmt(v) { var a = Math.abs(v); return (v < 0 ? "−" : "+") + (a < 0.1 ? a.toFixed(4) : a.toFixed(2)); }

  var edgeEls = [];
  edges.forEach(function (e) {
    var a = pos[e.from], b = pos[e.to];
    if (!a || !b) return;
    var d = edgePath(a, b);
    var path = el("path", { d: d, fill: "none",
      stroke: e.w >= 0 ? "var(--pos)" : "var(--neg)",
      "stroke-width": wScale(e.w), "stroke-linecap": "round",
      opacity: e.live ? 0.85 : 0.22, "class": "edge" }, edgeLayer);
    var hit = el("path", { d: d, "class": "hit" }, hitLayer);
    var rec = { path: path, base: e.live ? 0.85 : 0.22 };
    edgeEls.push(rec);
    bindTip(hit,
      function () { return '<b class="mono">' + e.from + " → " + e.to + '</b><br>' +
        '<span class="t2">weight <span class="mono">' + fmt(e.w) + '</span> · ' +
        (e.live ? "reaches an output" : "dead end") + '</span>'; },
      function () { focusEdge(rec); }, unfocus);
  });

  columns.forEach(function (col) {
    col.nodes.forEach(function (n) {
      var p = pos[n.key], g = el("g", { "class": "node" }, nodeLayer);
      var isOut = n.kind === "output", isIn = n.kind === "input";
      var c = el("circle", { cx: p.x, cy: p.y, r: isOut ? 11 : 9,
        fill: isIn ? "var(--ink-2)" : isOut ? "var(--ink)" : "var(--node-fill)",
        stroke: n.kind === "hidden" ? "var(--ink-2)" : "none", "stroke-width": 1.8,
        opacity: n.live ? 1 : 0.42 }, g);
      if (n.biasOnly) { c.setAttribute("stroke-dasharray", "3 3"); c.setAttribute("fill", "var(--node-fill)"); }
      if (isIn) {
        var t = el("text", { x: p.x - 20, y: p.y + 4, "text-anchor": "end",
          "class": "iolabel" + (n.live ? " live" : "") }, g); t.textContent = n.label;
        var id = el("text", { x: p.x, y: p.y - 14, "text-anchor": "middle", "class": "nid mono" }, g); id.textContent = n.key;
      } else if (isOut) {
        var to = el("text", { x: p.x + 20, y: p.y + 4, "class": "iolabel live" }, g); to.textContent = n.label;
      } else {
        var hid = el("text", { x: p.x, y: p.y - 14, "text-anchor": "middle", "class": "nid mono" }, g); hid.textContent = n.key;
      }
      var hit = el("circle", { cx: p.x, cy: p.y, r: 16, fill: "transparent", style: "cursor:pointer" }, hitLayer);
      bindTip(hit, function () {
        if (isIn) return '<b>' + n.label + '</b> <span class="t2 mono">(' + n.key + ')</span><br><span class="t2">' + n.sub + '</span>';
        var head = n.sub;
        var fn = (n.act === "sigmoid" || !n.act) ? "sigmoid" : n.act;
        return '<b class="mono">' + head + '</b><br><span class="t2">bias <span class="mono">' +
          fmt(n.bias) + '</span> · ' + fn + (n.live ? "" : " · dead end") + '</span>';
      }, function () { focusNode(n.key); }, unfocus);
    });
  });

  // ---- interaction ----
  function bindTip(target, htmlFn, onEnter, onLeave) {
    target.addEventListener("pointerenter", function (e) {
      tip.innerHTML = htmlFn(); tip.style.display = "block"; moveTip(e); onEnter();
    });
    target.addEventListener("pointermove", moveTip);
    target.addEventListener("pointerleave", function () { tip.style.display = "none"; onLeave(); });
  }
  function moveTip(e) {
    var pad = 14, x = e.clientX + pad, y = e.clientY + pad, r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  var incidentOf = {};
  edges.forEach(function (e, i) { (incidentOf[e.from] = incidentOf[e.from] || []).push(i);
    (incidentOf[e.to] = incidentOf[e.to] || []).push(i); });
  function focusNode(key) {
    var on = {}; (incidentOf[key] || []).forEach(function (i) { on[i] = 1; });
    edgeEls.forEach(function (r, i) { r.path.setAttribute("opacity", on[i] ? 0.95 : 0.08); });
  }
  function focusEdge(rec) { edgeEls.forEach(function (r) { r.path.setAttribute("opacity", r === rec ? 0.95 : 0.08); }); }
  function unfocus() { edgeEls.forEach(function (r) { r.path.setAttribute("opacity", r.base); }); }

  // ---- derived-formula section ----
  document.getElementById("exhead").textContent =
    meta.liveEdges + " of " + meta.enabledCount + " enabled connections actually reach an output";
  document.getElementById("exsub").textContent =
    "Everything drawn faded above dead-ends before any output. Expanding only the live edges, each output computes exactly this (σ = sigmoid):";
  document.getElementById("outs").innerHTML = DATA.formulas.map(function (f) {
    return '<div class="out"><h3>' + f.label + '</h3><div class="formula">' + f.label +
      ' = ' + f.html + '</div>' + (f.note ? '<div class="note">' + f.note + '</div>' : '') + '</div>';
  }).join("");

  function plural(n, one, many) { return n + " " + (n === 1 ? one : (many || one + "s")); }
  var bits = [plural(meta.omittedGenes, "disabled connection gene")];
  if (meta.omittedHidden + meta.omittedInputs)
    bits.push(plural(meta.omittedHidden, "hidden node") + " and " +
      plural(meta.omittedInputs, "input") + " with no enabled connection");
  var foot = "Omitted: " + bits.join(", ") + ". ";
  if (meta.biasOnly.length) {
    var many = meta.biasOnly.length > 1;
    foot += "Bias-only node" + (many ? "s " : " ") + meta.biasOnly.join(", ") +
      " (dashed ring) " + (many ? "have" : "has") +
      " enabled outgoing links but no incoming ones, so " + (many ? "they" : "it") +
      " emit" + (many ? "" : "s") + " a constant σ(bias) and seed" + (many ? "" : "s") + " layer 1. ";
  }
  foot += meta.activationNote;
  document.getElementById("foot").textContent = foot;
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
