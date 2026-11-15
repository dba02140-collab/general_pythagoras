"""
Pythagoras for ANY shape
------------------------
Draw a polygon on leg a of a right triangle. The app erects a similar copy of it
on leg b and on the hypotenuse c (each scaled by the side length), then compares

        area on a  +  area on b    vs.    area on c

Why it works: scaling a figure by a factor k multiplies its area by k^2.  If the
shape built on a segment of length 1 has area A0, then the copy on a side of
length s has area A0*s^2, so  A0*a^2 + A0*b^2 = A0*(a^2 + b^2) = A0*c^2.

Run locally:   streamlit run app.py
Requires:      streamlit >= 1.51 (for st.components.v2), plotly, pandas, numpy
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pythagoras for any shape", layout="wide")

# ---------------------------------------------------------------------------
# Geometry (all in Python)
# ---------------------------------------------------------------------------
# A shape is stored in a normalised frame attached to a unit segment:
#   the side runs from (0, 0) to (1, 0); +v points AWAY from the triangle.
# The same (u, v) coordinates are then mapped onto every side of the triangle.


def signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(pts: np.ndarray) -> float:
    return abs(signed_area(pts))


def centroid(pts: np.ndarray) -> np.ndarray:
    a = signed_area(pts)
    if abs(a) < 1e-12:
        return pts.mean(axis=0)
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    return np.array([np.sum((x + x1) * cross), np.sum((y + y1) * cross)]) / (6 * a)


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def is_self_intersecting(pts: np.ndarray) -> bool:
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges share a vertex
            if _segments_cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                return True
    return False


def triangle_vertices(a: float, b: float, gamma_deg: float):
    """C at the origin, leg a along the x-axis, angle gamma at C."""
    g = math.radians(gamma_deg)
    C = np.array([0.0, 0.0])
    B = np.array([a, 0.0])
    A = np.array([b * math.cos(g), b * math.sin(g)])
    return C, B, A


def place_on_side(shape_uv: np.ndarray, P: np.ndarray, Q: np.ndarray, inward: bool,
                  mirror: bool = False) -> np.ndarray:
    """Similarity map: (0,0)->P, (1,0)->Q, +v -> outward normal (or inward).

    The triangle is traversed counter-clockwise (C->B->A), so the outward normal
    of each directed edge is the direction rotated by -90 degrees.  Using a pure
    rotation+scale (no reflection) keeps all three copies directly similar.
    """
    d = Q - P
    n = np.array([d[1], -d[0]])  # outward
    if inward:
        n = -n
    u, v = shape_uv[:, 0:1], shape_uv[:, 1:2]
    if mirror:  # reflect the copy across the side's perpendicular bisector
        u = 1.0 - u
    return P + u * d + v * n


# ---------------------------------------------------------------------------
# Preset shapes (normalised frame)
# ---------------------------------------------------------------------------


def regular_polygon_on_base(n: int) -> list[list[float]]:
    pts, p, ang = [[0.0, 0.0]], np.array([0.0, 0.0]), 0.0
    for _ in range(n - 1):
        p = p + np.array([math.cos(ang), math.sin(ang)])
        pts.append(p.tolist())
        ang += 2 * math.pi / n
    return pts


def semicircle(k: int = 40) -> list[list[float]]:
    return [[0.5 + 0.5 * math.cos(t), 0.5 * math.sin(t)] for t in np.linspace(0, math.pi, k + 1)]


def star(points: int = 5) -> list[list[float]]:
    cx, cy, R, r = 0.5, 0.55, 0.5, 0.2
    out = []
    for i in range(2 * points):
        rad = R if i % 2 == 0 else r
        t = math.pi / 2 + i * math.pi / points
        out.append([cx + rad * math.cos(t), cy + rad * math.sin(t)])
    return out


def similar_triangle(a: float, b: float) -> list[list[float]]:
    """A copy of the right triangle with its hypotenuse on the unit side."""
    c2 = a * a + b * b
    return [[0.0, 0.0], [1.0, 0.0], [b * b / c2, a * b / c2]]


PRESETS = {
    "Square (the classic)": lambda a, b: [[0, 0], [1, 0], [1, 1], [0, 1]],
    "Equilateral triangle": lambda a, b: regular_polygon_on_base(3),
    "Semicircle (40-gon)": lambda a, b: semicircle(),
    "Regular hexagon": lambda a, b: regular_polygon_on_base(6),
    "Isosceles right triangle": lambda a, b: [[0, 0], [1, 0], [0.5, 0.5]],
    "House": lambda a, b: [[0, 0], [1, 0], [1, 0.7], [0.5, 1.2], [0, 0.7]],
    "Lopsided arrow (asymmetric)": lambda a, b: [
        [0, 0], [1, 0], [0.9, 0.35], [1.3, 0.35], [0.6, 1.1], [0.1, 0.35], [0.25, 0.35]],
    "Five-pointed star (floating)": lambda a, b: star(),
    "Copy of the triangle itself": similar_triangle,
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("verts", PRESETS["Square (the classic)"](3, 4))
ss.setdefault("mirror_c", False)
ss.setdefault("ver", 0)  # bump -> drawing pad reloads from Python
ss.setdefault("tbl_ver", 0)  # bump -> vertex table reloads from Python
ss.setdefault("tbl_df", pd.DataFrame(ss.verts, columns=["u", "v"]))


def set_verts(verts, reset_pad=True, reset_table=True):
    ss.verts = [[round(float(u), 4), round(float(v), 4)] for u, v in verts]
    if reset_pad:
        ss.ver += 1
    if reset_table:
        ss.tbl_ver += 1
        ss.tbl_df = pd.DataFrame(ss.verts, columns=["u", "v"])


def apply_preset():
    set_verts(PRESETS[ss.preset](ss.a, ss.b))
    if ss.preset == "Copy of the triangle itself":
        ss.mirror_c = True  # so the copy on c lands exactly on the triangle


def on_pad_edit():
    val = ss.get("shape_pad")
    edit = None
    if val is not None:
        edit = val.get("edit") if hasattr(val, "get") else getattr(val, "edit", None)
    if edit and len(edit.get("verts", [])) >= 3:
        set_verts(edit["verts"], reset_pad=False, reset_table=True)


# ---------------------------------------------------------------------------
# Drawing pad: an inline bidirectional component (HTML/SVG + JS)
# ---------------------------------------------------------------------------
PAD_CSS = """
.pt-root { font-family: sans-serif; user-select: none; }
.pt-root svg { width: 100%; max-width: 560px; display: block; touch-action: none;
               border: 1px solid rgba(128,128,128,.35); border-radius: 8px; cursor: crosshair; }
.pt-grid { stroke: rgba(128,128,128,.18); stroke-width: .004; }
.pt-axis { stroke: rgba(128,128,128,.45); stroke-width: .006; }
.pt-base { stroke: #1f77b4; stroke-width: .03; stroke-linecap: round; }
.pt-poly { fill: rgba(31,119,180,.28); stroke: #1f77b4; stroke-width: .012; stroke-linejoin: round; }
.pt-vert { fill: white; stroke: #1f77b4; stroke-width: .012; cursor: grab; }
.pt-vert:hover, .pt-vert.drag { fill: #ff7f0e; }
.pt-lbl { font-size: .07px; fill: rgba(128,128,128,.9); }
.pt-help { font-size: 13px; opacity: .75; margin: 6px 0 0 2px; line-height: 1.4; }
"""

PAD_JS = r"""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const NS = "http://www.w3.org/2000/svg";
  const X0 = -0.6, X1 = 1.6, Y0 = -0.6, Y1 = 1.6;   // world window (u, v)

  let root = parentElement.querySelector(".pt-root");
  if (!root) {
    root = document.createElement("div");
    root.className = "pt-root";
    parentElement.appendChild(root);
  }
  // Keep local edits unless Python sends a new version (preset / table edit)
  if (root._ver !== data.version) {
    root._ver = data.version;
    root._verts = data.verts.map(p => [p[0], p[1]]);
  }
  const snap = data.snap || 0;
  const S = v => Math.round(v * 1e4) / 1e4;

  root.innerHTML = "";
  const svg = document.createElementNS(NS, "svg");
  // y is flipped so +v (outward) points up on screen
  svg.setAttribute("viewBox", `${X0} ${-Y1} ${X1 - X0} ${Y1 - Y0}`);
  root.appendChild(svg);
  const help = document.createElement("div");
  help.className = "pt-help";
  help.innerHTML = "<b>Drag</b> a vertex to move it &middot; <b>click</b> empty space to add a vertex " +
                   "(inserted into the nearest edge) &middot; <b>double-click</b> or <b>right-click</b> a vertex to delete it.<br>" +
                   "The thick blue segment is leg <i>a</i>; up is <i>away</i> from the triangle.";
  root.appendChild(help);

  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e);
    return e;
  };

  // grid
  for (let t = -0.5; t <= 1.5001; t += 0.25) {
    el("line", { x1: t, y1: -Y1, x2: t, y2: -Y0, class: Math.abs(t) < 1e-9 ? "pt-axis" : "pt-grid" });
    el("line", { x1: X0, y1: -t, x2: X1, y2: -t, class: Math.abs(t) < 1e-9 ? "pt-axis" : "pt-grid" });
  }
  el("line", { x1: 0, y1: 0, x2: 1, y2: 0, class: "pt-base" });
  const lab = el("text", { x: 0.5, y: 0.1, class: "pt-lbl", "text-anchor": "middle" });
  lab.textContent = "leg a  (length 1 in this frame)";
  const tri = el("text", { x: 0.5, y: 0.45, class: "pt-lbl", "text-anchor": "middle" });
  tri.textContent = "▼ triangle is on this side ▼";

  const poly = el("polygon", { class: "pt-poly" });
  const g = el("g", {});
  let verts = root._verts;

  function toWorld(evt) {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    let u = p.x, v = -p.y;
    if (snap > 0) { u = Math.round(u / snap) * snap; v = Math.round(v / snap) * snap; }
    return [S(Math.min(X1, Math.max(X0, u))), S(Math.min(Y1, Math.max(Y0, v)))];
  }

  function send() { setTriggerValue("edit", { verts: verts.map(p => [p[0], p[1]]) }); }

  function draw() {
    poly.setAttribute("points", verts.map(p => `${p[0]},${-p[1]}`).join(" "));
    g.innerHTML = "";
    verts.forEach((p, i) => {
      const c = el("circle", { cx: p[0], cy: -p[1], r: 0.035, class: "pt-vert" }, g);
      c.dataset.i = i;
    });
  }
  draw();

  let dragIdx = -1;
  svg.addEventListener("pointerdown", evt => {
    if (evt.button !== 0) return;
    const t = evt.target;
    if (t.classList && t.classList.contains("pt-vert")) {
      dragIdx = +t.dataset.i;
      t.classList.add("drag");
      svg.setPointerCapture(evt.pointerId);
      evt.preventDefault();
      return;
    }
    // add a vertex: insert it into the edge whose replacement adds the least length
    const q = toWorld(evt);
    const d = (p, r) => Math.hypot(p[0] - r[0], p[1] - r[1]);
    let best = 0, bestCost = Infinity;
    for (let i = 0; i < verts.length; i++) {
      const p = verts[i], r = verts[(i + 1) % verts.length];
      const cost = d(p, q) + d(q, r) - d(p, r);
      if (cost < bestCost) { bestCost = cost; best = i; }
    }
    verts.splice(best + 1, 0, q);
    draw(); send();
  });
  svg.addEventListener("pointermove", evt => {
    if (dragIdx < 0) return;
    verts[dragIdx] = toWorld(evt);
    poly.setAttribute("points", verts.map(p => `${p[0]},${-p[1]}`).join(" "));
    const c = g.children[dragIdx];
    c.setAttribute("cx", verts[dragIdx][0]); c.setAttribute("cy", -verts[dragIdx][1]);
  });
  const endDrag = () => {
    if (dragIdx < 0) return;
    dragIdx = -1; draw(); send();
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  const del = evt => {
    const t = evt.target;
    if (t.classList && t.classList.contains("pt-vert")) {
      evt.preventDefault();
      if (verts.length > 3) { verts.splice(+t.dataset.i, 1); draw(); send(); }
    }
  };
  svg.addEventListener("dblclick", del);
  svg.addEventListener("contextmenu", del);
}
"""

shape_pad = st.components.v2.component("pythagoras_shape_pad", css=PAD_CSS, js=PAD_JS)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Triangle")
    st.slider("Leg a (the one you draw on)", 1.0, 10.0, 3.0, 0.5, key="a")
    st.slider("Leg b", 1.0, 10.0, 4.0, 0.5, key="b")
    st.slider(
        "Angle between a and b (°)", 30, 150, 90, 1, key="gamma",
        help="Leave at 90° for a right triangle. Change it to see the relationship break.",
    )
    st.header("Shape")
    st.selectbox("Load a preset", list(PRESETS), key="preset", on_change=apply_preset,
                 help="Choosing a preset replaces the current shape.")
    if st.button("Reload preset", width="stretch"):
        apply_preset()
    snap_label = st.radio("Snap to grid", ["Off", "0.05", "0.125", "0.25"], index=0, horizontal=True)
    st.toggle("Build shapes inward (over the triangle)", key="inward",
              help="Try it with 'Copy of the triangle itself' for Einstein's proof.")
    st.toggle("Mirror the copy on c", key="mirror_c",
              help="Reflect the shape on side c. Mirror images have the same area, so the balance is unaffected.")
    st.toggle("Show area labels on the figure", value=True, key="labels")

a, b, gamma = ss.a, ss.b, ss.gamma
right = gamma == 90
snap = 0.0 if snap_label == "Off" else float(snap_label)

st.title("Pythagoras works for *any* shape")
st.caption(
    "Draw a shape on leg *a*. The app builds a similar copy on every side of the triangle, "
    "scaled to that side, and adds up the areas."
)

left, right_col = st.columns([5, 6], gap="large")

# ---------------------------------------------------------------------------
# Left: shape editor
# ---------------------------------------------------------------------------
with left:
    st.subheader("1 · Design the shape on leg a")
    shape_pad(
        key="shape_pad",
        data={"verts": ss.verts, "version": ss.ver, "snap": snap},
        on_edit_change=on_pad_edit,
    )

    with st.expander("Edit vertex coordinates exactly"):
        st.caption("In this frame leg a runs from (0, 0) to (1, 0); positive v points away from the triangle. "
                   "Rows are vertices in order; add or delete rows freely.")
        edited = st.data_editor(
            ss.tbl_df, key=f"tbl_{ss.tbl_ver}", num_rows="dynamic", width="stretch",
            column_config={
                "u": st.column_config.NumberColumn("u (along leg)", step=0.01, format="%.3f"),
                "v": st.column_config.NumberColumn("v (outward)", step=0.01, format="%.3f"),
            },
        )
        clean = edited.dropna()
        new = clean[["u", "v"]].to_numpy(dtype=float).round(4).tolist()
        if len(new) >= 3 and new != ss.verts:
            set_verts(new, reset_pad=True, reset_table=False)
            st.rerun()  # redraw the pad (already rendered above) with the new shape
        elif len(new) < 3:
            st.warning("A polygon needs at least 3 vertices; the table edit is ignored until then.")

# ---------------------------------------------------------------------------
# Compute everything
# ---------------------------------------------------------------------------
shape = np.array(ss.verts, dtype=float)
A0 = polygon_area(shape)  # area of the shape built on a unit segment
C, B, A = triangle_vertices(a, b, gamma)
c = float(np.linalg.norm(A - B))

sides = [  # name, P, Q, length, colour
    ("a", C, B, a, "#1f77b4"),
    ("b", A, C, b, "#ff7f0e"),
    ("c", B, A, c, "#2ca02c"),
]
placed = {name: place_on_side(shape, P, Q, ss.inward, mirror=(name == "c" and ss.mirror_c))
          for name, P, Q, _, _ in sides}
areas = {name: polygon_area(pts) for name, pts in placed.items()}
lhs, rhs = areas["a"] + areas["b"], areas["c"]

# ---------------------------------------------------------------------------
# Right: the full picture and the comparison
# ---------------------------------------------------------------------------
with right_col:
    st.subheader("2 · Copies on all three sides")
    fig = go.Figure()
    for name, P, Q, length, col in [sides[2], sides[0], sides[1]]:  # c underneath a and b
        pts = placed[name]
        closed = np.vstack([pts, pts[:1]])
        fig.add_trace(go.Scatter(
            x=closed[:, 0], y=closed[:, 1], fill="toself", mode="lines",
            line=dict(color=col, width=2), fillcolor=col, opacity=0.45,
            name=f"on {'leg' if name != 'c' or not right else 'hypotenuse'} {name}: {areas[name]:.3f}",
            hoverinfo="name",
        ))
        if ss.labels:
            cx, cy = centroid(pts)
            fig.add_annotation(x=cx, y=cy, text=f"<b>{areas[name]:.2f}</b>", showarrow=False,
                               font=dict(size=14, color=col), bgcolor="rgba(255,255,255,.75)")
    tri = np.vstack([C, B, A, C])
    fig.add_trace(go.Scatter(x=tri[:, 0], y=tri[:, 1], mode="lines", line=dict(color="#444", width=3),
                             fill="toself", fillcolor="rgba(120,120,120,.12)", name="triangle",
                             hoverinfo="skip"))
    for name, P, Q, length, col in sides:
        mid = (P + Q) / 2
        fig.add_annotation(x=mid[0], y=mid[1], text=f"{name} = {length:.3g}", showarrow=False,
                           font=dict(size=12, color="#333"), bgcolor="rgba(255,255,255,.85)")
    if right:
        s = 0.06 * min(a, b)
        fig.add_trace(go.Scatter(x=[s, s, 0], y=[0, s, s], mode="lines",
                                 line=dict(color="#444", width=1.5), hoverinfo="skip", showlegend=False))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.02), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.subheader("3 · Compare the areas")
m1, m2, m3 = st.columns(3)
m1.metric("Area on a + area on b", f"{lhs:.4f}")
m2.metric(f"Area on {'hypotenuse ' if right else ''}c", f"{rhs:.4f}")
diff = lhs - rhs
balanced = abs(diff) < 1e-9 * max(1.0, rhs)
m3.metric("Difference", f"{0.0 if balanced else diff:+.4f}")
if balanced:
    st.success(f"Balanced: {areas['a']:.4f} + {areas['b']:.4f} = {areas['c']:.4f}")
else:
    st.error(f"Not balanced: {areas['a']:.4f} + {areas['b']:.4f} = {lhs:.4f}, but the shape on c has area {rhs:.4f}")

table = pd.DataFrame({
    "Side": ["a", "b", "c"],
    "Length": [a, b, c],
    "Length²": [a * a, b * b, c * c],
    "Area of shape on it": [areas["a"], areas["b"], areas["c"]],
    "Area ÷ Length²": [areas["a"] / (a * a), areas["b"] / (b * b), areas["c"] / (c * c)],
})
st.dataframe(table.style.format({"Length": "{:.4f}", "Length²": "{:.4f}",
                                 "Area of shape on it": "{:.4f}", "Area ÷ Length²": "{:.4f}"}),
             hide_index=True, width="stretch")

if is_self_intersecting(shape):
    st.warning("Your polygon crosses itself, so its 'area' is ambiguous (the shoelace formula lets loops "
               "cancel). The three copies are still treated identically, so the comparison holds, "
               "but try untangling it.")
if not right:
    cos_term = -2 * a * b * math.cos(math.radians(gamma))
    st.info(
        f"The angle is {gamma}°, not 90°, so c² = a² + b² − 2ab·cos γ = {a*a + b*b:.3f} "
        f"{'+' if cos_term >= 0 else '−'} {abs(cos_term):.3f}. The shapes still have Area ÷ Length² = "
        f"{A0:.4f} on every side, but now a² + b² ≠ c², so the areas don't balance. "
        f"(Area on a + b {'<' if diff < 0 else '>'} area on c exactly when the angle is "
        f"{'obtuse' if diff < 0 else 'acute'}.)"
    )

with st.expander("Why does this work?"):
    st.markdown(
        f"""
**Scaling squares the area.** Stretch a figure by a factor *k* in every direction and its area
is multiplied by *k*². (Each little grid square inside it becomes *k* by *k*.)

Your shape, built on a segment of length 1, has area **A₀ = {A0:.4f}**. Every copy on the triangle is
the same shape scaled to its side, so

* area on *a* = A₀·a² = {A0:.4f} × {a*a:.3g} = {areas['a']:.4f}
* area on *b* = A₀·b² = {A0:.4f} × {b*b:.3g} = {areas['b']:.4f}
* area on *c* = A₀·c² = {A0:.4f} × {c*c:.4g} = {areas['c']:.4f}

Adding the first two: A₀·a² + A₀·b² = A₀(a² + b²). That equals A₀·c² **exactly when a² + b² = c²**, i.e.
when the triangle is right-angled. The square is just the special case A₀ = 1. That's the last column
of the table: the ratio Area ÷ Length² is the same for all three sides, whatever shape you draw.

The shape doesn't have to touch the side, be convex, or even be "nice"; it only has to be
**similar** on all three sides, with matching size relative to the side.

**Einstein's proof.** Pick *Copy of the triangle itself* and turn on *Build shapes inward*. The copy on
the hypotenuse is the triangle itself, and the copies on the two legs are exactly the two pieces the
altitude from the right angle cuts it into. (Those pieces are mirror images of the big triangle, which is
why that preset also switches on *Mirror the copy on c*; reflection doesn't change area.) The two small areas visibly add up to the big one, and by the argument above, so do the
squares.

**The converse.** Move the angle slider off 90°. The ratio column stays constant, but the areas no
longer balance, because a² + b² ≠ c².
"""
    )
