/**
 * dashboard/js/components/graph.js
 *
 * The fixed physiological graph: 8 nodes, 15 directed edges, defined before
 * training. Lenses:
 *   structure   — the prior graph only (edge type, organ system)
 *   attribution — node size encodes cohort plain-gradient sensitivity share
 * Attention is deliberately not drawn as node importance: briefing §6.3–6.4
 * shows attention ranks nodes no better than random.
 */
(function () {
  "use strict";
  const PG = (window.PG = window.PG || {});
  const esc = (s) => PG.esc(s);

  const W = 760, H = 500;
  const POS = {
    resp_rate:   [150, 112],
    spo2:        [380, 72],
    heart_rate:  [610, 112],
    sbp:         [650, 312],
    creatinine:  [480, 430],
    lactate:     [380, 260],
    wbc:         [200, 410],
    temperature: [82, 300],
  };
  PG.GRAPH_POS = POS;

  const isPair = (e) => PG.EDGES.some((x) => x.from === e.to && x.to === e.from);

  function edgePath(e, r1, r2) {
    const [x1, y1] = POS[e.from], [x2, y2] = POS[e.to];
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
    const ux = dx / len, uy = dy / len;
    // Reciprocal pairs bow apart; single edges are straight.
    const bend = isPair(e) ? 26 : 0;
    const nx = -uy, ny = ux;
    const mx = (x1 + x2) / 2 + nx * bend, my = (y1 + y2) / 2 + ny * bend;
    // trim to node radii along the curve's end tangents
    const t1x = mx - x1, t1y = my - y1, l1 = Math.hypot(t1x, t1y);
    const t2x = x2 - mx, t2y = y2 - my, l2 = Math.hypot(t2x, t2y);
    const sx = x1 + (t1x / l1) * r1, sy = y1 + (t1y / l1) * r1;
    const ex = x2 - (t2x / l2) * (r2 + 4), ey = y2 - (t2y / l2) * (r2 + 4);
    return { d: `M${sx.toFixed(1)},${sy.toFixed(1)}Q${mx.toFixed(1)},${my.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`, mx, my };
  }

  // shares: {key: share} to size nodes by; null draws the plain structure.
  function radius(key, shares) {
    if (!shares) return 30;
    return 18 + Math.sqrt(shares[key] / 0.16) * 22; // area ∝ share
  }

  /** PG.renderGraph(el, { shares, node, edge, onNode, onEdge }) */
  PG.renderGraph = function (el, o) {
    const shares = o.shares || null;
    const selNode = o.node, selEdge = o.edge;
    const connected = new Set();
    if (selNode) PG.EDGES.forEach((e, i) => { if (e.from === selNode || e.to === selNode) connected.add(i); });

    let s = `<svg class="graph-svg" viewBox="0 0 ${W} ${H}" role="group" aria-label="Physiological graph, 8 nodes and 15 directed edges">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1L9,5L0,9Z" class="arrow"/></marker>
        <marker id="arr-hl" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1L9,5L0,9Z" class="arrow hl"/></marker>
        <marker id="arr-dim" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1L9,5L0,9Z" class="arrow dim"/></marker>
      </defs>`;

    PG.EDGES.forEach((e, i) => {
      const { d } = edgePath(e, radius(e.from, shares), radius(e.to, shares));
      const hl = selEdge === i || connected.has(i);
      const dim = (selNode || selEdge !== null && selEdge !== undefined) && !hl;
      const cls = `edge ${e.type}${hl ? " hl" : ""}${dim ? " dim" : ""}`;
      s += `<g class="edge-g" data-edge="${i}" tabindex="0" role="button" aria-label="${esc(PG.NODE[e.from].label)} to ${esc(PG.NODE[e.to].label)}: ${esc(e.rationale)}">
        <path class="edge-hit" d="${d}"/>
        <path class="${cls}" d="${d}" marker-end="url(#${hl ? "arr-hl" : dim ? "arr-dim" : "arr"})"/></g>`;
    });

    PG.NODES.forEach((n) => {
      const [x, y] = POS[n.key], r = radius(n.key, shares);
      const isSel = selNode === n.key;
      const dim = selNode && !isSel && !PG.EDGES.some((e) => (e.from === selNode && e.to === n.key) || (e.to === selNode && e.from === n.key));
      const color = PG.SYSTEMS[n.system].color;
      const labelBelow = y < H / 2 ? -(r + 12) : r + 22;
      s += `<g class="gnode${isSel ? " is-selected" : ""}${dim ? " dim" : ""}" data-node="${n.key}" transform="translate(${x},${y})" tabindex="0" role="button" aria-pressed="${isSel}" aria-label="${esc(n.label)}" style="--mc:${color}">
        <circle class="gn-halo" r="${r + 7}"/>
        <circle class="gn-body" r="${r}"/>
        <text class="gn-short" y="${shares ? -2 : 5}" text-anchor="middle">${esc(n.short)}</text>
        ${shares ? `<text class="gn-share" y="13" text-anchor="middle">${(shares[n.key] * 100).toFixed(1)}%</text>` : ""}
        <text class="gn-label" y="${labelBelow}" text-anchor="middle">${n.index} · ${esc(n.label)}</text>
      </g>`;
    });
    s += `</svg><div class="graph-tip" role="tooltip" hidden></div>`;
    el.innerHTML = s;

    const tip = el.querySelector(".graph-tip");
    const place = (target, html) => {
      tip.innerHTML = html; tip.hidden = false;
      const er = el.getBoundingClientRect(), r = target.getBoundingClientRect();
      tip.style.left = Math.min(Math.max(6, r.left - er.left + r.width / 2 - tip.offsetWidth / 2), er.width - tip.offsetWidth - 6) + "px";
      tip.style.top = Math.max(0, r.top - er.top - tip.offsetHeight - 6) + "px";
    };
    const hide = () => (tip.hidden = true);
    const act = (node, fn) => {
      node.addEventListener("click", fn);
      node.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); fn(); } });
    };

    el.querySelectorAll(".gnode").forEach((g) => {
      const n = PG.NODE[g.dataset.node];
      act(g, () => o.onNode && o.onNode(n.key));
      const out = PG.EDGES.filter((e) => e.from === n.key).length, inn = PG.EDGES.filter((e) => e.to === n.key).length;
      const html = `<b>${esc(n.label)}</b><span>Node ${n.index} · ${esc(PG.SYSTEMS[n.system].label)}</span><span>${inn} incoming · ${out} outgoing</span>` +
        (shares ? `<span>Gradient share ${(shares[n.key] * 100).toFixed(1)}%</span>` : "");
      g.addEventListener("mouseenter", () => place(g, html));
      g.addEventListener("mouseleave", hide);
    });
    el.querySelectorAll(".edge-g").forEach((g) => {
      const i = +g.dataset.edge, e = PG.EDGES[i];
      act(g, () => o.onEdge && o.onEdge(i));
      const html = `<b>${esc(PG.NODE[e.from].short)} → ${esc(PG.NODE[e.to].short)}</b><span>${esc(e.rationale)}</span><span>${e.type === "reflex" ? "Reciprocal reflex arc" : "Forward cascade"}</span>`;
      g.addEventListener("mouseenter", () => place(g.querySelector(".edge"), html));
      g.addEventListener("mouseleave", hide);
    });
  };
})();
