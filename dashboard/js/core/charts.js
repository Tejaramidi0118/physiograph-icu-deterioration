/**
 * dashboard/js/core/charts.js
 *
 * Dependency-free SVG charts. Observation status is encoded consistently:
 *   observed  → solid point on a solid line
 *   carried   → dashed line, no point (the value is a repeat)
 *   median    → gap in the line, tick on the baseline (no patient value)
 */
(function () {
  "use strict";
  const PG = (window.PG = window.PG || {});
  const esc = (s) => PG.esc(s);

  function niceTicks(lo, hi, count) {
    const span = hi - lo || 1;
    const step0 = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const err = step0 / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function extent(values, pad = 0.12) {
    const vs = values.filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
    if (!vs.length) return null;
    let lo = Math.min(...vs), hi = Math.max(...vs);
    if (lo === hi) { const d = Math.abs(lo) * 0.05 || 1; lo -= d; hi += d; }
    const p = (hi - lo) * pad;
    return [lo - p, hi + p];
  }

  /** Status-aware path pieces for a record series [{h, v, s}]. */
  function statusPaths(records, X, Y) {
    let solid = "", dashed = "";
    for (let i = 1; i < records.length; i++) {
      const a = records[i - 1], b = records[i];
      if (a.v === null || b.v === null) continue;
      const seg = `M${X(a.h).toFixed(1)},${Y(a.v).toFixed(1)}L${X(b.h).toFixed(1)},${Y(b.v).toFixed(1)}`;
      if (b.s === "carried") dashed += seg; else solid += seg;
    }
    return { solid, dashed };
  }

  PG.tickFmt = (v) => (Number.isInteger(+(+v).toFixed(6)) ? String(Math.round(v)) : (+v).toFixed(Math.abs(v) < 1 ? 2 : 1));

  // -------------------------------------------------------------- sparkline
  PG.sparkline = function (records, opts = {}) {
    const w = opts.w || 132, h = opts.h || 34, padY = 4;
    const color = opts.color || "var(--accent)";
    const ext = extent(records.map((r) => r.v), 0.15);
    const h0 = records[0].h, h1 = records[records.length - 1].h;
    const X = (hh) => 2 + ((hh - h0) / Math.max(1, h1 - h0)) * (w - 6);
    let body = "";
    if (!ext) {
      body = `<line x1="2" x2="${w - 2}" y1="${h / 2}" y2="${h / 2}" class="spark-none"/>`;
    } else {
      const Y = (v) => padY + (1 - (v - ext[0]) / (ext[1] - ext[0])) * (h - padY * 2 - 3);
      const p = statusPaths(records, X, Y);
      // soft area under each run of consecutive values
      let run = [];
      const flush = () => {
        if (run.length > 1) body += `<path class="spark-area" style="fill:${color}" d="M${run.map((r) => `${X(r.h).toFixed(1)},${Y(r.v).toFixed(1)}`).join("L")}L${X(run[run.length - 1].h).toFixed(1)},${h}L${X(run[0].h).toFixed(1)},${h}Z"/>`;
        run = [];
      };
      records.forEach((r) => (r.v === null ? flush() : run.push(r)));
      flush();
      body += `<path d="${p.solid}" class="spark-line" style="stroke:${color}"/>`;
      body += `<path d="${p.dashed}" class="spark-line spark-dash" style="stroke:${color}"/>`;
      records.forEach((r) => {
        if (r.s === "observed" && r.v !== null) body += `<circle cx="${X(r.h).toFixed(1)}" cy="${Y(r.v).toFixed(1)}" r="1.5" style="fill:${color}"/>`;
      });
      const last = records[records.length - 1];
      if (last.v !== null) body += `<circle cx="${X(last.h).toFixed(1)}" cy="${Y(last.v).toFixed(1)}" r="2.6" class="spark-last" style="stroke:${color}"/>`;
    }
    records.forEach((r) => {
      if (r.s === "median" || r.s === "unavailable") body += `<line x1="${X(r.h).toFixed(1)}" x2="${X(r.h).toFixed(1)}" y1="${h - 2.5}" y2="${h}" class="spark-gap"/>`;
    });
    return `<svg class="sparkline" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">${body}</svg>`;
  };

  // ------------------------------------------------------------------ xy
  /**
   * PG.chart(el, opts)
   *  opts.height, opts.margin
   *  opts.x: { domain:[a,b], ticks?, fmt?, label? }
   *  opts.y: { domain?:[a,b], ticks?, fmt?, label?, pad? }
   *  opts.series: [{ name, color, records:[{h, v, s?}], status?:bool, area?:bool, points?:bool, dash? }]
   *  opts.hlines: [{ y, label, cls }]
   *  opts.vband: { from, to, label, cls }
   *  opts.vlines: [{ x, label, cls }]
   *  opts.tooltip: (x) => html | null   (snaps to integer x within domain)
   *  opts.snap: array of x values to snap the tooltip to
   */
  PG.chart = function (el, opts) {
    const W = Math.max(260, Math.floor(el.clientWidth || 600));
    const H = opts.height || 220;
    const m = Object.assign({ t: 14, r: 16, b: 30, l: 44 }, opts.margin || {});
    const iw = W - m.l - m.r, ih = H - m.t - m.b;

    const [x0, x1] = opts.x.domain;
    let yd = opts.y.domain;
    if (!yd) {
      const vals = [];
      opts.series.forEach((s) => s.records.forEach((r) => vals.push(r.v)));
      (opts.hlines || []).forEach((l) => vals.push(l.y));
      yd = extent(vals, opts.y.pad === undefined ? 0.12 : opts.y.pad) || [0, 1];
    }
    const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
    const Y = (v) => m.t + (1 - (v - yd[0]) / (yd[1] - yd[0] || 1)) * ih;
    const xfmt = opts.x.fmt || ((v) => v);
    const yfmt = opts.y.fmt || ((v) => v);

    let g = "";
    // grid + axes
    const yt = opts.y.ticks || niceTicks(yd[0], yd[1], opts.y.count || 4);
    yt.forEach((v) => {
      if (v < yd[0] - 1e-9 || v > yd[1] + 1e-9) return;
      g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`;
      g += `<text class="tick" x="${m.l - 8}" y="${Y(v) + 3.5}" text-anchor="end">${esc(yfmt(v))}</text>`;
    });
    const xt = opts.x.ticks || niceTicks(x0, x1, Math.max(2, Math.floor(iw / 70)));
    xt.forEach((v) => {
      if (v < x0 - 1e-9 || v > x1 + 1e-9) return;
      g += `<line class="xtick" x1="${X(v)}" x2="${X(v)}" y1="${m.t + ih}" y2="${m.t + ih + 4}"/>`;
      g += `<text class="tick" x="${X(v)}" y="${m.t + ih + 17}" text-anchor="middle">${esc(xfmt(v))}</text>`;
    });
    g += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/>`;
    if (opts.y.label) g += `<text class="axis-label" x="${m.l}" y="${m.t - 4}" >${esc(opts.y.label)}</text>`;
    if (opts.x.label) g += `<text class="axis-label" x="${W - m.r}" y="${H - 2}" text-anchor="end">${esc(opts.x.label)}</text>`;

    if (opts.vband) {
      const b = opts.vband, a0 = Math.max(x0, b.from), a1 = Math.min(x1, b.to);
      if (a1 > a0) {
        g += `<rect class="${b.cls || "vband"}" x="${X(a0)}" y="${m.t}" width="${X(a1) - X(a0)}" height="${ih}"/>`;
        if (b.label) g += `<text class="band-label" x="${X(a0) + 6}" y="${m.t + 12}">${esc(b.label)}</text>`;
      }
    }
    (opts.hlines || []).forEach((l) => {
      if (l.y < yd[0] || l.y > yd[1]) return;
      g += `<line class="${l.cls || "hline"}" x1="${m.l}" x2="${W - m.r}" y1="${Y(l.y)}" y2="${Y(l.y)}"/>`;
      if (l.label) g += `<text class="hline-label" x="${W - m.r - 4}" y="${Y(l.y) - 5}" text-anchor="end">${esc(l.label)}</text>`;
    });
    (opts.vlines || []).forEach((l) => {
      g += `<line class="${l.cls || "vline"}" x1="${X(l.x)}" x2="${X(l.x)}" y1="${m.t}" y2="${m.t + ih}"/>`;
      if (l.label) g += `<text class="vline-label" x="${X(l.x) - 4}" y="${m.t + 10}" text-anchor="end">${esc(l.label)}</text>`;
    });

    // series
    opts.series.forEach((s) => {
      const recs = s.records;
      if (s.status) {
        const p = statusPaths(recs, X, Y);
        g += `<path class="series" d="${p.solid}" style="stroke:${s.color}"/>`;
        g += `<path class="series dashed" d="${p.dashed}" style="stroke:${s.color}"/>`;
        recs.forEach((r) => {
          if (r.s === "observed" && r.v !== null) g += `<circle class="pt" cx="${X(r.h)}" cy="${Y(r.v)}" r="2.6" style="fill:${s.color}"/>`;
          if (r.s === "median" || r.s === "unavailable") g += `<line class="gap-tick" x1="${X(r.h)}" x2="${X(r.h)}" y1="${m.t + ih - 6}" y2="${m.t + ih}"/>`;
        });
      } else {
        const pts = recs.filter((r) => r.v !== null);
        if (!pts.length) return;
        const d = pts.map((r, i) => `${i ? "L" : "M"}${X(r.h).toFixed(1)},${Y(r.v).toFixed(1)}`).join("");
        if (s.area) {
          g += `<path class="area" d="${d}L${X(pts[pts.length - 1].h)},${m.t + ih}L${X(pts[0].h)},${m.t + ih}Z" style="fill:${s.color}"/>`;
        }
        g += `<path class="series${s.dash ? " dashed" : ""}" d="${d}" style="stroke:${s.color}"/>`;
        if (s.points) pts.forEach((r) => (g += `<circle class="pt${r.cls ? " " + r.cls : ""}" cx="${X(r.h)}" cy="${Y(r.v)}" r="${r.r || 3}" style="fill:${s.color}"/>`));
      }
    });

    g += `<line class="cursor" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" style="display:none"/>`;
    g += `<rect class="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>`;

    el.classList.add("chart");
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(opts.aria || "Chart")}">${g}</svg><div class="chart-tip" hidden></div>`;

    if (opts.tooltip) {
      const svg = el.querySelector("svg"), tip = el.querySelector(".chart-tip"), cur = el.querySelector(".cursor");
      const snaps = opts.snap || null;
      const move = (ev) => {
        const rect = svg.getBoundingClientRect();
        const px = ((ev.clientX - rect.left) / rect.width) * W;
        let xv = x0 + ((px - m.l) / iw) * (x1 - x0);
        if (snaps) xv = snaps.reduce((a, b) => (Math.abs(b - xv) < Math.abs(a - xv) ? b : a), snaps[0]);
        else xv = Math.round(xv);
        xv = Math.max(x0, Math.min(x1, xv));
        const html = opts.tooltip(xv);
        if (!html) { tip.hidden = true; cur.style.display = "none"; return; }
        cur.setAttribute("x1", X(xv)); cur.setAttribute("x2", X(xv)); cur.style.display = "";
        tip.innerHTML = html; tip.hidden = false;
        const sx = (X(xv) / W) * rect.width;
        const left = sx + 14 + tip.offsetWidth > rect.width ? sx - tip.offsetWidth - 14 : sx + 14;
        tip.style.left = Math.max(0, left) + "px";
        tip.style.top = "8px";
      };
      svg.addEventListener("mousemove", move);
      svg.addEventListener("mouseleave", () => { tip.hidden = true; cur.style.display = "none"; });
    }
    return { X, Y, W, H };
  };
})();
