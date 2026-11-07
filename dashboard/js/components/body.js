/**
 * dashboard/js/components/body.js
 *
 * Front-facing anatomical figure with the eight PhysioGraph variables.
 *
 * Vital signs are placed at their anatomical reference sites. WBC, lactate and
 * creatinine are laboratory values from blood samples, so they live in a
 * separate laboratory callout connected to the antecubital (blood-draw) site.
 * The kidney is drawn only as the conceptual reference for creatinine.
 *
 * Geometry is authored in a centred coordinate frame (x = 0 is the midline,
 * negative x is the viewer's left) and mirrored where the body is symmetric.
 */
(function () {
  "use strict";
  const PG = (window.PG = window.PG || {});
  const esc = (s) => PG.esc(s);

  const VB_W = 740, VB_H = 900, CX = 300;

  // ------------------------------------------------------------ geometry
  // Right half of the body outline, from the neck down to the crotch.
  const OUTLINE_START = [14, 120];
  const OUTLINE = [
    [[15, 134], [17, 145], [24, 152]],
    [[40, 160], [62, 163], [82, 168]],
    [[100, 172], [110, 186], [112, 210]],
    [[115, 238], [117, 268], [119, 300]],
    [[122, 330], [128, 356], [132, 380]],
    [[138, 416], [142, 444], [146, 468]],
    [[152, 486], [157, 506], [155, 526]],
    [[153, 542], [145, 551], [139, 547]],
    [[133, 543], [131, 530], [129, 514]],
    [[127, 498], [124, 486], [121, 474]],
    [[114, 444], [106, 414], [102, 386]],
    [[98, 360], [95, 340], [93, 324]],
    [[90, 298], [88, 272], [86, 250]],
    [[85, 243], [82, 239], [77, 239]],
    [[78, 268], [77, 300], [72, 332]],
    [[68, 354], [65, 372], [67, 392]],
    [[72, 422], [80, 446], [82, 472]],
    [[85, 524], [83, 574], [77, 624]],
    [[73, 662], [75, 692], [75, 720]],
    [[75, 762], [69, 802], [63, 838]],
    [[63, 852], [71, 864], [81, 870]],
    [[87, 874], [85, 882], [73, 882]],
    [[57, 882], [45, 880], [43, 870]],
    [[41, 850], [43, 830], [43, 808]],
    [[43, 768], [39, 734], [37, 708]],
    [[35, 680], [33, 660], [31, 640]],
    [[27, 600], [21, 560], [15, 522]],
    [[11, 502], [5, 492], [0, 490]],
  ];
  const HEAD_START = [0, 18];
  const HEAD = [
    [[22, 18], [38, 36], [38, 62]],
    [[38, 82], [34, 98], [28, 110]],
    [[21, 122], [11, 129], [0, 130]],
  ];

  const p = (x, y) => `${x},${y}`;

  /** Close a right-half outline by mirroring it back to the start. */
  function mirrored(start, segs) {
    let d = `M${p(start[0], start[1])}`;
    segs.forEach(([c1, c2, e]) => (d += `C${p(c1[0], c1[1])} ${p(c2[0], c2[1])} ${p(e[0], e[1])}`));
    for (let i = segs.length - 1; i >= 0; i--) {
      const [c1, c2] = segs[i];
      const prev = i === 0 ? start : segs[i - 1][2];
      d += `C${p(-c2[0], c2[1])} ${p(-c1[0], c1[1])} ${p(-prev[0], prev[1])}`;
    }
    return d + "Z";
  }

  /** Mirror a path string authored on the right (x ↦ −x). */
  function flip(d) {
    return d.replace(/(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/g, (_, x, y) => `${-parseFloat(x)},${y}`);
  }

  function kidney(cx, cy, side) {
    // side = +1 → hilum faces the midline on the viewer's right kidney
    const s = side;
    const P = (dx, dy) => p(cx + dx * s, cy + dy);
    return `M${P(0, -22)}C${P(14, -24)} ${P(20, -10)} ${P(19, 2)}C${P(18, 16)} ${P(10, 24)} ${P(-1, 22)}` +
      `C${P(-8, 20)} ${P(-10, 12)} ${P(-6, 6)}C${P(-3, 2)} ${P(-3, -4)} ${P(-7, -8)}C${P(-11, -14)} ${P(-8, -21)} ${P(0, -22)}Z`;
  }

  const LUNG_R = "M22,178C42,180 58,204 63,236C68,268 68,296 62,312C52,316 42,312 36,306C32,294 26,284 16,278C13,252 12,230 12,206C13,190 16,180 22,178Z";
  const LUNG_L = "M-22,178C-42,180 -58,204 -63,236C-68,268 -68,296 -62,312C-48,316 -28,310 -12,302C-10,272 -9,232 -11,206C-13,190 -16,180 -22,178Z";
  const HEART = "M-10,256C-17,264 -17,280 -9,290C0,300 14,306 26,303C34,297 36,284 31,272C26,260 15,252 3,252C-3,252 -7,253 -10,256Z";

  const VESSELS = [
    "M2,248C0,232 6,220 16,220C26,220 26,236 20,252C14,300 8,360 6,452",
    "M6,452C2,470 -14,488 -26,506C-40,540 -50,600 -54,640C-56,680 -54,740 -52,800",
    "M6,452C12,470 26,488 34,506C46,540 54,600 56,640C58,680 56,740 54,800",
    "M8,222C8,200 10,170 12,140",
    "M2,226C-4,200 -10,170 -12,140",
    "M4,226C-24,212 -60,196 -88,206C-96,230 -100,270 -103,316C-110,360 -120,420 -132,462",
    "M14,222C40,210 70,200 92,208C100,232 104,270 107,316C114,360 124,420 136,462",
  ];
  const BRACHIAL_L = "M-88,206C-96,230 -100,270 -103,316";

  const CONTOURS = [
    "M6,160C30,158 60,164 84,170", flip("M6,160C30,158 60,164 84,170"),
    "M10,120C12,135 9,148 5,158", flip("M10,120C12,135 9,148 5,158"),
    "M0,302C20,312 44,322 62,330", flip("M0,302C20,312 44,322 62,330"),
    "M70,440C52,468 28,486 10,494", flip("M70,440C52,468 28,486 10,494"),
    "M44,630C50,642 62,642 68,630", flip("M44,630C50,642 62,642 68,630"),
    "M96,322C102,318 110,314 118,312", flip("M96,322C102,318 110,314 118,312"),
    "M0,312L0,470",
    "M-64,318C-40,302 -14,304 0,312C14,304 40,302 64,318",
  ];
  const EARS = ["M37,64C44,62 46,72 45,80C44,88 40,92 36,90", flip("M37,64C44,62 46,72 45,80C44,88 40,92 36,90")];
  const CUFF = "M-118,248C-108,252 -96,252 -86,249L-88,284C-98,287 -110,287 -120,283Z";

  const DRAW_SITE = [104, 316];
  const LAB_NAMES = { wbc: "WBC", lactate: "Lactate", creatinine: "Creatinine" };
  const LAB = { x: 180, y: 470, w: 250, rowY: 532, rowH: 36, keys: ["wbc", "lactate", "creatinine"] };

  // Marker sites and callout slots (side, y).
  const SITES = {
    temperature: { x: 0, y: 44, slot: ["L", 70], regions: ["forehead"] },
    resp_rate:   { x: -36, y: 258, slot: ["L", 214], regions: ["lungs", "airway"] },
    spo2:        { x: 38, y: 206, slot: ["R", 160], regions: ["lungs", "vessels"] },
    heart_rate:  { x: 10, y: 277, slot: ["R", 286], regions: ["heart"] },
    sbp:         { x: -103, y: 266, slot: ["L", 360], regions: ["cuff", "brachial"] },
    creatinine:  { x: 36, y: 386, slot: null, regions: ["kidneys", "drawsite"], lab: true },
    wbc:         { regions: ["vessels", "drawsite"], lab: true },
    lactate:     { regions: ["vessels", "drawsite"], lab: true },
  };
  PG.BODY_SITES = SITES;

  // ------------------------------------------------------------- helpers
  function statusText(c, key, rec, t) {
    if (rec.s === "observed") return `Observed · ${PG.fmt.hour(c, t)}`;
    if (rec.s === "carried") return `Carried · obs. ${PG.fmt.hour(c, rec.o)}`;
    if (rec.s === "median") return "Training-median fill";
    return "Unavailable";
  }

  function markerGlyph(node, rec, selected, lab) {
    const cls = `m-${rec.s}`;
    const color = PG.SYSTEMS[node.system].color;
    const shape = lab
      ? `<rect class="m-ring" x="-7" y="-7" width="14" height="14" rx="3" transform="rotate(45)"/>` +
        (rec.s === "observed" ? `<rect class="m-core" x="-2.6" y="-2.6" width="5.2" height="5.2" rx="1" transform="rotate(45)"/>` : "")
      : `<circle class="m-ring" r="8"/>` + (rec.s === "observed" ? `<circle class="m-core" r="3.2"/>` : "");
    return `<g class="marker-glyph ${cls}${selected ? " is-selected" : ""}" style="--mc:${color}">` +
      `<circle class="m-halo" r="16"/>${shape}</g>`;
  }

  function callout(c, node, rec, t, site) {
    const [side, sy] = site.slot;
    const left = side === "L";
    const endX = left ? -288 : 432;
    const elbowX = left ? -168 : 188;
    const txtX = left ? endX : elbowX + 8;
    const anchor = "start";
    const val = rec.v === null ? "—" : PG.fmt.value(node.key, rec.v);
    const color = PG.SYSTEMS[node.system].color;
    return `<g class="callout" style="--mc:${color}">
      <path class="callout-line" d="M${site.x},${site.y}L${elbowX},${sy}L${endX},${sy}"/>
      <circle class="callout-end" cx="${endX}" cy="${sy}" r="2.5"/>
      <text class="co-name" x="${txtX}" y="${sy - 34}" text-anchor="${anchor}">${esc(node.label)}</text>
      <text class="co-value${rec.s === "median" || rec.s === "unavailable" ? " is-imputed" : ""}" x="${txtX}" y="${sy - 9}" text-anchor="${anchor}">${esc(val)}<tspan class="co-unit" dx="5">${rec.v === null ? "" : esc(node.unitShort)}</tspan></text>
      <text class="co-status st-${rec.s}" x="${txtX}" y="${sy + 18}" text-anchor="${anchor}">${esc(statusText(c, node.key, rec, t))}</text>
    </g>`;
  }

  // --------------------------------------------------------------- render
  /**
   * PG.renderBody(el, { case, t, selected, mode: "interactive" | "annotated", onSelect })
   */
  PG.renderBody = function (el, o) {
    const c = o.case, t = o.t, sel = o.selected, annotated = o.mode === "annotated";
    const active = new Set(sel ? SITES[sel].regions : []);
    const hl = (r) => (active.has(r) ? " hl" : "");

    let s = `<svg class="body-svg" viewBox="0 0 ${VB_W} ${VB_H}" role="group" aria-label="Anatomical map of the eight PhysioGraph variables">
      <defs>
        <linearGradient id="pg-skin" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" style="stop-color:var(--body-edge)"/><stop offset="0.5" style="stop-color:var(--body-mid)"/><stop offset="1" style="stop-color:var(--body-edge)"/>
        </linearGradient>
        <radialGradient id="pg-heat"><stop offset="0" style="stop-color:var(--sys-systemic);stop-opacity:.5"/><stop offset="1" style="stop-color:var(--sys-systemic);stop-opacity:0"/></radialGradient>
        <filter id="pg-glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <g transform="translate(${CX},0)">
        <path class="body-fill" d="${mirrored(OUTLINE_START, OUTLINE)}"/>
        <path class="body-fill" d="${mirrored(HEAD_START, HEAD)}"/>
        ${EARS.map((d) => `<path class="contour ear" d="${d}"/>`).join("")}
        ${CONTOURS.map((d) => `<path class="contour" d="${d}"/>`).join("")}
        <ellipse class="region forehead${hl("forehead")}" cx="0" cy="44" rx="30" ry="16" fill="url(#pg-heat)"/>
        <g class="region vessels${hl("vessels")}">${VESSELS.map((d) => `<path d="${d}"/>`).join("")}</g>
        <path class="region brachial${hl("brachial")}" d="${BRACHIAL_L}"/>
        <path class="region airway${hl("airway")}" d="M0,140L0,206M0,206C-6,214 -12,220 -18,228M0,206C6,214 12,220 18,228"/>
        <g class="region lungs${hl("lungs")}">
          <path class="lung" d="${LUNG_L}"/><path class="lung" d="${LUNG_R}"/>
          <path class="fissure" d="M-62,252C-46,250 -28,242 -14,230"/><path class="fissure" d="M-64,292C-48,272 -30,256 -14,244"/>
          <path class="fissure" d="M62,262C48,256 32,246 16,232"/>
        </g>
        <g class="region heart${hl("heart")}"><path d="${HEART}"/><path class="groove" d="M4,256C10,270 17,288 24,301"/></g>
        <g class="region kidneys${hl("kidneys")}"><path d="${kidney(-36, 394, -1)}"/><path d="${kidney(36, 386, 1)}"/></g>
        <path class="region cuff${hl("cuff")}" d="${CUFF}"/>`;

    // Laboratory callout --------------------------------------------------
    const labSel = sel && SITES[sel].lab;
    s += `<path class="lab-link${labSel ? " hl" : ""}" d="M${DRAW_SITE[0]},${DRAW_SITE[1]}L160,${DRAW_SITE[1]}Q168,${DRAW_SITE[1]} 168,${DRAW_SITE[1] + 8}L168,${LAB.y + 14}Q168,${LAB.y + 22} ${LAB.x},${LAB.y + 22}"/>`;
    s += `<g class="region drawsite${hl("drawsite")}" transform="translate(${DRAW_SITE[0]},${DRAW_SITE[1]})"><path d="M0,-7C3,-3 5,0 5,3A5,5 0 1 1 -5,3C-5,0 -3,-3 0,-7Z"/></g>`;
    const crRow = LAB.rowY + LAB.keys.indexOf("creatinine") * LAB.rowH + LAB.rowH / 2;
    if (annotated || sel === "creatinine") {
      s += `<path class="kidney-ref" d="M${SITES.creatinine.x + 10},${SITES.creatinine.y + 6}C110,${SITES.creatinine.y + 60} 120,${crRow} ${LAB.x},${crRow}"/>`;
    }
    const labH = 64 + LAB.keys.length * LAB.rowH + 8;
    s += `<g class="lab-panel">
      <rect class="lab-bg" x="${LAB.x}" y="${LAB.y}" width="${LAB.w}" height="${labH}" rx="8"/>
      <text class="lab-title" x="${LAB.x + 16}" y="${LAB.y + 26}">BLOOD LABORATORY</text>
      <text class="lab-sub" x="${LAB.x + 16}" y="${LAB.y + 46}">Sampled values · not organ-measured</text>`;
    LAB.keys.forEach((key, i) => {
      const node = PG.NODE[key], rec = PG.record(c, key, t);
      const y = LAB.rowY + i * LAB.rowH, isSel = sel === key;
      const val = rec.v === null ? "—" : PG.fmt.value(key, rec.v);
      s += `<g class="lab-row marker${isSel ? " is-selected" : ""}" data-key="${key}" tabindex="0" role="button" aria-pressed="${isSel}" aria-label="${esc(node.label)}">
        <rect class="lab-row-bg" x="${LAB.x + 6}" y="${y}" width="${LAB.w - 12}" height="${LAB.rowH - 2}" rx="5"/>
        <g transform="translate(${LAB.x + 24},${y + LAB.rowH / 2 - 1})">${markerGlyph(node, rec, isSel, true)}</g>
        <text class="lab-name" x="${LAB.x + 42}" y="${y + 15}">${esc(LAB_NAMES[key])}</text>
        <text class="lab-status st-${rec.s}" x="${LAB.x + 42}" y="${y + 29}">${esc(PG.STATUS[rec.s].label)}</text>
        <text class="lab-value${rec.s === "median" || rec.s === "unavailable" ? " is-imputed" : ""}" x="${LAB.x + LAB.w - 64}" y="${y + 22}" text-anchor="end">${esc(val)}</text>
        <text class="lab-unit" x="${LAB.x + LAB.w - 58}" y="${y + 22}">${rec.v === null ? "" : esc(node.unitShort)}</text>
      </g>`;
    });
    s += `</g>`;

    // Callouts -------------------------------------------------------------
    PG.NODES.forEach((node) => {
      const site = SITES[node.key];
      if (!site.slot) return;
      if (annotated || sel === node.key) s += callout(c, node, PG.record(c, node.key, t), t, site);
    });

    // Body markers -----------------------------------------------------------
    PG.NODES.forEach((node) => {
      const site = SITES[node.key];
      if (site.x === undefined) return;
      const rec = PG.record(c, node.key, t), isSel = sel === node.key;
      s += `<g class="marker body-marker" data-key="${node.key}" transform="translate(${site.x},${site.y})" tabindex="0" role="button" aria-pressed="${isSel}" aria-label="${esc(node.label)}${site.lab ? " (kidney reference)" : ""}">
        ${markerGlyph(node, rec, isSel, !!site.lab)}<circle class="m-hit" r="18"/></g>`;
    });

    s += `</g></svg><div class="body-tip" role="tooltip" hidden></div>`;
    el.innerHTML = s;

    // Interaction ------------------------------------------------------------
    const tip = el.querySelector(".body-tip");
    el.querySelectorAll(".marker").forEach((m) => {
      const key = m.dataset.key;
      const choose = () => o.onSelect && o.onSelect(key);
      m.addEventListener("click", choose);
      m.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); choose(); }
      });
      const show = () => {
        const node = PG.NODE[key], rec = PG.record(c, key, t);
        const val = rec.v === null ? "—" : `${PG.fmt.value(key, rec.v)} ${node.unitShort}`;
        tip.innerHTML = `<div class="tip-name">${esc(node.label)}</div><div class="tip-val">${esc(val)}</div><div class="tip-st st-${rec.s}">${esc(statusText(c, key, rec, t))}</div>`;
        tip.hidden = false;
        const er = el.getBoundingClientRect(), mr = m.getBoundingClientRect();
        let x = mr.left - er.left + mr.width / 2, y = mr.top - er.top;
        tip.style.left = Math.min(Math.max(8, x - tip.offsetWidth / 2), er.width - tip.offsetWidth - 8) + "px";
        tip.style.top = Math.max(0, y - tip.offsetHeight - 8) + "px";
      };
      m.addEventListener("mouseenter", show);
      m.addEventListener("focus", show);
      m.addEventListener("mouseleave", () => (tip.hidden = true));
      m.addEventListener("blur", () => (tip.hidden = true));
    });
  };
})();
