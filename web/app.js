/* Ripen (antigo AvocadoCheck): mobile web app for the Hass avocado ripening PoC.
 * Talks to server.py (/api/info, /api/analyze). History, profile and settings
 * live in this device's localStorage. */

'use strict';

/* Base URL of the model server. Empty = same address as the page (python server.py). */
const API = String(window.AVOCHECK_API || '').replace(/\/+$/, '');

/* ------------------------------------------------------------------ storage */
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem('avc:' + key); return v ? JSON.parse(v) : fallback; }
    catch (e) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem('avc:' + key, JSON.stringify(value)); return true; }
    catch (e) { toast('Sem espaço para salvar. Apague análises antigas no Perfil.'); return false; }
  },
};

const DEFAULT_SETTINGS = { condition: 'T10', exportDays: 10, domesticDays: 4, margin: 2, quantile: 0.9, review: 0.6 };

const S = {
  profile: store.get('profile', null),
  settings: Object.assign({}, DEFAULT_SETTINGS, store.get('settings', {})),
  analyses: store.get('analyses', []),
  info: null,
  online: null,
  pending: null,
  cam: { stream: null, facing: 'environment', torch: false },
};

function saveAnalyses() { store.set('analyses', S.analyses); }

/* ------------------------------------------------------------- vocabulary */
const STAGE_NAMES = { 1: 'Verde e firme', 2: 'Começando a amadurecer', 3: 'Amadurecendo', 4: 'Pronto para comer', 5: 'Passando do ponto' };
const STAGE_COLORS = { 1: '#2F7D32', 2: '#6BA539', 3: '#B49A2F', 4: '#7A4E3A', 5: '#4A2436' };
const CONDITIONS = [
  { code: 'T10', label: 'Câmara fria', sub: '10 °C', icon: 'thermo' },
  { code: 'T20', label: 'Climatizado', sub: '20 °C', icon: 'box' },
  { code: 'Tam', label: 'Ambiente', sub: 'sem controle', icon: 'leaf' },
];
const STATUS = {
  green: { label: 'Verde', chip: 'chip-dark' },
  ripening: { label: 'Em maturação', chip: 'chip-amber' },
  ready: { label: 'Pronto', chip: 'chip-green' },
  passed: { label: 'Passou do ponto', chip: 'chip-wine' },
};
const DEST = {
  export: { name: 'Exportação marítima', chip: 'Ideal para exportação', icon: 'ship', cls: 'chip-green' },
  domestic: { name: 'Mercado interno', chip: 'Ideal para mercado interno', icon: 'truck', cls: 'chip-green' },
  local: { name: 'Venda local', chip: 'Venda local imediata', icon: 'store', cls: 'chip-amber' },
  processing: { name: 'Processamento', chip: 'Destinar ao processamento', icon: 'factory', cls: 'chip-wine' },
  review: { name: 'Revisão humana', chip: 'Precisa de revisão', icon: 'eye', cls: 'chip-amber' },
};

/* ------------------------------------------------------------- helpers */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const DAY = 86400000;
const MONTHS = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'];

function fmtDate(ts) {
  const d = new Date(ts);
  return d.toLocaleDateString('pt-BR') + ' • ' + d.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
}
function fmtDay(ts) { const d = new Date(ts); return `${String(d.getDate()).padStart(2, '0')}/${String(d.getMonth() + 1).padStart(2, '0')}`; }
function fmtLong(ts) { const d = new Date(ts); return `${d.getDate()} de ${MONTHS[d.getMonth()]}`; }
function fmtRange(a, b) {
  const da = new Date(a), db = new Date(b);
  if (da.toDateString() === db.toDateString()) return fmtLong(a);
  if (da.getMonth() === db.getMonth()) return `${da.getDate()} a ${db.getDate()} de ${MONTHS[db.getMonth()]}`;
  return `${fmtLong(a)} a ${fmtLong(b)}`;
}
function startOfDay(ts) { const d = new Date(ts); d.setHours(0, 0, 0, 0); return d.getTime(); }
function plural(n, one, many) { return `${n} ${n === 1 ? one : many}`; }
function initials(name) { return (name || '?').trim().split(/\s+/).slice(0, 2).map((p) => p[0]).join('').toUpperCase(); }

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove('show'), 2600);
}

function percentile(values, q) {
  const v = [...values].sort((a, b) => a - b);
  if (!v.length) return 0;
  const pos = (v.length - 1) * q / 100;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

/* ------------------------------------------------------------- the science */
function readyStage() { return (S.info && S.info.ready_stage) || 4; }
function mae() { return (S.info && S.info.quality && S.info.quality.days_mae) || 1; }

/* Maturity % = expected stage mapped from stage 1 (0%) to the ready stage (100%). */
function fruitMaturity(f) {
  const e = f.probs.reduce((acc, p, i) => acc + p * (i + 1), 0);
  return clamp((e - 1) / (readyStage() - 1), 0, 1) * 100;
}
function fruitStatus(f) {
  if (f.stage >= 5) return 'passed';
  const m = fruitMaturity(f);
  if (m >= 85) return 'ready';
  if (m >= 35) return 'ripening';
  return 'green';
}
/* Same rule as pipeline.decide_destination. */
function decide(fruits, days) {
  const s = S.settings;
  const q = percentile(days, 100 * (1 - s.quantile));
  const conf = fruits.reduce((a, f) => a + f.confidence, 0) / fruits.length;
  if (conf < s.review) return 'review';
  if (q >= s.exportDays + s.margin) return 'export';
  if (q >= s.domesticDays + s.margin) return 'domestic';
  if (fruits.filter((f) => f.stage >= 5).length / fruits.length >= 0.5) return 'processing';
  return 'local';
}
function elapsedDays(a) { return Math.max(0, (Date.now() - a.date) / DAY); }

/* Everything the screens need about one analysis, recomputed with the current settings. */
function summarize(a) {
  const el = elapsedDays(a);
  const daysNow = a.fruits.map((f) => Math.max(0, f.days - el));
  const worst = percentile(daysNow, 100 * (1 - S.settings.quantile));
  const maturity = a.fruits.reduce((acc, f) => acc + fruitMaturity(f), 0) / a.fruits.length;
  const passedShare = a.fruits.filter((f) => fruitStatus(f) === 'passed').length / a.fruits.length;
  const status = passedShare >= 0.5 ? 'passed' : maturity >= 85 ? 'ready' : maturity >= 35 ? 'ripening' : 'green';
  const destination = decide(a.fruits, daysNow);
  const conf = a.fruits.reduce((acc, f) => acc + f.confidence, 0) / a.fruits.length;
  return { daysNow, worst, maturity, status, destination, conf, readyAt: a.date + (worst + el) * DAY };
}
function daysRange(d) {
  if (d < 0.5) return null;
  const m = mae();
  return [Math.max(0, Math.floor(d - m)), Math.ceil(d + m)];
}
function rangeText(d) {
  const r = daysRange(d);
  if (!r) return 'Já está no ponto';
  if (r[0] === r[1]) return `Em ${plural(r[0], 'dia', 'dias')}`;
  return `Em ${r[0]} a ${r[1]} dias`;
}
function lotName(a) { return `Lote ${a.lot}${a.plot ? ' - Talhão ' + a.plot : ''}`; }
function condLabel(code) { const c = CONDITIONS.find((x) => x.code === code); return c ? `${c.label} (${c.sub})` : code; }

function idealConditions(sm) {
  if (sm.destination === 'export' || sm.destination === 'domestic') {
    return 'Para segurar o amadurecimento até o embarque, mantenha os frutos refrigerados (por volta de 5 a 7 °C), sem quebrar a cadeia de frio.';
  }
  if (sm.destination === 'processing') {
    return 'Separe estes frutos do restante do lote e encaminhe logo para polpa, guacamole ou óleo, antes que estraguem.';
  }
  return 'Mantenha em local fresco e bem ventilado, à sombra e com temperatura entre 18 °C e 25 °C. Evite exposição direta ao sol.';
}

/* ------------------------------------------------------------- server */
function fetchTimeout(url, opts = {}, ms = 90000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), ms);
  return fetch(url, Object.assign({}, opts, { signal: ctl.signal })).finally(() => clearTimeout(t));
}

async function loadInfo() {
  /* A free Render server sleeps after 15 minutes idle and takes up to a minute to wake up. */
  const slow = setTimeout(() => {
    const el = document.getElementById('wake');
    if (el) el.hidden = false;
  }, 3500);
  try {
    const r = await fetchTimeout(API + '/api/info', { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    S.info = await r.json();
    S.online = true;
    const codes = S.info.conditions.map((c) => c.code);
    if (!codes.includes(S.settings.condition)) { S.settings.condition = codes[0]; store.set('settings', S.settings); }
  } catch (e) {
    S.online = false;
  } finally {
    clearTimeout(slow);
  }
}

async function analyze(blob, meta) {
  const fd = new FormData();
  fd.append('file', blob, 'foto.jpg');
  fd.append('condition', meta.condition);
  const r = await fetchTimeout(API + '/api/analyze', { method: 'POST', body: fd }, 120000);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || 'O servidor não respondeu. Confira se o server.py está rodando.');
  return data;
}

/* ------------------------------------------------------------- routing */
const app = $('#app');
function go(path, replace = false) {
  stopCamera();
  closeSheet();
  if (replace) history.replaceState(null, '', '#/' + path);
  else history.pushState(null, '', '#/' + path);
  render();
}
function back(parent = 'home') { go(parent, true); }
window.addEventListener('popstate', () => { stopCamera(); closeSheet(); render(); });

function currentRoute() {
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  return { name: parts[0] || 'home', arg: decodeURIComponent(parts[1] || '') };
}

function render() {
  const { name, arg } = currentRoute();
  if (!S.profile && !['splash', 'welcome'].includes(name)) { history.replaceState(null, '', '#/welcome'); return render(); }
  const screen = SCREENS[name] || SCREENS.home;
  app.innerHTML = screen.html(arg);
  if (screen.mount) screen.mount(arg);
  const sc = $('.scroll', app);
  if (sc) sc.scrollTop = 0;
}

function nav(active) {
  const items = [['home', 'Início', 'home'], ['history', 'Histórico', 'history'], ['tips', 'Dicas', 'bulb'], ['profile', 'Perfil', 'user']];
  return `<nav class="nav">${items.map(([r, label, ic]) => `
    <button data-go="${r}" class="${active === r ? 'on' : ''}" aria-current="${active === r ? 'page' : 'false'}">
      ${icon(active === r && ic === 'home' ? 'homeFill' : ic, 22)}<span>${label}</span></button>`).join('')}</nav>`;
}
function topbar(title, { left = false, backTo = 'home', right = '' } = {}) {
  return `<header class="topbar ${left ? 'left' : ''}">
    <button class="icon-btn" data-back="${backTo}" aria-label="Voltar">${icon(left ? 'arrowLeft' : 'chevronLeft', 22)}</button>
    <h1>${title}</h1><div>${right}</div></header>`;
}

/* Delegated clicks for [data-go] and [data-back]. */
document.addEventListener('click', (ev) => {
  const g = ev.target.closest('[data-go]');
  if (g) { ev.preventDefault(); go(g.dataset.go); return; }
  const b = ev.target.closest('[data-back]');
  if (b) { ev.preventDefault(); back(b.dataset.back); }
});

/* ------------------------------------------------------------- UI pieces */
function ring(pct, color, size = 76) {
  const r = (size - 10) / 2, c = 2 * Math.PI * r, off = c * (1 - clamp(pct, 0, 100) / 100);
  return `<svg class="ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img" aria-label="${Math.round(pct)}% de maturação">
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="#EDF2EE" stroke-width="8"/>
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${color}" stroke-width="8" stroke-linecap="round"
      stroke-dasharray="${c}" stroke-dashoffset="${off}" transform="rotate(-90 ${size / 2} ${size / 2})"/>
    <text x="50%" y="53%" text-anchor="middle" dominant-baseline="middle" font-size="${size * 0.22}" fill="#18231B">${Math.round(pct)}%</text></svg>`;
}
function ringColor(status) { return { green: '#2F7D32', ripening: '#E0A526', ready: '#22A45D', passed: '#6B2D45' }[status]; }
function statusChip(status) { return `<span class="chip ${STATUS[status].chip}">${STATUS[status].label}</span>`; }
function destChip(dest, lg = false) {
  return `<span class="chip ${DEST[dest].cls} ${lg ? 'chip-lg' : ''}">${icon(dest === 'review' ? 'eye' : 'checkCircle', 15)}${DEST[dest].chip}</span>`;
}

function analysisRow(a, showPct = false) {
  const sm = summarize(a);
  return `<button class="row" data-go="details/${a.id}">
    <img class="thumb" src="${a.fruits[0].crop}" alt="">
    <div><p class="row-title">${esc(lotName(a))}</p><p class="row-sub">${fmtDate(a.date)}${a.fruits.length > 1 ? ` • ${a.fruits.length} frutos` : ''}</p></div>
    <div class="row-end">${showPct ? `<span class="pct" style="color:${ringColor(sm.status)}">${Math.round(sm.maturity)}%</span>` : ''}${statusChip(sm.status)}</div>
  </button>`;
}

/* ------------------------------------------------------------- sheets */
function openSheet(html, onMount) {
  const root = $('#sheet-root');
  root.innerHTML = `<div class="sheet-backdrop" data-close></div><div class="sheet" role="dialog" aria-modal="true"><div class="grab"></div>${html}</div>`;
  $('[data-close]', root).addEventListener('click', closeSheet);
  if (onMount) onMount($('.sheet', root));
}
function closeSheet() { const r = $('#sheet-root'); if (r) r.innerHTML = ''; }

/* ------------------------------------------------------------- camera */
async function startCamera() {
  const video = $('#cam-video');
  if (!video || !navigator.mediaDevices || !window.isSecureContext) return false;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: S.cam.facing, width: { ideal: 1920 } }, audio: false });
    S.cam.stream = stream;
    video.srcObject = stream;
    await video.play();
    $('#cam-hint').hidden = true;
    const track = stream.getVideoTracks()[0];
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    $('#cam-torch').hidden = !caps.torch;
    return true;
  } catch (e) {
    return false;
  }
}
function stopCamera() {
  if (S.cam.stream) { S.cam.stream.getTracks().forEach((t) => t.stop()); S.cam.stream = null; }
}
function captureFrame() {
  const video = $('#cam-video');
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  return new Promise((res) => c.toBlob(res, 'image/jpeg', 0.92));
}
function pickFile(id) {
  const input = $(id);
  input.value = '';
  input.onchange = () => { if (input.files && input.files[0]) startIdentify(input.files[0]); };
  input.click();
}

/* ------------------------------------------------------------- analysis flow */
function startIdentify(blob) {
  S.pending = { blob, url: URL.createObjectURL(blob) };
  stopCamera();
  const last = S.analyses[0];
  const lotDefault = last ? last.lot : '1';
  const plotDefault = last ? last.plot : '1';
  openSheet(`
    <h3>Identifique o lote</h3>
    <p class="sub">Assim você encontra esta análise depois no histórico.</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div class="field"><label for="f-lot">Lote</label><input id="f-lot" inputmode="numeric" value="${esc(lotDefault)}"></div>
      <div class="field"><label for="f-plot">Talhão</label><input id="f-plot" inputmode="numeric" value="${esc(plotDefault)}"></div>
    </div>
    <div class="field"><label>Onde os frutos estão guardados?</label>
      <div class="seg" id="f-cond">${conditionButtons(S.settings.condition)}</div>
      <small>O abacate amadurece mais rápido no calor, então isso muda a previsão de dias.</small></div>
    <button class="btn btn-primary" id="f-go">${icon('scan', 20)} Analisar foto</button>`,
  (sheet) => {
    bindSeg($('#f-cond', sheet));
    $('#f-go', sheet).addEventListener('click', () => {
      const meta = {
        lot: $('#f-lot', sheet).value.trim() || '1',
        plot: $('#f-plot', sheet).value.trim(),
        condition: $('#f-cond .on', sheet).dataset.code,
      };
      S.settings.condition = meta.condition;
      store.set('settings', S.settings);
      S.pending.meta = meta;
      go('analyzing', true);
    });
  });
}
function conditionButtons(active) {
  const codes = S.info ? S.info.conditions.map((c) => c.code) : CONDITIONS.map((c) => c.code);
  return CONDITIONS.filter((c) => codes.includes(c.code)).map((c) => `
    <button type="button" data-code="${c.code}" class="${c.code === active ? 'on' : ''}">${icon(c.icon, 20)}<span>${c.label}</span><small class="muted">${c.sub}</small></button>`).join('');
}
function bindSeg(root) {
  $$('button', root).forEach((b) => b.addEventListener('click', () => {
    $$('button', root).forEach((x) => x.classList.remove('on'));
    b.classList.add('on');
  }));
}

/* ------------------------------------------------------------- screens */
const SCREENS = {};

SCREENS.splash = {
  html: () => `<section class="screen splash-land">
    <div class="brand">${logoMark(78, false)}<h2>Ripen</h2>
    <p>Tecnologia que ajuda o seu abacate a chegar mais longe.</p>
    <p id="wake" class="wake" hidden><span class="spinner"></span>Conectando ao servidor. Na primeira vez do dia isso pode levar até 1 minuto.</p></div>
    <img class="land" src="assets/landscape.jpg" alt=""></section>`,
};

SCREENS.welcome = {
  html: () => `<section class="screen welcome">
    <div class="photo" role="img" aria-label="Abacate Hass no pé"></div>
    <div class="brand">${logoMark(72, true)}<h2>Ripen</h2><p>Do campo ao mercado, no ponto certo.</p></div>
    <div class="cta"><button class="btn btn-primary" id="start">Começar ${icon('arrowRight', 20)}</button></div></section>`,
  mount() {
    $('#start').addEventListener('click', () => {
      openSheet(`<h3>Como podemos te chamar?</h3><p class="sub">Seu nome fica só neste aparelho.</p>
        <div class="field"><label for="p-name">Nome</label><input id="p-name" autocomplete="name" placeholder="Ex.: Gabriel"></div>
        <div class="field"><label for="p-mail">E-mail (opcional)</label><input id="p-mail" type="email" autocomplete="email"></div>
        <button class="btn btn-primary" id="p-save">Continuar ${icon('arrowRight', 20)}</button>`,
      (sheet) => {
        $('#p-name', sheet).focus();
        $('#p-save', sheet).addEventListener('click', () => {
          const name = $('#p-name', sheet).value.trim();
          if (!name) { toast('Digite seu nome para continuar.'); return; }
          S.profile = { name, email: $('#p-mail', sheet).value.trim() };
          store.set('profile', S.profile);
          go('home', true);
        });
      });
    });
  },
};

SCREENS.home = {
  html: () => {
    const first = (S.profile.name || '').split(' ')[0];
    const recent = S.analyses.slice(0, 3);
    return `<section class="screen">
    <div class="scroll" style="padding-top:calc(26px + env(safe-area-inset-top))">
      <p class="hello">Olá, ${esc(first)}! 👋</p>
      <p class="hello-sub">Vamos verificar a maturação dos seus abacates e planejar o melhor momento para o envio.</p>
      ${S.online === false ? `<div class="warnbox" style="margin-bottom:16px">${icon('wifi', 20)}<div>Sem conexão com o servidor dos modelos. Se o app estiver online, espere um minuto e recarregue a página (o servidor grátis pode estar acordando).</div></div>` : ''}
      <div class="actions">
        <button class="action" data-go="camera">${icon('camera', 34)}<b>Tirar foto</b><small>Use a câmera para analisar o fruto agora</small></button>
        <button class="action" id="home-gallery">${icon('image', 34)}<b>Escolher da galeria</b><small>Use uma foto já existente no seu celular</small></button>
      </div>
      <div class="section-head"><h3>Últimas análises</h3>${S.analyses.length ? `<button class="link" data-go="history">Ver todas ${icon('chevronRight', 16)}</button>` : ''}</div>
      ${recent.length ? `<div class="list">${recent.map((a) => analysisRow(a)).join('')}</div>`
        : `<div class="list empty">${icon('avocado', 40)}Nenhuma análise ainda.<br>Tire uma foto de um abacate para começar.</div>`}
    </div>${nav('home')}</section>`;
  },
  mount() { $('#home-gallery').addEventListener('click', () => pickFile('#pick-gallery')); },
};

SCREENS.camera = {
  html: () => `<section class="screen camera">
    <header class="topbar"><button class="icon-btn" data-back="home" aria-label="Fechar">${icon('x', 24)}</button><h1></h1>
      <button class="icon-btn" id="cam-torch" aria-label="Lanterna" hidden>${icon('zap', 22)}</button></header>
    <div class="viewfinder">
      <video id="cam-video" playsinline muted></video>
      <div class="hint" id="cam-hint">${icon('camera', 46)}<span>Toque no botão branco para abrir a câmera do celular.</span></div>
      <div class="frame"><i></i><i></i><i></i><i></i></div>
      <p class="cam-tip">Enquadre um abacate inteiro, com boa luz e fundo claro</p>
    </div>
    <div class="cam-controls">
      <button class="cam-side" id="cam-gallery" aria-label="Galeria">${icon('image', 22)}</button>
      <button class="shutter" id="cam-shot" aria-label="Tirar foto"><span></span></button>
      <button class="cam-side" id="cam-flip" aria-label="Virar câmera">${icon('flip', 22)}</button>
    </div>
    <div class="cam-tabs"><button class="on">Foto</button><button id="cam-tab-gallery">Galeria</button></div>
  </section>`,
  async mount() {
    const live = await startCamera();
    $('#cam-shot').addEventListener('click', async () => {
      if (S.cam.stream) startIdentify(await captureFrame());
      else pickFile('#pick-camera');
    });
    $('#cam-gallery').addEventListener('click', () => pickFile('#pick-gallery'));
    $('#cam-tab-gallery').addEventListener('click', () => pickFile('#pick-gallery'));
    $('#cam-flip').addEventListener('click', async () => {
      S.cam.facing = S.cam.facing === 'environment' ? 'user' : 'environment';
      stopCamera();
      if (!(await startCamera())) toast('Não foi possível trocar de câmera.');
    });
    $('#cam-torch').addEventListener('click', async () => {
      const track = S.cam.stream && S.cam.stream.getVideoTracks()[0];
      if (!track) return;
      S.cam.torch = !S.cam.torch;
      try { await track.applyConstraints({ advanced: [{ torch: S.cam.torch }] }); } catch (e) { toast('Lanterna indisponível.'); }
    });
    if (!live) $('#cam-flip').style.visibility = 'hidden';
  },
};

SCREENS.analyzing = {
  html: () => {
    if (!S.pending) return SCREENS.home.html();
    return `<section class="screen analyzing">
      <div class="preview"><img src="${S.pending.url}" alt="Foto enviada"><div class="scan-line"></div></div>
      <div class="spinner" aria-hidden="true"></div>
      <h2>Analisando a casca do abacate...</h2>
      <p>Procurando os frutos e estimando a maturação.</p></section>`;
  },
  async mount() {
    if (!S.pending) { go('home', true); return; }
    const { blob, meta } = S.pending;
    try {
      const data = await analyze(blob, meta);
      const a = {
        id: Date.now().toString(36),
        date: Date.now(),
        lot: meta.lot, plot: meta.plot, condition: data.condition,
        fruits: data.fruits.map((f) => ({ crop: f.crop, cam: f.cam, stage: f.stage, probs: f.probs, confidence: f.confidence, days: f.days })),
      };
      S.analyses.unshift(a);
      saveAnalyses();
      S.pending = null;
      go('result/' + a.id, true);
    } catch (e) {
      if (e instanceof TypeError || e.name === 'AbortError') { S.online = false; e = new Error('Sem conexão com o servidor dos modelos. Espere um minuto e tente de novo.'); }
      app.innerHTML = `<section class="screen analyzing">${icon('alert', 46)}<h2>Não deu certo</h2>
        <p>${esc(e.message)}</p><button class="btn btn-primary" style="max-width:280px" data-go="camera">Tentar outra foto</button>
        <button class="btn btn-ghost" style="max-width:280px" data-go="home">Voltar ao início</button></section>`;
    }
  },
};

function findAnalysis(id) { return S.analyses.find((a) => a.id === id); }
function missing() {
  return `<section class="screen">${topbar('Análise')}<div class="scroll"><div class="empty">${icon('search', 40)}Análise não encontrada.</div></div></section>`;
}

SCREENS.result = {
  html: (id) => {
    const a = findAnalysis(id);
    if (!a) return missing();
    const sm = summarize(a);
    const many = a.fruits.length > 1;
    const deadline = shipDeadline(a, sm);
    return `<section class="screen">
    ${topbar('Resultado da análise', { backTo: 'home' })}
    <div class="scroll" style="padding-top:4px">
      <div class="hero-fruit"><img src="${a.fruits[0].crop}" alt="Abacate analisado"></div>
      ${many ? `<p class="center muted" style="font-size:.8rem;margin:-6px 0 10px">${a.fruits.length} frutos encontrados nesta foto</p>` : ''}
      <div class="card">
        <div class="mat">${ring(sm.maturity, ringColor(sm.status))}
          <div><h4>Maturação</h4>${destChip(sm.destination)}</div></div>
        <div class="divider"></div>
        <div class="info-row">${icon('calendar', 22)}<div><h5>Previsão de envio</h5>
          <p class="big">${deadline.title}</p><p>${deadline.text}</p></div></div>
        <div class="tipbox info-row">${icon('leaf', 22)}<div><h5>Condições ideais</h5><p>${idealConditions(sm)}</p></div></div>
      </div>
      <button class="link mt" style="width:100%;justify-content:center" data-go="details/${a.id}">Ver detalhes da análise ${icon('chevronRight', 16)}</button>
    </div>
    <div class="actions-bottom"><button class="btn btn-primary" data-go="camera">Nova análise ${icon('refresh', 18)}</button></div>
    </section>`;
  },
};

/* Shipping window: last day the lot can leave for its destination and still arrive green. */
function shipDeadline(a, sm) {
  const s = S.settings;
  const today = startOfDay(Date.now());
  const need = { export: s.exportDays + s.margin, domestic: s.domesticDays + s.margin }[sm.destination];
  if (need !== undefined) {
    const slack = Math.max(0, Math.floor(sm.worst - need));
    const end = today + slack * DAY;
    return {
      title: slack === 0 ? 'Envie hoje' : `Até ${fmtLong(end)}`,
      text: `Janela para ${DEST[sm.destination].name.toLowerCase()}: o lote ainda tem cerca de ${Math.round(sm.worst)} dias de vida verde e a viagem pede ${need}.`,
      start: today, end, need,
    };
  }
  if (sm.destination === 'local') {
    return { title: rangeText(sm.worst), text: 'É o tempo até o fruto atingir o ponto de consumo. Venda perto, nos próximos dias.', start: today, end: today + Math.max(0, Math.floor(sm.worst)) * DAY };
  }
  if (sm.destination === 'processing') {
    return { title: 'Já está no ponto', text: 'A maior parte dos frutos passou do ponto de consumo. Encaminhe hoje para processamento.', start: today, end: today };
  }
  return { title: rangeText(sm.worst), text: 'O sistema ficou em dúvida sobre estes frutos. Confira a olho antes de definir o destino.', start: today, end: today };
}

SCREENS.details = {
  html: (id) => {
    const a = findAnalysis(id);
    if (!a) return missing();
    const sm = summarize(a);
    const f0 = a.fruits[0];
    const ready = rangeText(sm.worst);
    const matText = { green: 'Ainda verde. Tem bastante tempo de prateleira pela frente.',
      ripening: 'Ainda não está no ponto ideal para consumo.', ready: 'No ponto ideal para consumo.',
      passed: 'Passou do ponto ideal de consumo.' }[sm.status];
    return `<section class="screen">
    ${topbar('Detalhes da análise', { left: true, backTo: 'history', right: `<button class="icon-btn" id="del" aria-label="Apagar análise">${icon('trash', 20)}</button>` })}
    <div class="scroll" style="padding-top:8px">
      <div class="detail-head"><img src="${f0.crop}" alt="">
        <div><h2>${esc(lotName(a))}</h2><p>${fmtDate(a.date)}</p>${statusChip(sm.status)}</div></div>
      ${a.fruits.length > 1 ? `<div class="multi" id="fruits">${a.fruits.map((f, i) => `<button data-i="${i}" class="${i === 0 ? 'on' : ''}"><img src="${f.crop}" alt="">Fruto ${i + 1}</button>`).join('')}</div>` : ''}
      <div class="card flat"><div class="mat">${ring(sm.maturity, ringColor(sm.status), 70)}<div><h4>Maturação</h4><p>${matText}</p></div></div></div>
      <div class="card flat"><div class="info-row">${icon('calendar', 22)}<div><h5>Ponto de consumo</h5><p class="big">${ready}</p>
        <p>Guardado em ${condLabel(a.condition).toLowerCase()}. Erro médio do modelo: ${mae().toFixed(1).replace('.', ',')} dia.</p></div></div></div>
      <div class="card flat"><div class="info-row">${icon(DEST[sm.destination].icon, 22)}<div><h5>Destino recomendado</h5><p class="big">${DEST[sm.destination].name}</p>
        <p>Certeza média de ${Math.round(sm.conf * 100)}%.</p></div></div></div>
      ${accordion('tips', 'bulb', 'Dicas', `<p>${idealConditions(sm)}</p><p>Faça uma nova foto em 2 ou 3 dias para acompanhar a evolução do lote.</p>`, true)}
      ${accordion('probs', 'chart', 'Estágio de maturação', `<div id="probs">${probBars(f0)}</div>`)}
      ${f0.cam ? accordion('cam', 'eye', 'Onde a IA olhou', `<div id="camview">${camPair(f0)}</div>`) : ''}
      <div style="display:grid;gap:10px;margin-top:14px">
        <button class="btn btn-ghost" data-go="forecast/${a.id}">${icon('calendar', 18)} Ver previsão de envio</button>
        <button class="btn btn-ghost" id="compare">${icon('trend', 18)} Comparar com outra análise</button>
      </div>
    </div></section>`;
  },
  mount(id) {
    const a = findAnalysis(id);
    if (!a) return;
    bindAccordions();
    $$('#fruits button').forEach((b) => b.addEventListener('click', () => {
      $$('#fruits button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      const f = a.fruits[+b.dataset.i];
      const p = $('#probs'); if (p) p.innerHTML = probBars(f);
      const c = $('#camview'); if (c && f.cam) c.innerHTML = camPair(f);
      $('.detail-head img').src = f.crop;
    }));
    $('#del').addEventListener('click', () => {
      openSheet(`<h3>Apagar esta análise?</h3><p class="sub">${esc(lotName(a))}, ${fmtDate(a.date)}. Não dá para desfazer.</p>
        <div style="display:grid;gap:10px"><button class="btn btn-primary" style="background:#B03A48;box-shadow:none" id="yes">Apagar análise</button>
        <button class="btn btn-ghost" data-close>Cancelar</button></div>`, (sheet) => {
        $('[data-close]', sheet).addEventListener('click', closeSheet);
        $('#yes', sheet).addEventListener('click', () => {
          S.analyses = S.analyses.filter((x) => x.id !== a.id);
          saveAnalyses();
          toast('Análise apagada');
          go('history', true);
        });
      });
    });
    $('#compare').addEventListener('click', () => openCompare(a));
  },
};

function accordion(key, ic, title, body, open = false) {
  return `<div class="card flat"><button class="acc-head" aria-expanded="${open}" data-acc="${key}">${icon(ic, 22)}<h5>${title}</h5>${icon(open ? 'chevronUp' : 'chevronDown', 18)}</button>
    <div class="acc-body" id="acc-${key}" ${open ? '' : 'hidden'}>${body}</div></div>`;
}
function bindAccordions() {
  $$('[data-acc]').forEach((h) => h.addEventListener('click', () => {
    const body = $('#acc-' + h.dataset.acc);
    const open = body.hidden;
    body.hidden = !open;
    h.setAttribute('aria-expanded', open);
    h.lastElementChild.outerHTML = icon(open ? 'chevronUp' : 'chevronDown', 18);
  }));
}
function probBars(f) {
  return f.probs.map((p, i) => `<div class="prob"><span>${i + 1}. ${STAGE_NAMES[i + 1]}</span>
    <span class="bar"><i style="width:${Math.round(p * 100)}%;background:${STAGE_COLORS[i + 1]}"></i></span><b>${Math.round(p * 100)}%</b></div>`).join('');
}
function camPair(f) {
  return `<div class="cam-pair"><div><img src="${f.crop}" alt=""><small>Foto</small></div><div><img src="${f.cam}" alt="Mapa de calor">
    <small>Áreas claras pesaram mais</small></div></div>`;
}

function openCompare(a) {
  const others = S.analyses.filter((x) => x.id !== a.id);
  if (!others.length) { toast('Faça outra análise para poder comparar.'); return; }
  openSheet(`<h3>Comparar com outra análise</h3><p class="sub">Escolha a análise para colocar lado a lado.</p>
    <div class="list">${others.slice(0, 20).map((o) => `<button class="row" data-cmp="${o.id}"><img class="thumb" src="${o.fruits[0].crop}" alt="">
      <div><p class="row-title">${esc(lotName(o))}</p><p class="row-sub">${fmtDate(o.date)}</p></div>${statusChip(summarize(o).status)}</button>`).join('')}</div>`,
  (sheet) => $$('[data-cmp]', sheet).forEach((b) => b.addEventListener('click', () => {
    const o = findAnalysis(b.dataset.cmp);
    const col = (x) => {
      const sm = summarize(x);
      return `<div class="col"><img src="${x.fruits[0].crop}" alt=""><h5>${esc(lotName(x))}</h5><small>${fmtDate(x.date)}</small>
        <div class="kv"><span>Maturação</span><b>${Math.round(sm.maturity)}%</b></div>
        <div class="kv"><span>Estado</span><b>${STATUS[sm.status].label}</b></div>
        <div class="kv"><span>Dias de vida verde</span><b>${sm.worst.toFixed(1).replace('.', ',')}</b></div>
        <div class="kv"><span>Destino</span><b>${DEST[sm.destination].name}</b></div></div>`;
    };
    const diff = summarize(a).maturity - summarize(o).maturity;
    openSheet(`<h3>Comparação</h3><p class="sub">${Math.abs(diff) < 3 ? 'As duas análises estão em maturação parecida.'
      : `A análise de ${fmtDay(a.date)} está ${Math.round(Math.abs(diff))} pontos ${diff > 0 ? 'mais' : 'menos'} madura.`}</p>
      <div class="compare">${col(a)}${col(o)}</div><button class="btn btn-ghost mt" data-close>Fechar</button>`,
    (s2) => $('[data-close]', s2).addEventListener('click', closeSheet));
  })));
}

SCREENS.history = {
  html: (q) => `<section class="screen">
    <div class="scroll" style="padding-top:calc(22px + env(safe-area-inset-top))">
      <h2 class="page-title">Histórico de análises</h2>
      <label class="searchbox">${icon('search', 18)}<input id="q" placeholder="Buscar por lote, talhão ou data..." value="${esc(q)}" aria-label="Buscar"></label>
      <div class="filters" id="filters">${[['all', 'Todos'], ['ready', 'Pronto'], ['ripening', 'Em maturação'], ['green', 'Verde'], ['passed', 'Passou do ponto']]
        .map(([k, l], i) => `<button data-f="${k}" class="${i === 0 ? 'on' : ''}">${l}</button>`).join('')}</div>
      <div id="hist"></div>
    </div>${nav('history')}</section>`,
  mount() {
    let filter = 'all';
    const draw = () => {
      const q = $('#q').value.trim().toLowerCase();
      const items = S.analyses.filter((a) => {
        const sm = summarize(a);
        if (filter !== 'all' && sm.status !== filter) return false;
        if (!q) return true;
        return (lotName(a) + ' ' + fmtDate(a.date)).toLowerCase().includes(q);
      });
      $('#hist').innerHTML = items.length ? `<div class="list">${items.map((a) => analysisRow(a, true)).join('')}</div>`
        : `<div class="list empty">${icon('search', 40)}${S.analyses.length ? 'Nenhuma análise com esse filtro.' : 'Você ainda não fez nenhuma análise.'}</div>`;
    };
    $('#q').addEventListener('input', draw);
    $$('#filters button').forEach((b) => b.addEventListener('click', () => {
      $$('#filters button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on'); filter = b.dataset.f; draw();
    }));
    draw();
  },
};

SCREENS.forecast = {
  html: (id) => {
    const a = findAnalysis(id);
    if (!a) return missing();
    const sm = summarize(a);
    const dl = shipDeadline(a, sm);
    const s = S.settings;
    const today = startOfDay(Date.now());
    const milestones = [
      ['ship', 'Último dia para navio', s.exportDays + s.margin],
      ['truck', 'Último dia para caminhão', s.domesticDays + s.margin],
      ['store', 'Pronto para consumo', 0],
    ].map(([ic, label, need]) => {
      const slack = Math.floor(sm.worst - need);
      return `<li class="${slack < 0 ? 'off' : ''}">${icon(ic, 20)}<span>${label}</span><b>${slack < 0 ? 'não dá mais' : fmtDay(today + slack * DAY)}</b></li>`;
    }).join('');
    return `<section class="screen">
    ${topbar('Previsão de envio ao mercado', { left: true, backTo: 'details/' + a.id })}
    <div class="scroll" style="padding-top:8px">
      <div class="lead-row">${icon('refresh', 24)}<span>Baseado na análise da última foto e no histórico do lote.</span></div>
      <div class="card flat">
        <div style="display:grid;grid-template-columns:44px 1fr auto;gap:12px;align-items:center">
          <img class="thumb" style="width:44px;height:44px" src="${a.fruits[0].crop}" alt="">
          <div><p class="row-title" style="font-size:.98rem">${esc(lotName(a))}</p><p class="row-sub">Última análise: ${fmtDay(a.date)}</p></div>
          <span class="chip chip-green chip-lg" style="font-weight:600">${Math.round(sm.maturity)}%</span></div>
        <div class="window">${icon('calendar', 30)}<div><h5>Janela ideal de envio</h5>
          <p class="big">${dl.start === dl.end ? fmtLong(dl.start) : fmtRange(dl.start, dl.end)}</p>
          <small>Destino recomendado: ${DEST[sm.destination].name.toLowerCase()}</small></div></div>
        <ul class="timeline mt">${milestones}</ul>
      </div>
      <div class="warnbox mt">${icon('alert', 20)}<div>A previsão pode variar conforme o clima e as condições de armazenamento. Ela vale para frutos guardados em ${condLabel(a.condition).toLowerCase()}.</div></div>
      <div class="card flat mt"><p class="chart-title">Evolução da maturação (estimada)</p>${maturityChart(a, sm)}</div>
    </div></section>`;
  },
};

/* Measured maturity of every analysis of the same lot, plus a dashed projection to 100%. */
function maturityChart(a, sm) {
  const same = S.analyses.filter((x) => x.lot === a.lot && x.plot === a.plot).sort((x, y) => x.date - y.date);
  const byDay = new Map();  // one point per day: the latest analysis of that day
  same.forEach((x) => byDay.set(startOfDay(x.date), summarize(x).maturity));
  const pts = [...byDay.entries()].map(([t, m]) => ({ t, m }));
  const lastT = startOfDay(a.date);
  const endT = Math.max(lastT + DAY, startOfDay(sm.readyAt));
  const firstT = Math.min(pts[0].t, lastT) - DAY;
  const W = 320, H = 150, L = 34, R = 10, T = 10, B = 26;
  const x = (t) => L + (t - firstT) / (endT - firstT) * (W - L - R);
  const y = (m) => T + (1 - m / 100) * (H - T - B);
  const grid = [0, 50, 100].map((m) => `<line x1="${L}" x2="${W - R}" y1="${y(m)}" y2="${y(m)}" stroke="#EDF2EE"/>
    <text x="${L - 6}" y="${y(m) + 4}" font-size="10" text-anchor="end" fill="#8A948E">${m}%</text>`).join('');
  const ticks = [];
  const step = Math.max(1, Math.round((endT - firstT) / DAY / 5));
  for (let t = firstT; t <= endT; t += step * DAY) ticks.push(`<text x="${x(t)}" y="${H - 6}" font-size="10" text-anchor="middle" fill="#8A948E">${fmtDay(t)}</text>`);
  const line = pts.map((p) => `${x(p.t)},${y(p.m)}`).join(' ');
  const cur = { t: lastT, m: sm.maturity };
  const proj = `<line x1="${x(cur.t)}" y1="${y(cur.m)}" x2="${x(endT)}" y2="${y(100)}" stroke="#22A45D" stroke-width="2" stroke-dasharray="5 5"/>`;
  const band = `<rect x="${x(lastT)}" y="${T}" width="${Math.max(2, x(Math.min(endT, lastT + Math.max(0, sm.worst - S.settings.margin) * DAY)) - x(lastT))}" height="${H - T - B}" fill="#E8F6EC" opacity=".7"/>`;
  const dots = pts.map((p) => `<circle cx="${x(p.t)}" cy="${y(p.m)}" r="4" fill="#22A45D"/>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Maturação estimada ao longo dos dias">${grid}${band}
    <polyline points="${line}" fill="none" stroke="#22A45D" stroke-width="2"/>${proj}${dots}
    <circle cx="${x(endT)}" cy="${y(100)}" r="4" fill="#fff" stroke="#22A45D" stroke-width="2"/>${ticks.join('')}</svg>
    <p class="row-sub" style="margin-top:6px">Pontos: análises deste lote. Tracejado: projeção até o ponto de consumo.</p>`;
}

const TIPS = [
  { key: 'maturacao', icon: 'avocado', title: 'Como identificar a maturação', sub: 'Aprenda a reconhecer o ponto ideal dos seus abacates.',
    body: `<p>O Hass muda de cor enquanto amadurece: começa verde vivo, passa por tons oliva e termina roxo quase preto. A casca também fica menos brilhante e o fruto cede um pouco ao toque.</p>
    <h3>Os 5 estágios usados pelo app</h3><ul>${Object.entries(STAGE_NAMES).map(([k, v]) => `<li><b>${k}.</b> ${v}</li>`).join('')}</ul>
    <h3>Para a foto sair boa</h3><ul><li>Um fruto inteiro no quadro, sem cortar as pontas.</li><li>Luz clara e uniforme, sem sombra forte.</li><li>Fundo claro e liso, de preferência branco.</li></ul>` },
  { key: 'armazenamento', icon: 'box', title: 'Armazenamento', sub: 'Dicas para conservar a qualidade dos frutos.',
    body: `<p>O abacate Hass não amadurece no pé: o relógio começa a contar na colheita. Quanto mais quente, mais rápido ele amadurece.</p>
    <ul><li>Resfrie logo depois da colheita. Atraso no resfriamento gasta vida útil.</li><li>Para segurar o amadurecimento, use câmara fria e não quebre a cadeia de frio.</li><li>Não guarde junto de frutas que liberam muito etileno, como banana madura.</li></ul>` },
  { key: 'transporte', icon: 'truck', title: 'Transporte', sub: 'Evite perdas e mantenha a qualidade até o mercado.',
    body: `<p>Na exportação marítima, o fruto passa de 20 a 30 dias entre packing house, porto, navio e distribuição. Ele precisa chegar firme.</p>
    <ul><li>Mande para longe só lotes em que até os frutos mais maduros tenham dias de sobra.</li><li>Use a margem de segurança das configurações para atrasos no porto.</li><li>Evite impactos e caixas apertadas: amassado vira porta para fungos.</li></ul>` },
  { key: 'campo', icon: 'sprout', title: 'Cuidados no campo', sub: 'Boas práticas para uma produção mais eficiente.',
    body: `<ul><li>Colha nas horas mais frescas do dia e deixe os frutos à sombra.</li><li>Corte com tesoura, deixando um pedaço do pedúnculo.</li><li>Separe por talhão e data de colheita: lotes homogêneos amadurecem juntos.</li><li>Acompanhe a matéria seca para saber quando colher.</li></ul>` },
];

SCREENS.tips = {
  html: () => `<section class="screen"><div class="scroll" style="padding-top:calc(22px + env(safe-area-inset-top))">
    <h2 class="page-title">Dicas e boas práticas</h2>
    ${TIPS.map((t) => `<button class="tip-card" data-go="tip/${t.key}"><span class="badge">${icon(t.icon, 24)}</span>
      <span><h5>${t.title}</h5><p>${t.sub}</p></span>${icon('chevronRight', 18)}</button>`).join('')}
    <div class="banner">${icon('leaf', 26)}<b>Abacate de qualidade gera mais valor!</b>${logoMark(44, false)}</div>
  </div>${nav('tips')}</section>`,
};
SCREENS.tip = {
  html: (key) => {
    const t = TIPS.find((x) => x.key === key) || TIPS[0];
    return `<section class="screen">${topbar('Dicas', { left: true, backTo: 'tips' })}
      <div class="scroll article"><span class="badge" style="display:inline-grid;width:52px;height:52px;border-radius:50%;background:var(--green-50);color:var(--green-700);place-items:center">${icon(t.icon, 26)}</span>
      <h2>${t.title}</h2>${t.body}</div></section>`;
  },
};

SCREENS.profile = {
  html: () => `<section class="screen"><div class="scroll" style="padding-top:calc(26px + env(safe-area-inset-top))">
    <div class="profile-head"><div class="avatar">${esc(initials(S.profile.name))}</div>
      <div><h2>${esc(S.profile.name)}</h2><p>${esc(S.profile.email || 'Sem e-mail cadastrado')}</p></div></div>
    <div class="list menu">
      ${menuRow('lots', 'folder', 'Meus lotes', 'Gerencie seus lotes e talhões')}
      ${menuRow('settings', 'settings', 'Configurações', 'Ajustes do app')}
      ${menuRow('help', 'help', 'Ajuda', 'Dúvidas e suporte')}
      ${menuRow('about', 'info', 'Sobre o app', 'Versão ' + ((S.info && S.info.version) || '1.0.0'))}
    </div>
    <button class="btn btn-ghost" style="margin-top:28px;color:var(--green-700)" id="logout">Sair</button>
  </div>${nav('profile')}</section>`,
  mount() {
    $('#logout').addEventListener('click', () => {
      openSheet(`<h3>Sair do app?</h3><p class="sub">Seu nome sai deste aparelho. O histórico de análises continua salvo.</p>
        <div style="display:grid;gap:10px"><button class="btn btn-primary" id="yes">Sair</button><button class="btn btn-ghost" data-close>Cancelar</button></div>`,
      (sheet) => {
        $('[data-close]', sheet).addEventListener('click', closeSheet);
        $('#yes', sheet).addEventListener('click', () => { S.profile = null; store.set('profile', null); go('welcome', true); });
      });
    });
  },
};
function menuRow(route, ic, title, sub) {
  return `<button class="row" data-go="${route}">${icon(ic, 22)}<div><p class="row-title">${title}</p><p class="row-sub">${sub}</p></div>${icon('chevronRight', 18)}</button>`;
}

SCREENS.lots = {
  html: () => {
    const groups = {};
    S.analyses.forEach((a) => { const k = lotName(a); (groups[k] = groups[k] || []).push(a); });
    const keys = Object.keys(groups);
    return `<section class="screen">${topbar('Meus lotes', { left: true, backTo: 'profile' })}
      <div class="scroll">${keys.length ? `<div class="list">${keys.map((k) => {
        const last = groups[k][0];
        const sm = summarize(last);
        return `<button class="row" data-lot="${esc(k)}"><img class="thumb" src="${last.fruits[0].crop}" alt="">
          <div><p class="row-title">${esc(k)}</p><p class="row-sub">${plural(groups[k].length, 'análise', 'análises')} • última em ${fmtDay(last.date)}</p></div>
          <div class="row-end"><span class="pct">${Math.round(sm.maturity)}%</span>${statusChip(sm.status)}</div></button>`;
      }).join('')}</div>` : `<div class="list empty">${icon('folder', 40)}Os lotes aparecem aqui depois da primeira análise.</div>`}</div></section>`;
  },
  mount() {
    $$('[data-lot]').forEach((b) => b.addEventListener('click', () => go('history/' + encodeURIComponent(b.dataset.lot))));
  },
};

SCREENS.settings = {
  html: () => {
    const s = S.settings;
    return `<section class="screen">${topbar('Configurações', { left: true, backTo: 'profile' })}
    <div class="scroll">
      <div class="field"><label for="s-name">Nome</label><input id="s-name" value="${esc(S.profile.name)}"></div>
      <div class="field"><label for="s-mail">E-mail</label><input id="s-mail" type="email" value="${esc(S.profile.email || '')}"></div>
      <div class="field"><label>Armazenamento padrão</label><div class="seg" id="s-cond">${conditionButtons(s.condition)}</div></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
        <div class="field"><label for="s-exp">Viagem de navio (dias)</label><input id="s-exp" type="number" min="0" max="60" value="${s.exportDays}"></div>
        <div class="field"><label for="s-dom">Viagem de caminhão (dias)</label><input id="s-dom" type="number" min="0" max="30" value="${s.domesticDays}"></div>
      </div>
      <div class="field"><label for="s-mar">Margem de segurança (dias)</label><input id="s-mar" type="number" min="0" max="10" step="0.5" value="${s.margin}">
        <small>Dias extras exigidos para cobrir atrasos. O lote só vai para longe se até os frutos mais maduros tiverem esse tempo de sobra.</small></div>
      <div class="field"><label>Conexão com o modelo</label>
        <div class="status-line"><span class="dot ${S.online ? 'ok' : S.online === false ? 'bad' : ''}"></span>
        ${S.online ? `Conectado ao servidor (${esc(API ? API.replace(/^https?:\/\//, '') : location.host)})` : 'Sem conexão com o servidor'}</div></div>
      <button class="btn btn-primary" id="s-save">Salvar ajustes</button>
      <button class="btn btn-ghost mt" id="s-clear" style="color:#B03A48">${icon('trash', 18)} Apagar todo o histórico</button>
    </div></section>`;
  },
  mount() {
    bindSeg($('#s-cond'));
    $('#s-save').addEventListener('click', () => {
      const num = (id, d) => { const v = parseFloat($(id).value); return Number.isFinite(v) && v >= 0 ? v : d; };
      S.profile = { name: $('#s-name').value.trim() || S.profile.name, email: $('#s-mail').value.trim() };
      store.set('profile', S.profile);
      Object.assign(S.settings, {
        condition: $('#s-cond .on').dataset.code,
        exportDays: num('#s-exp', 10), domesticDays: num('#s-dom', 4), margin: num('#s-mar', 2),
      });
      store.set('settings', S.settings);
      toast('Ajustes salvos');
      back('profile');
    });
    $('#s-clear').addEventListener('click', () => {
      openSheet(`<h3>Apagar todo o histórico?</h3><p class="sub">${plural(S.analyses.length, 'análise será apagada', 'análises serão apagadas')} deste aparelho.</p>
        <div style="display:grid;gap:10px"><button class="btn btn-primary" style="background:#B03A48;box-shadow:none" id="yes">Apagar histórico</button><button class="btn btn-ghost" data-close>Cancelar</button></div>`,
      (sheet) => {
        $('[data-close]', sheet).addEventListener('click', closeSheet);
        $('#yes', sheet).addEventListener('click', () => { S.analyses = []; saveAnalyses(); closeSheet(); toast('Histórico apagado'); });
      });
    });
  },
};

SCREENS.help = {
  html: () => `<section class="screen">${topbar('Ajuda', { left: true, backTo: 'profile' })}<div class="scroll">
    ${accordion('h1', 'camera', 'Como tirar uma boa foto?', '<p>Um abacate inteiro, bem iluminado, sobre fundo claro. Evite sombra forte e reflexo. Você também pode fotografar vários frutos juntos numa bandeja.</p>', true)}
    ${accordion('h2', 'chart', 'O que significa a porcentagem?', '<p>É o quanto o fruto já andou do verde (0%) até o ponto de consumo (100%). Ela vem das chances que o modelo dá para cada um dos 5 estágios de maturação.</p>')}
    ${accordion('h3', 'thermo', 'Por que o app pergunta onde o fruto está guardado?', '<p>O abacate amadurece bem mais rápido no calor. O mesmo fruto pode ter 15 dias de vida verde na câmara fria e só 6 em temperatura ambiente.</p>')}
    ${accordion('h4', 'ship', 'Como o destino é escolhido?', '<p>O lote vale o que valem os seus frutos mais maduros. O app olha para os 10% mais maduros e compara os dias que eles ainda têm com o tempo de viagem mais a margem de segurança.</p>')}
    ${accordion('h5', 'wifi', 'O app não conecta', '<p>Se o app estiver online (Netlify), o servidor grátis pode estar acordando: espere um minuto e tente de novo. Se estiver rodando no seu computador, confira se o <b>server.py</b> está aberto e se o celular está no mesmo Wi-Fi.</p>')}
    ${accordion('h6', 'alert', 'Limitações', '<p>Os modelos foram treinados com fotos de estúdio, com fundo branco e luz controlada. Fotos muito diferentes disso podem errar mais. Use o resultado como apoio à decisão, não como verdade absoluta.</p>')}
  </div></section>`,
  mount: bindAccordions,
};

SCREENS.about = {
  html: () => {
    const q = (S.info && S.info.quality) || {};
    const pct = (v) => (v == null ? '—' : Math.round(v * 100) + '%');
    return `<section class="screen">${topbar('Sobre o app', { left: true, backTo: 'profile' })}<div class="scroll article">
      <div class="center" style="margin:8px 0 6px">${logoMark(60, false)}</div>
      <h2 class="center">Ripen</h2><p class="center muted" style="margin-top:-6px">Versão ${(S.info && S.info.version) || '1.0.0'}</p>
      <p>Prova de conceito da disciplina SIN 323 (Inteligência Artificial, UFV) para decidir o destino de lotes de abacate Hass a partir de fotos.</p>
      <h3>Como funciona</h3><ul><li>Detector de frutos: ${esc((S.info && S.info.detector) || 'YOLO')}</li>
        <li>Estágio de maturação: ${esc((S.info && S.info.models.stage) || '—')}</li><li>Dias restantes: ${esc((S.info && S.info.models.days) || '—')}</li></ul>
      <h3>Desempenho na validação</h3><ul><li>Estágio exato correto: ${pct(q.stage_accuracy)}</li><li>Erro de no máximo 1 estágio: ${pct(q.stage_within_1)}</li>
        <li>Erro médio nos dias: ${q.days_mae == null ? '—' : q.days_mae.toFixed(2).replace('.', ',') + ' dia'}</li><li>Dias com erro de até 2: ${pct(q.days_within_2)}</li></ul>
      <h3>Dados</h3><p>Treinado com o Hass Avocado Ripening Photographic Dataset, com frutos fotografados todos os dias em três condições de armazenamento.</p>
    </div></section>`;
  },
};

/* ------------------------------------------------------------- boot */
(async function boot() {
  const target = location.hash && location.hash !== '#/splash' ? location.hash : (S.profile ? '#/home' : '#/welcome');
  history.replaceState(null, '', '#/splash');
  render();
  const t0 = Date.now();
  await loadInfo();
  await new Promise((r) => setTimeout(r, Math.max(0, 1300 - (Date.now() - t0))));
  history.replaceState(null, '', target);
  render();
})();
