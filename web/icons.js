/* Line icons in the style of the mockup (24px grid, 2px stroke). */
const ICON_PATHS = {
  home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h5v-6h4v6h5V9.5"/>',
  homeFill: '<path d="M12 2.6 2.4 10.6a1 1 0 0 0 .64 1.77H4.5V21a1 1 0 0 0 1 1H10v-6h4v6h4.5a1 1 0 0 0 1-1v-8.63h1.46a1 1 0 0 0 .64-1.77L12 2.6z" fill="currentColor" stroke="none"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/><path d="M3.5 8.5 2 7"/>',
  history: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>',
  bulb: '<path d="M9 18h6"/><path d="M10 21h4"/><path d="M12 3a6 6 0 0 0-3.6 10.8c.7.6 1.1 1.3 1.1 2.2h5c0-.9.4-1.6 1.1-2.2A6 6 0 0 0 12 3z"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21c1-4 4.5-6 8-6s7 2 8 6"/>',
  camera: '<path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/><circle cx="12" cy="13.5" r="3.5"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>',
  gallery: '<rect x="7" y="3" width="14" height="14" rx="2"/><path d="M3 7v12a2 2 0 0 0 2 2h12"/><circle cx="12" cy="8" r="1.6"/><path d="m21 13-4-4-6 6"/>',
  chevronRight: '<path d="m9 6 6 6-6 6"/>',
  chevronLeft: '<path d="m15 6-6 6 6 6"/>',
  chevronDown: '<path d="m6 9 6 6 6-6"/>',
  chevronUp: '<path d="m6 15 6-6 6 6"/>',
  arrowLeft: '<path d="M19 12H5"/><path d="m11 18-6-6 6-6"/>',
  arrowRight: '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
  x: '<path d="M6 6l12 12"/><path d="M18 6 6 18"/>',
  zap: '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8z"/>',
  flip: '<path d="M20 11a8 8 0 0 0-14.7-4.4L4 8"/><path d="M4 3v5h5"/><path d="M4 13a8 8 0 0 0 14.7 4.4L20 16"/><path d="M20 21v-5h-5"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
  calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18"/><path d="M8 3v4"/><path d="M16 3v4"/><path d="M8 14h2"/><path d="M14 14h2"/><path d="M8 17h2"/>',
  leaf: '<path d="M5 19c0-9 6-14 15-15-1 9-6 15-15 15z"/><path d="M5 19c3-4 6-7 10-9"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  chart: '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M3 20h18"/>',
  trend: '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
  box: '<path d="M3 7l9-4 9 4v10l-9 4-9-4V7z"/><path d="M3 7l9 4 9-4"/><path d="M12 11v10"/>',
  truck: '<path d="M3 6h11v10H3z"/><path d="M14 9h4l3 3v4h-7"/><circle cx="7" cy="18" r="2"/><circle cx="17" cy="18" r="2"/>',
  sprout: '<path d="M12 21v-9"/><path d="M12 12C12 7 8 5 4 5c0 4 3 7 8 7z"/><path d="M12 14c0-4 3-7 8-7 0 4-3 7-8 7z"/>',
  avocado: '<path d="M12 3c-3 0-5 3.5-5.8 7.6C5.3 15 7.6 21 12 21s6.7-6 5.8-10.4C17 6.5 15 3 12 3z"/><circle cx="12" cy="14" r="2.6"/>',
  alert: '<path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
  check: '<path d="m5 12 4 4 10-10"/>',
  checkCircle: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
  trash: '<path d="M4 7h16"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M6 7l1 13h10l1-13"/><path d="M9 7V4h6v3"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14"/><path d="M12 17h.01"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7.5h.01"/>',
  layers: '<path d="M12 3 2 8l10 5 10-5-10-5z"/><path d="m2 13 10 5 10-5"/>',
  folder: '<path d="M3 6a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6z"/><path d="M8 13h8"/>',
  logout: '<path d="M15 4h4v16h-4"/><path d="M10 8l-4 4 4 4"/><path d="M6 12h10"/>',
  thermo: '<path d="M10 14V5a2 2 0 1 1 4 0v9a4 4 0 1 1-4 0z"/><path d="M12 9v6"/>',
  sparkle: '<path d="M12 3v4"/><path d="M12 17v4"/><path d="M3 12h4"/><path d="M17 12h4"/><path d="m6 6 2.5 2.5"/><path d="m15.5 15.5 2.5 2.5"/><path d="m6 18 2.5-2.5"/><path d="m15.5 8.5 2.5-2.5"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  scan: '<path d="M4 8V5a1 1 0 0 1 1-1h3"/><path d="M16 4h3a1 1 0 0 1 1 1v3"/><path d="M20 16v3a1 1 0 0 1-1 1h-3"/><path d="M8 20H5a1 1 0 0 1-1-1v-3"/>',
  ship: '<path d="M3 17c2 1.5 4 1.5 6 0 2 1.5 4 1.5 6 0 2 1.5 4 1.5 6 0"/><path d="M5 14 4 10h16l-1 4"/><path d="M8 10V6h8v4"/><path d="M12 3v3"/>',
  store: '<path d="M4 10v10h16V10"/><path d="M3 10l2-6h14l2 6"/><path d="M3 10a3 3 0 0 0 6 0 3 3 0 0 0 6 0 3 3 0 0 0 6 0"/><path d="M10 20v-5h4v5"/>',
  factory: '<path d="M3 21V10l6 4V10l6 4V6l6 3v12z"/><path d="M7 17h2"/><path d="M13 17h2"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16v4z"/><path d="m14 6 4 4"/>',
  wifi: '<path d="M2 9a15 15 0 0 1 20 0"/><path d="M5 12.5a10 10 0 0 1 14 0"/><path d="M8.5 16a5 5 0 0 1 7 0"/><path d="M12 19.5h.01"/>',
};

function icon(name, size = 22, extra = '') {
  const p = ICON_PATHS[name] || '';
  return `<svg class="ic ${extra}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
}

/* AvocadoCheck logo: outlined avocado with a leaf, as in the mockup. */
function logoMark(size = 64, light = true) {
  const stroke = light ? '#8EE36B' : '#2E8B3E';
  const flesh = light ? 'rgba(142,227,107,.18)' : '#E6F6C7';
  const pit = light ? '#8EE36B' : '#2E8B3E';
  return `<svg width="${size}" height="${size * 1.15}" viewBox="0 0 64 74" aria-hidden="true">
    <path d="M33 13c3-7 9-10 15-10-1.5 6-6 10.5-13 11.5" fill="${stroke}"/>
    <path d="M32 12c-9 0-14.5 10-16.5 22C13.3 47.5 20 64 32 64s18.7-16.5 16.5-30C46.5 22 41 12 32 12z" fill="${flesh}" stroke="${stroke}" stroke-width="4.5" stroke-linejoin="round"/>
    <circle cx="32" cy="43" r="8.5" fill="${pit}"/>
  </svg>`;
}
