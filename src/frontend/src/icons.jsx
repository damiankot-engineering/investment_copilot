/* Minimal icon set — stroke-only, lucide-style.
   Globals: window.Icon (renders by name) and named components on window. */

const I = (paths, opts = {}) => ({ paths, ...opts });

const PATHS = {
  // sidebar / nav
  wallet:        I([['rect', { x: 3, y: 6, width: 18, height: 14, rx: 2 }], ['path', { d: 'M3 10h18' }], ['path', { d: 'M16 14h2' }]]),
  barChart:      I([['path', { d: 'M3 3v18h18' }], ['rect', { x: 7,  y: 12, width: 3, height: 6 }], ['rect', { x: 12, y: 8,  width: 3, height: 10 }], ['rect', { x: 17, y: 5,  width: 3, height: 13 }]]),
  sparkles:      I([['path', { d: 'M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6L12 3z' }], ['path', { d: 'M19 14l.8 2.2L22 17l-2.2.8L19 20l-.8-2.2L16 17l2.2-.8L19 14z' }]]),
  fileText:      I([['path', { d: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z' }], ['path', { d: 'M14 2v6h6' }], ['path', { d: 'M8 13h8M8 17h6' }]]),
  activity:      I([['path', { d: 'M22 12h-4l-3 9L9 3l-3 9H2' }]]),

  // actions
  refresh:       I([['path', { d: 'M3 12a9 9 0 0 1 15-6.7L21 8' }], ['path', { d: 'M21 3v5h-5' }], ['path', { d: 'M21 12a9 9 0 0 1-15 6.7L3 16' }], ['path', { d: 'M3 21v-5h5' }]]),
  edit:          I([['path', { d: 'M12 20h9' }], ['path', { d: 'M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z' }]]),
  plus:          I([['path', { d: 'M12 5v14M5 12h14' }]]),
  minus:         I([['path', { d: 'M5 12h14' }]]),
  trash:         I([['path', { d: 'M3 6h18' }], ['path', { d: 'M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2' }], ['path', { d: 'M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6' }]]),
  close:         I([['path', { d: 'M18 6 6 18M6 6l12 12' }]]),
  download:      I([['path', { d: 'M12 3v12' }], ['path', { d: 'M7 10l5 5 5-5' }], ['path', { d: 'M5 21h14' }]]),
  eye:           I([['path', { d: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z' }], ['circle', { cx: 12, cy: 12, r: 3 }]]),
  play:          I([['path', { d: 'M6 4l14 8-14 8V4z' }]]),
  check:         I([['path', { d: 'M20 6 9 17l-5-5' }]]),
  chevronDown:   I([['path', { d: 'm6 9 6 6 6-6' }]]),
  chevronRight:  I([['path', { d: 'm9 6 6 6-6 6' }]]),
  search:        I([['circle', { cx: 11, cy: 11, r: 7 }], ['path', { d: 'm20 20-3.5-3.5' }]]),
  settings:      I([['circle', { cx: 12, cy: 12, r: 3 }], ['path', { d: 'M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 0 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h.1a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5h.1a1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v.1a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z' }]]),

  // status
  trendingUp:    I([['path', { d: 'M3 17l6-6 4 4 8-8' }], ['path', { d: 'M14 7h7v7' }]]),
  trendingDown:  I([['path', { d: 'M3 7l6 6 4-4 8 8' }], ['path', { d: 'M14 17h7v-7' }]]),
  alertTriangle: I([['path', { d: 'M10.3 2.9 1.8 17a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 2.9a2 2 0 0 0-3.4 0z' }], ['path', { d: 'M12 9v4M12 17h.01' }]]),
  alertCircle:   I([['circle', { cx: 12, cy: 12, r: 10 }], ['path', { d: 'M12 8v4M12 16h.01' }]]),
  info:          I([['circle', { cx: 12, cy: 12, r: 10 }], ['path', { d: 'M12 16v-4M12 8h.01' }]]),
  shield:        I([['path', { d: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z' }]]),
  flame:         I([['path', { d: 'M8.5 14.5A2.5 2.5 0 0 0 11 17a4 4 0 0 0 4-4c0-2-1-3.5-2-5-2-3-3-4-3-6 0 2-2 3-3 5s-1 5 1.5 7.5z' }]]),
  zap:           I([['path', { d: 'M13 2 3 14h7l-1 8 10-12h-7l1-8z' }]]),
  circle:        I([['circle', { cx: 12, cy: 12, r: 9 }]]),
  dot:           I([['circle', { cx: 12, cy: 12, r: 3 }]]),
  clock:         I([['circle', { cx: 12, cy: 12, r: 10 }], ['path', { d: 'M12 6v6l4 2' }]]),
  calendar:      I([['rect', { x: 3, y: 4, width: 18, height: 18, rx: 2 }], ['path', { d: 'M16 2v4M8 2v4M3 10h18' }]]),
  filter:        I([['path', { d: 'M22 3H2l8 9.5V20l4-2v-5.5L22 3z' }]]),
  arrowUp:       I([['path', { d: 'M12 19V5M5 12l7-7 7 7' }]]),
  arrowRight:    I([['path', { d: 'M5 12h14M13 5l7 7-7 7' }]]),
  arrowDown:     I([['path', { d: 'M12 5v14M19 12l-7 7-7-7' }]]),
  copilot:       I([
    ['path', { d: 'M12 2l2.4 5.6L20 10l-5.6 2.4L12 18l-2.4-5.6L4 10l5.6-2.4L12 2z' }],
  ]),
  globe:         I([['circle', { cx: 12, cy: 12, r: 10 }], ['path', { d: 'M2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20' }]]),
  bell:          I([['path', { d: 'M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9' }], ['path', { d: 'M10 21a2 2 0 0 0 4 0' }]]),
  layers:        I([['path', { d: 'M12 2 2 7l10 5 10-5-10-5z' }], ['path', { d: 'M2 17l10 5 10-5' }], ['path', { d: 'M2 12l10 5 10-5' }]]),
  target:        I([['circle', { cx: 12, cy: 12, r: 10 }], ['circle', { cx: 12, cy: 12, r: 6 }], ['circle', { cx: 12, cy: 12, r: 2 }]]),
  fileBarChart:  I([['path', { d: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z' }], ['path', { d: 'M14 2v6h6' }], ['path', { d: 'M8 18v-3M12 18v-5M16 18v-7' }]]),
  pulse:         I([['path', { d: 'M3 12h4l2-6 4 12 2-6h6' }]]),
  external:      I([['path', { d: 'M14 4h6v6' }], ['path', { d: 'M20 4 10 14' }], ['path', { d: 'M20 14v6H4V4h6' }]]),
};

function Icon({ name, size = 16, className = '', strokeWidth = 1.75, style }) {
  const def = PATHS[name];
  if (!def) return null;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      style={style}
      aria-hidden="true"
    >
      {def.paths.map(([tag, attrs], i) => React.createElement(tag, { key: i, ...attrs }))}
    </svg>
  );
}

window.Icon = Icon;
