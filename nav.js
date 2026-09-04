/*
 * Shared site header/nav, included via `<script src="/nav.js"></script>`
 * as the first thing inside <body> on every page.
 *
 * This is the whole component: no build step, no fetch, no per-page config.
 * It reads its own breadcrumb trail from `location.pathname` and the page's
 * <title> (expects the "Tool Name — description" convention already used
 * across the site), then inserts itself via `insertAdjacentHTML` right where
 * the <script> tag sits in the markup -- so it renders inline during initial
 * parse, in place, with no flash/fetch round-trip.
 *
 * Uses its own hardcoded colors rather than each page's CSS variables (which
 * differ per tool -- amber/teal vs. violet) so this bar reads as shared site
 * chrome, not part of whichever tool it's sitting on top of.
 */
(function () {
  var path = window.location.pathname;
  var onHome = path === '/' || path === '/index.html';
  var onLabIndex = path === '/lab/' || path === '/lab/index.html';

  var crumbs = [{ label: 'Michael Krauklis', href: '/' }];
  if (!onHome) {
    crumbs.push({ label: 'Lab', href: '/lab/' });
    if (!onLabIndex) {
      var title = (document.title || '').split(/[—-]/)[0].trim();
      crumbs.push({ label: title || 'Tool', href: null });
    }
  }

  var crumbHtml = crumbs.map(function (c, i) {
    var sep = i > 0 ? '<span class="site-nav-sep">/</span>' : '';
    var item = c.href
      ? '<a href="' + c.href + '">' + c.label + '</a>'
      : '<span class="site-nav-current">' + c.label + '</span>';
    return sep + item;
  }).join('');

  var html =
    '<style>' +
    '.site-nav{background:#0c0e13;border-bottom:1px solid #2a3040;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;}' +
    '.site-nav-wrap{max-width:780px;margin:0 auto;padding:14px 24px;display:flex;align-items:center;gap:8px;font-size:0.82rem;}' +
    '.site-nav a{color:#9aa0ae;text-decoration:none;}' +
    '.site-nav a:hover{color:#e0a54c;text-decoration:underline;}' +
    '.site-nav-sep{color:#454c5e;}' +
    '.site-nav-current{color:#e9e7de;}' +
    '</style>' +
    '<nav class="site-nav"><div class="site-nav-wrap">' + crumbHtml + '</div></nav>';

  var here = document.currentScript;
  if (here) {
    here.insertAdjacentHTML('beforebegin', html);
  } else {
    document.write(html);
  }
})();
