// VENDORED from DocumenterVitepress (template/src/.vitepress/config.mts).
//
// DocumenterVitepress substitutes its own copy only when this file is absent, so
// this copy is frozen at the version it was taken from. There are six local
// changes, all marked below: the favicon links (DocumenterVitepress emits a
// single hardcoded favicon.ico link and offers no hook for a second, so it
// cannot follow the reader's light/dark browser theme), the navbar logo, the
// style rule that switches the light and dark figures (docs/figure_theme.jl),
// the sitemap, the canonical link on every page, and the script that draws the
// stepped rule beside the "On this page" outline.
//
// On a DocumenterVitepress upgrade: diff this against the new template and
// re-apply that one line. Every placeholder token in this file must stay intact
// -- the build substitutes them in place by exact string match.

import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import { mathjaxPlugin } from './mathjax-plugin'
import { juliaReplTransformer } from './julia-repl-transformer'
import footnote from "markdown-it-footnote";
import path from 'path'

const mathjax = mathjaxPlugin()

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  // LOCAL CHANGE: emit sitemap.xml for search engines. Page URLs are resolved
  // against this hostname, so it carries the full stable prefix (with trailing
  // slash). Every build (dev, versions, previews) lists the stable URLs; those
  // other builds are noindex anyway (DocumenterVitepress' noindex_non_stable).
  // Submit https://maltepuetz.github.io/QuditClifford.jl/stable/sitemap.xml in
  // Google Search Console.
  sitemap: {
    hostname: 'https://maltepuetz.github.io/QuditClifford.jl/stable/',
  },
  // LOCAL CHANGE: a self-referencing canonical link on every page. GitHub Pages
  // serves each page both at its clean URL and with `.html` appended (and the
  // home page also as `index.html`), so without it Google reports the variants
  // as "Duplicate without user-selected canonical". The link names the clean
  // URL under this build's own base, so the stable build points at /stable/
  // and the noindex builds (dev, versions, previews) at themselves.
  transformPageData(pageData) {
    const page = pageData.relativePath
      .replace(/(^|\/)index\.md$/, '$1')
      .replace(/\.md$/, '')
    pageData.frontmatter.head ??= []
    pageData.frontmatter.head.push([
      'link',
      { rel: 'canonical', href: `https://maltepuetz.github.io${baseTemp.base}${page}` },
    ])
  },
  head: [
    // Dark-mode favicon, shipped as two .ico files rather than SVG. Safari ranks
    // an .ico above an SVG icon and paints the .ico even when both are offered,
    // so an SVG variant would never be seen there. Two further mechanisms that
    // look like they should work do not either: a `prefers-color-scheme` rule
    // inside the icon is evaluated as light whatever the OS theme, and `media`
    // on the link is not honoured everywhere. So the media-scoped links are the
    // progressive-enhancement base and the script is the fallback -- it removes
    // and re-appends rather than assigning href, because changing href alone
    // does not always make a browser re-read the icon, and it re-runs when the
    // OS theme changes. The DocumenterVitepress favicon placeholder is dropped:
    // it only ever emits one fixed `favicon.ico` link, which Safari would then
    // prefer over these.
    ['link', { rel: 'icon', media: '(prefers-color-scheme: light)', href: `${baseTemp.base}favicon-light.ico` }],
    ['link', { rel: 'icon', media: '(prefers-color-scheme: dark)',  href: `${baseTemp.base}favicon-dark.ico` }],
    ['script', {}, `(function(){var b=${JSON.stringify(baseTemp.base)},q=window.matchMedia('(prefers-color-scheme: dark)');function a(){var o=document.querySelectorAll('link[rel="icon"]');for(var i=0;i<o.length;i++)o[i].parentNode.removeChild(o[i]);var l=document.createElement('link');l.rel='icon';l.href=b+(q.matches?'favicon-dark.ico':'favicon-light.ico');document.head.appendChild(l);}q.addEventListener?q.addEventListener('change',a):q.addListener(a);a();})();`],
    // LOCAL CHANGE: every figure is embedded twice (docs/figure_theme.jl); show the
    // one that matches the VitePress theme toggle, which sets the class `dark` on <html>,
    // centered in the content column.
    ['style', {}, 'img.light-only,img.dark-only{display:block;margin-left:auto;margin-right:auto} html.dark img.light-only{display:none} html:not(.dark) img.dark-only{display:none}'],
    // LOCAL CHANGE: the stepped rule beside the "On this page" outline (styled in
    // theme/overrides.css). VitePress's outline component offers no hook, and
    // replacing it would mean vendoring theme/index.ts as well, so this script
    // finds the outline in the DOM. It draws the rule as an SVG path that runs
    // down the left of the links, 10px further in per heading level, bending
    // between levels. Every section whose heading is on screen (below the
    // navbar) is in view; with no heading on screen, the section the text on
    // screen belongs to is, the one whose heading was passed last. An in-view
    // link gets `data-in-view`, and a second copy of the path in the brand
    // color is clipped to the in-view links, which are always consecutive
    // because headings appear in outline order. A dot rides
    // the path (CSS offset-path) at the end of that stretch you are scrolling
    // towards: the bottom while scrolling down, the top while scrolling up.
    // VitePress's own single `active` link is not used.
    //
    // Scroll and resize events, a MutationObserver (the outline being filled
    // in, the page changing) and a ResizeObserver (links rewrapping) each
    // schedule one redraw per frame. It waits for Vue to mount the app
    // (`__vue_app__` is set once hydration is done), because a node it
    // inserted into the server-rendered markup earlier would be a hydration
    // mismatch. Apart from inserting its three elements once per outline, its
    // own changes are attributes other than `class`, so they do not wake the
    // observer again. The dev server inserts head tags without running them,
    // so the rule shows only in a built site.
    ['script', {}, `(function () {
  var NS = 'http://www.w3.org/2000/svg';
  // STEP is the indent per heading level and PAD the vertical padding of a
  // link (both as in theme/overrides.css); each bend fills the padding of the
  // two links it joins, so the highlight, clipped to the text, starts and
  // ends on a straight stretch.
  var STEP = 10, PAD = 5;
  var queued = false, resizer = null, watched = null;
  var lastY = window.scrollY, down = true, shape = '', length = 0;
  function schedule() {
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }
  function onScroll() {
    var y = window.scrollY;
    if (y !== lastY) { down = y > lastY; lastY = y; }
    schedule();
  }
  function make(cls) {
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('class', cls);
    svg.setAttribute('aria-hidden', 'true');
    svg.appendChild(document.createElementNS(NS, 'path'));
    return svg;
  }
  // Distance along the path to height y; the path only ever goes down.
  function along(path, y) {
    var lo = 0, hi = length;
    for (var k = 0; k < 20; k++) {
      var mid = (lo + hi) / 2;
      if (path.getPointAtLength(mid).y < y) lo = mid; else hi = mid;
    }
    return hi;
  }
  function update() {
    queued = false;
    var app = document.getElementById('app');
    if (!app || !app.__vue_app__) return;
    var content = document.querySelector('.VPDocAsideOutline .content');
    if (!content) return;
    if (resizer && watched !== content) {
      resizer.disconnect();
      resizer.observe(content);
      watched = content;
    }
    var base = content.querySelector(':scope > svg.outline-rule:not(.highlight)');
    var high = content.querySelector(':scope > svg.outline-rule.highlight');
    var dot = content.querySelector(':scope > .outline-dot');
    if (!base) { base = make('outline-rule'); content.appendChild(base); shape = ''; }
    if (!high) { high = make('outline-rule highlight'); content.appendChild(high); }
    if (!dot) {
      dot = document.createElement('div');
      dot.className = 'outline-dot';
      dot.setAttribute('aria-hidden', 'true');
      content.appendChild(dot);
    }
    var links = content.querySelectorAll('.outline-link');
    var n = links.length, heads = [];
    for (var i = 0; i < n; i++) {
      var id = (links[i].getAttribute('href') || '').split('#')[1];
      var head = id ? document.getElementById(decodeURIComponent(id)) : null;
      heads.push(head ? head.getBoundingClientRect() : null);
    }
    var navH = parseFloat(getComputedStyle(document.documentElement)
      .getPropertyValue('--vp-nav-height')) || 0;
    var box = content.getBoundingClientRect();
    var d = '', x0 = 0, b0 = 0, width = 1, first = null, last = null;
    // Headings on screen; failing that, the last one scrolled past.
    var seen = [], any = false, passed = -1;
    for (var i = 0; i < n; i++) {
      var hr = heads[i];
      seen.push(!!hr && hr.bottom > navH && hr.top < window.innerHeight);
      if (seen[i]) any = true;
      if (hr && hr.bottom <= navH) passed = i;
    }
    if (!any && passed >= 0) seen[passed] = true;
    for (var i = 0; i < n; i++) {
      if (links[i].hasAttribute('data-in-view') !== seen[i]) {
        if (seen[i]) links[i].setAttribute('data-in-view', '');
        else links[i].removeAttribute('data-in-view');
      }
    }
    for (var i = 0; i < n; i++) {
      var a = links[i], level = 0;
      for (var p = a.parentElement; p && p !== content; p = p.parentElement)
        if (p.classList.contains('nested')) level++;
      var r = a.getBoundingClientRect();
      var x = 0.5 + STEP * level, t = r.top - box.top, b = r.bottom - box.top;
      if (i === 0) d = 'M' + x + ' ' + t;
      else if (x !== x0) {
        var m = (b0 + t) / 2;
        d += ' L' + x0 + ' ' + (b0 - PAD) + ' C' + x0 + ' ' + m + ' ' + x + ' ' + m + ' ' + x + ' ' + (t + PAD);
      }
      x0 = x; b0 = b; width = Math.max(width, x + 1);
      if (a.hasAttribute('data-in-view')) {
        if (first === null) first = t + PAD;
        last = b - PAD;
      }
    }
    if (n) d += ' L' + x0 + ' ' + b0;
    var h = content.scrollHeight;
    [base, high].forEach(function (svg) {
      svg.setAttribute('width', width);
      svg.setAttribute('height', h);
      if (svg.firstChild.getAttribute('d') !== d) svg.firstChild.setAttribute('d', d);
    });
    if (d !== shape) {
      shape = d;
      length = n ? base.firstChild.getTotalLength() : 0;
      dot.style.offsetPath = n ? "path('" + d + "')" : 'none';
    }
    if (first !== null) {
      high.style.clipPath = 'inset(' + first + 'px 0 ' + Math.max(0, h - last) + 'px 0)';
      high.style.opacity = '1';
      dot.style.offsetDistance = along(base.firstChild, down ? last : first) + 'px';
      dot.style.opacity = '1';
    } else {
      high.style.opacity = '0';
      dot.style.opacity = '0';
    }
  }
  new MutationObserver(schedule).observe(document.documentElement, {
    subtree: true, childList: true, attributes: true, attributeFilter: ['class']
  });
  if (window.ResizeObserver) resizer = new ResizeObserver(schedule);
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', schedule);
  schedule();
})();`],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', {src: `${baseTemp.base}siteinfo.js`}],
    // REPLACE_ME_DOCUMENTER_VITEPRESS_NOINDEX
  ],
  
  markdown: {
    codeTransformers: [juliaReplTransformer()],
    config(md) {
      md.use(tabsMarkdownPlugin);
      md.use(footnote);
      mathjax.markdownConfig(md);
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  vite: {
    plugins: [
      mathjax.vitePlugin,
    ],
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [ 
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ], 
    }, 
    ssr: { 
      noExternal: [ 
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ], 
    },
  },
  themeConfig: {
    outline: 'deep',
    // Navbar logo, set directly instead of through the placeholder.
    // DocumenterVitepress only ever emits a single `logo: { src: '/logo.svg' }`,
    // but themeConfig.logo is a ThemeableImage, so VitePress can pick between the
    // two variants with its own `.dark` class -- no second asset that adapts via
    // prefers-color-scheme, and no CSS repaint. VPImage applies withBase(), so the
    // absolute paths survive a deploy under a base prefix. The build logs a
    // "No logo.png file found" warning because no logo.svg exists any more; it is
    // expected, and the replacement it would have made is a no-op here.
    logo: { light: '/logo-light.svg', dark: '/logo-dark.svg', width: 24, height: 24 },
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    sidebarDrawer: 'REPLACE_ME_DOCUMENTER_VITEPRESS_SIDEBAR_DRAWER',
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
