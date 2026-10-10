/* LASAGNA project page: theme toggle, scroll reveal, active nav, copy buttons, figure lightbox, video. */
(function () {
  "use strict";
  window.siteReady = true; // the inline watchdog in <head> drops the .js class if this file never runs
  var doc = document.documentElement;
  var live = document.getElementById("live");
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  function announce(msg) { if (live) { live.textContent = ""; setTimeout(function () { live.textContent = msg; }, 30); } }

  /* ---- theme ---- */
  var toggle = document.getElementById("theme-toggle");
  var systemDark = window.matchMedia("(prefers-color-scheme: dark)");
  function isDark() {
    var t = doc.getAttribute("data-theme");
    return t ? t === "dark" : systemDark.matches;
  }
  function syncToggle() {
    if (!toggle) return;
    var dark = isDark();
    toggle.setAttribute("aria-pressed", dark ? "true" : "false");
    toggle.title = dark ? "Switch to light theme" : "Switch to dark theme";
  }
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      doc.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable */ }
      syncToggle();
    });
    if (systemDark.addEventListener) systemDark.addEventListener("change", syncToggle);
    syncToggle();
  }

  /* ---- nav border on scroll ---- */
  var nav = document.getElementById("nav");
  function onScroll() { if (nav) nav.classList.toggle("is-scrolled", window.scrollY > 8); }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---- reveal on scroll ---- */
  var reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !reduceMotion.matches) {
    var ro = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("is-visible"); ro.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    reveals.forEach(function (el) { ro.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add("is-visible"); });
  }

  /* ---- active section in nav ---- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".nav-links a[href^='#']"));
  var byId = {};
  links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
  var tracked = Object.keys(byId).map(function (id) { return document.getElementById(id); }).filter(Boolean);
  if ("IntersectionObserver" in window && tracked.length) {
    var current = null;
    var so = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        if (current) current.removeAttribute("aria-current");
        current = byId[e.target.id];
        if (current) {
          current.setAttribute("aria-current", "true");
          var bar = current.parentElement;
          if (bar && bar.scrollWidth > bar.clientWidth) {
            var left = current.offsetLeft - bar.clientWidth / 2 + current.offsetWidth / 2;
            bar.scrollTo({ left: left, behavior: reduceMotion.matches ? "auto" : "smooth" });
          }
        }
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    tracked.forEach(function (s) { so.observe(s); });
    // Above the first tracked section (the hero), no nav item is current.
    window.addEventListener("scroll", function () {
      if (current && window.scrollY < tracked[0].offsetTop - window.innerHeight * 0.45) {
        current.removeAttribute("aria-current");
        current = null;
        var bar = links[0] && links[0].parentElement;
        if (bar) bar.scrollLeft = 0;
      }
    }, { passive: true });
  }

  /* ---- copy buttons ---- */
  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    var target = document.getElementById(btn.getAttribute("data-copy"));
    if (!target) { btn.hidden = true; return; }
    btn.addEventListener("click", function () {
      var text = target.textContent;
      function done() {
        btn.textContent = "Copied";
        announce("Copied to clipboard");
        setTimeout(function () { btn.textContent = "Copy"; }, 1600);
      }
      function selectFallback() {
        var r = document.createRange();
        r.selectNodeContents(target);
        var s = window.getSelection();
        s.removeAllRanges();
        s.addRange(r);
        announce("Text selected; press Command or Control plus C to copy");
      }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, selectFallback);
      } else {
        selectFallback();
      }
    });
  });

  /* ---- figure lightbox ---- */
  var dialog = document.getElementById("lightbox");
  var lbImg = document.getElementById("lightbox-img");
  var lbCap = document.getElementById("lightbox-caption");
  var opener = null;
  if (dialog && typeof dialog.showModal === "function") {
    document.querySelectorAll("a[data-zoom]").forEach(function (a) {
      var img = a.querySelector("img");
      var cap = a.closest("figure") && a.closest("figure").querySelector("figcaption");
      a.setAttribute("aria-label", "Enlarge figure: " + (img ? img.alt : ""));
      a.addEventListener("click", function (ev) {
        if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button === 1) return;
        ev.preventDefault();
        opener = a;
        lbImg.src = img ? (img.currentSrc || img.src) : a.href;
        lbImg.alt = img ? img.alt : "";
        lbImg.width = img ? img.getAttribute("width") : 1600;
        lbImg.height = img ? img.getAttribute("height") : 1000;
        lbCap.textContent = cap ? cap.textContent : "";
        dialog.showModal();
        var close = dialog.querySelector(".lightbox-close");
        if (close) close.focus();
      });
    });
    dialog.addEventListener("click", function (ev) { if (ev.target === dialog) dialog.close(); });
    dialog.addEventListener("close", function () { if (opener) opener.focus(); });
  }

  /* ---- explainer video: muted, plays on click ---- */
  var shell = document.getElementById("video-shell");
  var video = document.getElementById("explainer");
  var play = document.getElementById("play");
  if (shell && video) {
    video.muted = true;
    var fallbacks = ["video/poster.png", "assets/fig-overview.png"];
    var probe = function (src) {
      var im = new Image();
      im.onerror = function () {
        var next = fallbacks.shift();
        if (!next) return;
        video.poster = next;
        if (next.indexOf("fig-overview") !== -1) shell.classList.add("is-fallback");
        probe(next);
      };
      im.src = src;
    };
    probe(video.getAttribute("poster"));

    var source = video.querySelector("source");
    var broken = function () {
      shell.classList.add("is-fallback");
      video.removeAttribute("controls");
      video.poster = "assets/fig-overview.png";
    };
    if (source) source.addEventListener("error", broken);

    // Before the first play, the overlay button is the one control; the native controls
    // (present in the HTML for visitors without JS) come back as soon as playback starts.
    var start = function () {
      if (shell.classList.contains("is-fallback")) return;
      video.muted = true;
      video.setAttribute("controls", "");
      shell.classList.add("is-playing");
      var p = video.play();
      if (p && p.catch) p.catch(function () { /* the native controls remain */ });
    };
    if (play) {
      video.removeAttribute("controls");
      play.addEventListener("click", function () { start(); video.focus(); });
      video.addEventListener("click", function () { if (!shell.classList.contains("is-playing")) start(); });
    }
    video.addEventListener("play", function () {
      shell.classList.add("is-playing");
      if (!video.hasAttribute("controls")) video.setAttribute("controls", "");
    });
  }
  /* ---- terminal demo: autoplays muted on loop, unless reduced motion is set ---- */
  var demo = document.getElementById("demo");
  if (demo && reduceMotion.matches) {
    demo.removeAttribute("autoplay");
    demo.pause();
    demo.controls = true;
  }
})();
