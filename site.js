/* LASAGNA project page: theme toggle, scroll reveal, active nav, copy buttons, figure lightbox, video, podcast. */
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

  /* ---- explainer video: autoplays muted on loop, unless reduced motion is set ---- */
  var shell = document.getElementById("video-shell");
  var video = document.getElementById("explainer");
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
    if (source) source.addEventListener("error", function () {
      shell.classList.add("is-fallback");
      video.removeAttribute("controls");
      video.poster = "assets/fig-overview.png";
    });

    if (reduceMotion.matches) {
      video.removeAttribute("autoplay");
      video.pause();
    } else {
      // A browser that refuses autoplay (data saver, low power mode) leaves the native controls.
      var p = video.play();
      if (p && p.catch) p.catch(function () {});
    }
  }
  /* ---- terminal demo: autoplays muted on loop, unless reduced motion is set ---- */
  var demo = document.getElementById("demo");
  if (demo && reduceMotion.matches) {
    demo.removeAttribute("autoplay");
    demo.pause();
    demo.controls = true;
  }

  /* ---- podcast: chapters seek the episode, the current chapter and transcript turn are marked,
     and a mini player stays on screen while it plays off screen ---- */
  var audio = document.getElementById("podcast-audio");
  var seekers = document.querySelectorAll("[data-seek]");
  if (audio && audio.canPlayType && audio.canPlayType("audio/mp4")) {
    var mmss = function (t) {
      var m = Math.floor(t / 60), s = Math.floor(t % 60);
      return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
    };
    var chapterBtns = Array.prototype.slice.call(document.querySelectorAll(".chapters [data-chapter]"));
    var chapters = chapterBtns.map(function (b) {
      return { t: +b.getAttribute("data-seek"), title: b.querySelector(".ch-title").textContent };
    });
    var turns = Array.prototype.slice.call(document.querySelectorAll(".tr-turn[data-t]"));
    var turnStarts = turns.map(function (p) { return +p.getAttribute("data-t"); });
    var transcript = document.querySelector("#podcast details.transcript");
    var chapterList = document.querySelector("#podcast .chapters");
    var dock = document.getElementById("podcast-dock");
    var dockPlay = document.getElementById("dock-play");
    var dockChapter = document.getElementById("dock-chapter");
    var dockTime = document.getElementById("dock-time");
    var dockDismissed = false, audioOnScreen = true;

    var lastBefore = function (starts, t) {
      var k = -1;
      for (var i = 0; i < starts.length && starts[i] <= t + 0.05; i++) k = i;
      return k;
    };
    Array.prototype.forEach.call(seekers, function (b) {
      b.addEventListener("click", function () {
        var t = +b.getAttribute("data-seek");
        audio.currentTime = t;
        var p = audio.play();
        if (p && p.catch) p.catch(function () {});
        var c = chapters[lastBefore(chapters.map(function (x) { return x.t; }), t)];
        announce("Playing from " + mmss(t) + (c ? ", " + c.title : ""));
      });
    });

    var shownChapter = -2, shownTurn = -2;
    var sync = function () {
      var t = audio.currentTime;
      var ci = (t > 0 || !audio.paused) ? lastBefore(chapters.map(function (x) { return x.t; }), t) : -1;
      if (ci !== shownChapter) {
        shownChapter = ci;
        chapterBtns.forEach(function (b, i) {
          if (i === ci) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current");
        });
        if (dockChapter) dockChapter.textContent = ci >= 0 ? chapters[ci].title : "";
      }
      if (dockTime) dockTime.textContent = mmss(t);
      var ti = (transcript && transcript.open && t > 0) ? lastBefore(turnStarts, t) : -1;
      if (ti !== shownTurn) {
        if (shownTurn >= 0) turns[shownTurn].classList.remove("is-current");
        if (ti >= 0) turns[ti].classList.add("is-current");
        shownTurn = ti;
      }
    };
    audio.addEventListener("timeupdate", sync);
    audio.addEventListener("seeked", sync);
    if (transcript) transcript.addEventListener("toggle", sync);

    var syncDock = function () {
      var playing = !audio.paused && !audio.ended;
      if (chapterList) chapterList.classList.toggle("is-paused", !playing);
      if (!dock) return;
      dock.classList.toggle("is-paused", !playing);
      dockPlay.setAttribute("aria-label", playing ? "Pause the podcast" : "Play the podcast");
      var started = audio.currentTime > 0 && !audio.ended;
      dock.hidden = !(started && !audioOnScreen && !dockDismissed);
    };
    if (dock) {
      dockPlay.addEventListener("click", function () {
        if (audio.paused) { var p = audio.play(); if (p && p.catch) p.catch(function () {}); }
        else audio.pause();
      });
      document.getElementById("dock-close").addEventListener("click", function () {
        dockDismissed = true; syncDock();
      });
      if ("IntersectionObserver" in window) {
        new IntersectionObserver(function (entries) {
          audioOnScreen = entries[0].isIntersecting; syncDock();
        }).observe(audio);
      }
    }
    audio.addEventListener("play", function () { dockDismissed = false; syncDock(); });
    ["pause", "ended", "seeked"].forEach(function (e) { audio.addEventListener(e, syncDock); });

    if ("mediaSession" in navigator && window.MediaMetadata) {
      audio.addEventListener("play", function () {
        navigator.mediaSession.metadata = new MediaMetadata({
          title: "Why Stacked RAG Components Fail",
          artist: "LASAGNA podcast",
          album: "Retrieval Augmented Generation Compositionality",
          artwork: [{ src: "apple-touch-icon.png", sizes: "180x180", type: "image/png" }]
        });
      }, { once: true });
    }
  } else {
    // No playable audio: the chapter and transcript times stay as plain labels.
    Array.prototype.forEach.call(seekers, function (b) { b.disabled = true; });
  }
})();
