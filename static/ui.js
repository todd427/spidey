/* ui.js — bubbles + smart autoscroll + Shift+Enter sends */
(function () {
  const elLog   = document.getElementById("log");
  const elMsg   = document.getElementById("msg");
  const elSend  = document.getElementById("send");
  const elNew   = document.getElementById("reset");
  const elReady = document.getElementById("ready");
  if (!elLog || !elMsg || !elSend || !elReady) {
    console.error("[ui] missing a required element");
    return;
  }

  // ---- status --------------------------------------------------------------
  function setStatus(kind, label) {
    elReady.classList.remove("ready", "sending", "error");
    if (kind === "sending") {
      elReady.classList.add("sending");
      elReady.textContent = label || "sending…";
    } else if (kind === "error") {
      elReady.classList.add("error");
      elReady.textContent = label || "error";
    } else {
      elReady.classList.add("ready");
      elReady.textContent = "ready";
    }
  }

  // ---- smart autoscroll ----------------------------------------------------
  function isNearBottom(el, pad = 120) {
    // tolerate sub-pixel rounding
    const gap = el.scrollHeight - el.clientHeight - el.scrollTop;
    return gap <= pad + 1;
  }
  function scrollToBottom(el, { force = false } = {}) {
    if (!force && !isNearBottom(el)) return;
    // scroll after layout paints
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  }
  elLog.addEventListener("scroll", () => {
    // no-op; just allowing isNearBottom to read fresh values
  });
  // keep pinned when nodes stream in
  const mo = new MutationObserver(() => scrollToBottom(elLog));
  mo.observe(elLog, { childList: true });

  // ---- bubbles -------------------------------------------------------------
  function makeBubble(role /* 'user'|'bot' */, text) {
    const row = document.createElement("div");
    row.className = role === "user" ? "msg me" : "msg bot";

    const bubble = document.createElement("div");
    bubble.className = role === "user" ? "bubble user" : "bubble bot";

    const pre = document.createElement("pre");
    pre.textContent = text;

    bubble.appendChild(pre);
    row.appendChild(bubble);
    return row;
  }
  function appendBubble(role, text, { forceScroll = false } = {}) {
    const node = makeBubble(role, text);
    elLog.appendChild(node);
    scrollToBottom(elLog, { force: forceScroll });
  }
  function clearLog() { elLog.innerHTML = ""; }

  // ---- helpers -------------------------------------------------------------
  async function fetchJSONorText(url, opts = {}) {
    const res = await fetch(url, opts);
    const ct = res.headers.get("content-type") || "";
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    return ct.includes("application/json") ? res.json() : res.text();
  }

  // minimal session id for threading
  const SESSION_KEY = "toddric_session_id";
  const s4 = () => Math.floor((1 + Math.random()) * 0x10000).toString(16).slice(1);
  const newSession = () => `${s4()}-${s4()}-${s4()}-${s4()}-${Date.now().toString(16)}`;
  function getSession(reset=false) {
    if (reset) {
      const id = newSession();
      localStorage.setItem(SESSION_KEY, id);
      return id;
    }
    return localStorage.getItem(SESSION_KEY) || getSession(true);
  }
  let sessionId = getSession();

  // ---- send flow -----------------------------------------------------------
  let sending = false;
  let slowTimer = null;

  async function doSend() {
    if (sending) return;
    const msg = elMsg.value.trim();
    if (!msg) return;

    appendBubble("user", msg, { forceScroll: true });
    elMsg.value = "";
    elMsg.focus();

    sending = true;
    setStatus("sending");
    slowTimer = setTimeout(() => setStatus("sending", "sending… (slow)"), 8000);

    try {
      const res = await fetchJSONorText("/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ message: msg, session_id: sessionId }),
      });
      clearTimeout(slowTimer);

      const text = (typeof res === "object" && typeof res.text === "string")
        ? res.text
        : (typeof res === "string" ? res : "[error] malformed response");

      appendBubble("bot", text);           // gentle autoscroll only if near bottom
      setStatus("ready");
    } catch (e) {
      clearTimeout(slowTimer);
      appendBubble("bot", `[error] ${e.message || e}`);
      setStatus("error");
    } finally {
      sending = false;
      elMsg.focus();
    }
  }

  // ---- events --------------------------------------------------------------
  elSend.addEventListener("click", (e) => { e.preventDefault(); doSend(); });

  // Shift+Enter sends; Enter = newline
  elMsg.addEventListener("keydown", (e) => {
    if (e.isComposing) return;
    const isEnter = (e.key === "Enter" || e.keyCode === 13);
    if (!isEnter) return;
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      e.preventDefault();
      e.stopPropagation();
      doSend();
    }
  });

  const form = elMsg.closest("form");
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      e.stopPropagation();
      doSend();
    });
  }

  elNew.addEventListener("click", (e) => {
    e.preventDefault();
    sessionId = getSession(true);
    clearLog();
    appendBubble("bot", "New chat ready.", { forceScroll: true });
    setStatus("ready");
    elMsg.focus();
  });

  // ---- init ----------------------------------------------------------------
  setStatus("ready");
})();
