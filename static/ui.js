/* ui.js — renders bubbles + keybindings + smart autoscroll
   Shift+Enter sends; Enter inserts newline
*/
(function () {
  // ---- Elements -------------------------------------------------------------
  const elLog   = document.getElementById("log");   // <main id="log">
  const elMsg   = document.getElementById("msg");   // <textarea id="msg">
  const elSend  = document.getElementById("send");  // <button id="send">
  const elNew   = document.getElementById("reset"); // <button id="reset">
  const elReady = document.getElementById("ready"); // <span id="ready">
  if (!elLog || !elMsg || !elSend || !elReady) {
    console.error("[ui] missing a required element");
    return;
  }

  // ---- Status ---------------------------------------------------------------
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

  // ---- Smart autoscroll -----------------------------------------------------
  let userPinnedTop = false; // true if user scrolled up away from bottom

  function isNearBottom(el, pad = 120) {
    return el.scrollHeight - el.clientHeight - el.scrollTop <= pad;
  }

  function scrollToBottom(el, { force = false } = {}) {
    if (force || isNearBottom(el)) {
      el.scrollTop = el.scrollHeight;
    }
  }

  // track whether user is near the bottom
  elLog.addEventListener("scroll", () => {
    userPinnedTop = !isNearBottom(elLog);
  });

  // optional: keep pinned when streaming/adding nodes rapidly
  const mo = new MutationObserver(() => scrollToBottom(elLog));
  mo.observe(elLog, { childList: true, subtree: false });

  // ---- Bubbles --------------------------------------------------------------
  function makeBubble(role /* 'user'|'bot' */, text) {
    // wrapper controls alignment (right for user, left for bot)
    const msg = document.createElement("div");
    msg.className = role === "user" ? "msg me" : "msg bot";

    const bubble = document.createElement("div");
    bubble.className = role === "user" ? "bubble user" : "bubble bot";

    const pre = document.createElement("pre");
    pre.textContent = text;

    bubble.appendChild(pre);
    msg.appendChild(bubble);
    return msg;
  }

  function appendBubble(role, text) {
    const node = makeBubble(role, text);
    elLog.appendChild(node);
    scrollToBottom(elLog); // only scroll if user is already near bottom
  }

  function clearLog() {
    elLog.innerHTML = "";
  }

  // ---- Helpers --------------------------------------------------------------
  async function fetchJSONorText(url, opts = {}) {
    const res = await fetch(url, opts);
    const ct = res.headers.get("content-type") || "";
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    return ct.includes("application/json") ? res.json() : res.text();
  }

  // Session id so /chat can thread
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

  // ---- Send flow ------------------------------------------------------------
  let sending = false;
  let slowTimer = null;

  async function doSend() {
    if (sending) return;
    const msg = elMsg.value.trim();
    if (!msg) return;

    // render user bubble right, clear input, force snap-to-bottom (user just acted)
    appendBubble("user", msg);
    scrollToBottom(elLog, { force: true });
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
      appendBubble("bot", text);
      scrollToBottom(elLog); // gentle: only if near bottom
      setStatus("ready");
    } catch (e) {
      clearTimeout(slowTimer);
      appendBubble("bot", `[error] ${e.message || e}`);
      scrollToBottom(elLog);
      setStatus("error");
    } finally {
      sending = false;
      elMsg.focus();
    }
  }

  // ---- Events ---------------------------------------------------------------
  elSend.addEventListener("click", (e) => { e.preventDefault(); doSend(); });

  // Shift+Enter sends; Enter = newline
  elMsg.addEventListener("keydown", (e) => {
    if (e.isComposing) return;
    if (e.key !== "Enter" && e.keyCode !== 13) return;
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
    appendBubble("bot", "New chat ready.");
    scrollToBottom(elLog, { force: true });
    setStatus("ready");
    elMsg.focus();
  });

  // ---- Init -----------------------------------------------------------------
  setStatus("ready");
})();

