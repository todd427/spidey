/* ui.js — minimal vanilla chat UI for toddric
   - Enter inserts newline, Shift+Enter sends
   - Bottom input, bubbles, status chip, slow-response hint
   - Reads system prompt from /debug/prompt
*/

(function () {
  // ---- Grab elements (be tolerant if template changes) ---------------------
  const $ = (id) => document.getElementById(id);

  const elChat = $("chat") || document.querySelector("[data-chat]") || document.body;
  const elInput = $("input") || $("msg") || document.querySelector("textarea");
  const elSend = $("send") || document.querySelector("[data-send]");
  const elNew = $("newchat") || document.querySelector("[data-newchat]");
  const elStatus = $("status") || document.querySelector("[data-status]");
  const elSysPrompt = $("system-prompt") || document.querySelector("[data-system-prompt]");

  // Footer tip (optional)
  const elTip = $("tip") || document.querySelector("[data-tip]");

  // Apply default tips
  if (elTip) elTip.textContent = "Tip: Shift+Enter sends • Enter adds a newline.";

  // ---- Session handling -----------------------------------------------------
  const SESSION_KEY = "toddric_session_id";
  function newSessionId() {
    // uuid-lite
    const s4 = () => Math.floor((1 + Math.random()) * 0x10000).toString(16).slice(1);
    return `${s4()}-${s4()}-${s4()}-${s4()}-${Date.now().toString(16)}`;
  }
  function getSessionId(reset = false) {
    if (reset) {
      const sid = newSessionId();
      localStorage.setItem(SESSION_KEY, sid);
      return sid;
    }
    let sid = localStorage.getItem(SESSION_KEY);
    if (!sid) {
      sid = newSessionId();
      localStorage.setItem(SESSION_KEY, sid);
    }
    return sid;
  }

  // Initialize
  let sessionId = getSessionId(false);

  // ---- UI helpers ----------------------------------------------------------
  function setStatus(kind, extra) {
    // kind: 'ready' | 'sending' | 'error'
    if (!elStatus) return;
    const flag = (k) => {
      elStatus.classList.remove("ready", "sending", "error");
      elStatus.classList.add(k);
    };
    switch (kind) {
      case "sending":
        flag("sending");
        elStatus.textContent = extra || "sending…";
        break;
      case "error":
        flag("error");
        elStatus.textContent = extra || "error";
        break;
      default:
        flag("ready");
        elStatus.textContent = "ready";
    }
  }

  function createBubble(role, text) {
    const wrap = document.createElement("div");
    wrap.className = role === "user" ? "bubble user" : "bubble bot";

    // Allow basic markdown-ish line breaks; keep it simple/safe.
    const pre = document.createElement("pre");
    pre.textContent = text;
    wrap.appendChild(pre);
    return wrap;
  }

  function appendBubble(role, text) {
    const bubble = createBubble(role, text);
    elChat.appendChild(bubble);
    // Auto-scroll: if near bottom, stick to bottom after append
    try {
      const scroller = document.scrollingElement || document.documentElement;
      const nearBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 160;
      if (nearBottom) {
        bubble.scrollIntoView({ behavior: "smooth", block: "end" });
      }
    } catch (_) {}
    return bubble;
  }

  function clearChat() {
    // Remove only message bubbles, not the fixed UI
    const bubbles = elChat.querySelectorAll(".bubble");
    bubbles.forEach((b) => b.remove());
  }

  // ---- Network -------------------------------------------------------------
  async function fetchJSON(url, opts = {}) {
    const res = await fetch(url, opts);
    const ct = res.headers.get("content-type") || "";
    if (!res.ok) {
      let detail = await res.text().catch(() => "");
      // Try JSON if lied about content-type
      if (!detail && ct.includes("application/json")) {
        try {
          const j = await res.json();
          detail = JSON.stringify(j);
        } catch {}
      }
      const err = new Error(`HTTP ${res.status} ${res.statusText} — ${detail || "no body"}`);
      err.status = res.status;
      err.body = detail;
      throw err;
    }
    if (ct.includes("application/json")) return res.json();
    return res.text();
  }

  async function loadSystemPrompt() {
    if (!elSysPrompt) return;
    try {
      const txt = await fetchJSON("/debug/prompt");
      // The endpoint often returns text/plain; just drop it in.
      elSysPrompt.textContent = `system prompt: ${String(txt).trim()}`;
    } catch (e) {
      elSysPrompt.textContent = "system prompt: (unavailable)";
    }
  }

  // ---- Send flow -----------------------------------------------------------
  let sendLock = false;
  let slowTimer = null;

  async function doSend() {
    if (sendLock) return;
    const msg = (elInput && elInput.value) ? elInput.value.trim() : "";
    if (!msg) return;

    sendLock = true;
    setStatus("sending");
    const slowHint = () => setStatus("sending", "sending… (slow)");
    slowTimer = setTimeout(slowHint, 8000);

    // UI: show user bubble immediately
    appendBubble("user", msg);

    // Prepare payload
    const payload = {
      message: msg,
      session_id: sessionId
    };

    // POST /chat
    let resp;
    try {
      resp = await fetchJSON("/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload)
      });
    } catch (err) {
      // Network or HTTP error
      const code = err && err.status ? err.status : "error";
      appendBubble("bot", `[error ${code}] ${err.message || err}`);
      setStatus("error", `error ${code}`);
      clearTimeout(slowTimer);
      sendLock = false;
      return;
    }

    clearTimeout(slowTimer);

    // Response contract: { text, latency_ms, ... }
    if (!resp || typeof resp.text !== "string") {
      appendBubble("bot", "[error] malformed response");
      setStatus("error");
    } else {
      appendBubble("bot", resp.text);
      setStatus("ready");
    }

    // Clear input (but leave focus)
    if (elInput) {
      elInput.value = "";
      elInput.focus();
    }
    sendLock = false;
  }

  // ---- Events --------------------------------------------------------------
  if (elSend) {
    elSend.addEventListener("click", (e) => {
      e.preventDefault();
      doSend();
    });
  }

  if (elInput) {
    // Enter = newline, Shift+Enter = send
    elInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        if (e.shiftKey) {
          e.preventDefault();
          doSend();
        } else {
          // default: newline (let it through)
        }
      }
    });
  }

  if (elNew) {
    elNew.addEventListener("click", (e) => {
      e.preventDefault();
      sessionId = getSessionId(true);
      clearChat();
      setStatus("ready");
      // Optional: seed a friendly first bubble
      appendBubble("user", "New chat.");
      appendBubble("bot", "Hello! What shall we do next?");
      if (elInput) elInput.focus();
    });
  }

  // ---- Initial paint -------------------------------------------------------
  setStatus("ready");
  loadSystemPrompt();

  // Cosmetic seed if chat area empty
  if (elChat && !elChat.querySelector(".bubble")) {
    appendBubble("user", "Hello, what are you up to today?");
  }

})();

