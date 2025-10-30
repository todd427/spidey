/* ui.js — binds to #log, #msg, #send, #reset, #ready
   Shift+Enter sends; Enter inserts newline
*/

(function () {
  // ---- Elements (match ui.html) --------------------------------------------
  const elChat   = document.getElementById("log");      // <main id="log">
  const elInput  = document.getElementById("msg");      // <textarea id="msg">
  const elSend   = document.getElementById("send");     // <button id="send">
  const elNew    = document.getElementById("reset");    // <button id="reset">
  const elStatus = document.getElementById("ready");    // <span id="ready">
  const elSys    = document.getElementById("system-prompt");

  if (!elChat || !elInput || !elSend) {
    console.error("[ui] missing required elements (#log, #msg, #send)");
    return;
  }

  // ---- Status ---------------------------------------------------------------
  function setStatus(kind, extra) {
    if (!elStatus) return;
    elStatus.classList.remove("ready", "sending", "error");
    if (kind === "sending") {
      elStatus.classList.add("sending");
      elStatus.textContent = extra || "sending…";
    } else if (kind === "error") {
      elStatus.classList.add("error");
      elStatus.textContent = extra || "error";
    } else {
      elStatus.classList.add("ready");
      elStatus.textContent = "ready";
    }
  }

  // ---- Chat bubbles ---------------------------------------------------------
  function bubble(role, text) {
    const wrap = document.createElement("div");
    wrap.className = role === "user" ? "bubble user" : "bubble bot";
    const pre = document.createElement("pre");
    pre.textContent = text;
    wrap.appendChild(pre);
    return wrap;
  }
  function append(role, text) {
    const b = bubble(role, text);
    elChat.appendChild(b);
    b.scrollIntoView({ behavior: "smooth", block: "end" });
  }
  function clearChat() {
    elChat.querySelectorAll(".bubble").forEach((n) => n.remove());
  }

  // ---- Helpers --------------------------------------------------------------
  async function fetchJSONorText(url, opts = {}) {
    const res = await fetch(url, opts);
    const ct = res.headers.get("content-type") || "";
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    if (ct.includes("application/json")) return res.json();
    return res.text();
  }

  async function loadSystemPrompt() {
    if (!elSys) return;
    try {
      const raw = await fetchJSONorText("/debug/prompt");
      let head;
      if (typeof raw === "string") head = raw.trim();
      else if (raw && typeof raw === "object")
        head = raw.system_prompt_head || raw.system || raw.text || JSON.stringify(raw);
      else head = String(raw);
      elSys.textContent = head;
    } catch {
      elSys.textContent = "(unavailable)";
    }
  }

  // ---- Send flow ------------------------------------------------------------
  const SESSION_KEY = "toddric_session_id";
  const s4 = () => Math.floor((1 + Math.random()) * 0x10000).toString(16).slice(1);
  const newSessionId = () => `${s4()}-${s4()}-${s4()}-${s4()}-${Date.now().toString(16)}`;
  function getSession(reset = false) {
    if (reset) {
      const id = newSessionId();
      localStorage.setItem(SESSION_KEY, id);
      return id;
    }
    return localStorage.getItem(SESSION_KEY) || getSession(true);
  }
  let sessionId = getSession();

  let sendLock = false;
  let slowTimer = null;

  async function doSend() {
    if (sendLock) return;
    const msg = elInput.value.trim();
    if (!msg) return;
    elInput.value = "";
    elInput.blur();   // optional: prevents stray newline on Shift+Enter
    elInput.focus();

    sendLock = true;
    setStatus("sending");
    slowTimer = setTimeout(() => setStatus("sending", "sending… (slow)"), 8000);

    append("user", msg);

    try {
      const res = await fetchJSONorText("/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ message: msg, session_id: sessionId }),
      });
      clearTimeout(slowTimer);
      const text = (typeof res === "object" && typeof res.text === "string") ? res.text
                 : (typeof res === "string" ? res : "[error] malformed response");
      append("bot", text);
      setStatus("ready");
    } catch (e) {
      clearTimeout(slowTimer);
      append("bot", `[error] ${e.message || e}`);
      setStatus("error");
    } finally {
      elInput.value = "";
      elInput.focus();
      sendLock = false;
    }
  }

  // ---- Events ---------------------------------------------------------------
  elSend.addEventListener("click", (e) => { e.preventDefault(); doSend(); });

  // Shift+Enter sends; Enter = newline; robust across browsers/IME
  elInput.addEventListener("keydown", (e) => {
    if (e.isComposing) return;                 // ignore IME composition
    if (e.key !== "Enter" && e.keyCode !== 13) return;

    if (e.shiftKey || e.ctrlKey || e.metaKey) { // Shift (primary), Ctrl/Cmd (backup)
      e.preventDefault();
      e.stopPropagation();
      doSend();
      return;
    }
    // plain Enter => newline (let it happen)
  });

  // Prevent form submit from stealing Enter
  const form = elInput.closest("form");
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      e.stopPropagation();
      doSend();
    });
  }

  if (elNew) {
    elNew.addEventListener("click", (e) => {
      e.preventDefault();
      sessionId = getSession(true);
      clearChat();
      setStatus("ready");
      append("bot", "New chat ready.");
      elInput.focus();
    });
  }

  // ---- Init -----------------------------------------------------------------
  setStatus("ready");
  loadSystemPrompt();
})();
