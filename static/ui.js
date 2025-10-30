// ui.js — minimal chat front-end with Shift+Enter=send, Enter=newline

const $  = (s) => document.querySelector(s);
const log = $("#log");
const form = $("#f");
const box  = $("#msg");
const sendBtn = $("#send");
const readyChip = $("#ready");

let session = crypto.randomUUID();

function setReady(t) { if (readyChip) readyChip.textContent = t; }
function setSending(on) {
  sendBtn.disabled = !!on;
  sendBtn.textContent = on ? "Sending…" : "Send";
}

function scrollToBottom() {
  log.scrollTo({ top: log.scrollHeight, behavior: "smooth" });
}

function addBubble(role, text) {
  const d = document.createElement("div");
  d.className = "msg " + (role === "me" ? "me" : "bot");
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  d.appendChild(b);
  log.appendChild(d);
  scrollToBottom();
}

$("#reset")?.addEventListener("click", () => {
  session = crypto.randomUUID();
  log.innerHTML = "";
  addBubble("bot", "New chat started.");
});

// Invert keys: Shift+Enter = send, Enter = newline
box.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.shiftKey) {
    e.preventDefault();
    form.requestSubmit(); // triggers submit -> send()
  }
  // Plain Enter falls through = newline (no preventDefault)
});

// Submit handler (works for button click and Shift+Enter)
form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = box.value.trim();
  if (!text) return;
  box.value = "";
  addBubble("me", text);

  setSending(true);
  setReady("sending…");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ message: text, session_id: session }),
    });

    if (!res.ok) {
      const err = await res.text();
      addBubble("bot", `[error ${res.status}] ${err}`);
      return;
    }
    const data = await res.json();
    addBubble("bot", data.text || "");
  } catch (e) {
    addBubble("bot", `[client error] ${e?.message || e}`);
    console.error(e);
  } finally {
    setSending(false);
    setReady("ready");
  }
});

// Fill footer system prompt
(async () => {
  try {
    const r = await fetch("/debug/prompt");
    if (r.ok) {
      const j = await r.json();
      const head = (j.system_prompt_head || "").trim();
      $("#system-prompt").textContent = head + (j.len > 240 ? " …" : "");
    } else {
      $("#system-prompt").textContent = "unavailable";
    }
  } catch {
    $("#system-prompt").textContent = "unavailable";
  }
})();

