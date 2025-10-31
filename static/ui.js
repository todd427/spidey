// ui.js — robust autoscroll, Shift+Enter send, disabled Send while busy,
// louder status labels, and clean "New chat" reset.

// Small helpers
const $ = (q, el=document) => el.querySelector(q);
const logEl    = $("#log");
const statusEl = $(".status");
const sendBtn  = $("#send");
const inputEl  = $("#input");
const newBtn   = $("#newChat");
const tipWrap  = $("#tipWrap");

// Autoscroll when near bottom (so you can scroll up without being yanked down)
const nearBottom = () => {
  const pad = 140;
  return logEl.scrollHeight - (logEl.scrollTop + logEl.clientHeight) < pad;
};
const scrollToBottom = () => {
  requestAnimationFrame(() => { logEl.scrollTop = logEl.scrollHeight; });
};

// Render helpers
const addMsg = (role, text) => {
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  row.appendChild(b);
  logEl.appendChild(row);
  if (nearBottom()) scrollToBottom();
};

// UI state
const setStatus = (kind, label) => {
  statusEl.classList.remove("ready","sending","slow");
  statusEl.classList.add(kind);
  statusEl.textContent = label;
};
let slowTm = null;
const setBusy = (busy) => {
  sendBtn.disabled = busy;
  inputEl.readOnly = busy;
  clearTimeout(slowTm);
  if (busy) {
    setStatus("sending", "sending…");
    slowTm = setTimeout(() => setStatus("sending", "sending… (slow)"), 3000);
  } else {
    setStatus("ready", "ready");
  }
};

// API
async function postJSON(url, data, timeoutMs=90000){
  const ctrl = new AbortController();
  const tm = setTimeout(()=>ctrl.abort(), timeoutMs);
  try{
    const res = await fetch(url, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify(data),
      signal: ctrl.signal
    });
    if(!res.ok){
      const errText = await res.text().catch(()=>res.statusText);
      throw new Error(`${res.status} ${errText}`);
    }
    return await res.json();
  } finally {
    clearTimeout(tm);
  }
}

// Handlers
async function sendMessage(){
  const msg = inputEl.value.trim();
  if(!msg) return;
  const shouldStick = nearBottom();

  // Render user bubble and clear composer
  addMsg("user", msg);
  inputEl.value = "";
  inputEl.style.height = "56px";
  if (shouldStick) scrollToBottom();

  try{
    setBusy(true);
    const { text } = await postJSON("/chat", { message: msg });
    addMsg("bot", text || "(no output)");
  } catch(e){
    addMsg("bot", `[error] ${e.message}`);
  } finally {
    setBusy(false);
    if (nearBottom()) scrollToBottom();
  }
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e)=>{
  // Shift+Enter sends; Enter inserts newline
  if(e.key === "Enter" && e.shiftKey){
    e.preventDefault();
    sendMessage();
  }
});
newBtn.addEventListener("click", ()=>{
  logEl.innerHTML = "";
  setStatus("ready","ready");
  inputEl.value = "";
  inputEl.style.height = "56px";
  inputEl.focus();
  scrollToBottom();
});

// Grow textarea with content (cap ~38% of viewport height)
inputEl.addEventListener("input", ()=>{
  inputEl.style.height = "auto";
  const max = Math.max(140, window.innerHeight * 0.38);
  inputEl.style.height = Math.min(inputEl.scrollHeight, max) + "px";
});

// First paint tweaks
window.addEventListener("load", ()=>{
  // Tip hidden by default; toggle with Ctrl/Cmd + ?
  if (tipWrap) tipWrap.style.display = "none";
  setStatus("ready","ready");
  inputEl.focus();
  scrollToBottom();
});

// Optional: toggle tip with Ctrl/Cmd + ?
window.addEventListener("keydown",(e)=>{
  if(e.key === "?" && (e.ctrlKey || e.metaKey)){
    e.preventDefault();
    if (tipWrap) tipWrap.style.display = tipWrap.style.display === "none" ? "block" : "none";
    if (nearBottom()) scrollToBottom();
  }
});
