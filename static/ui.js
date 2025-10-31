/* toddric-spidey UI — robust selectors, autoscroll, disabled send, slow indicator */
(() => {
  // --------- helpers ----------
  const qsAny = (selList) => {
    for (const sel of selList) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    return null;
  };

  // Elements (fallbacks so it survives markup tweaks)
  const log     = qsAny(['#log','main#log','.chat-log','[data-role="log"]']);
  const form    = qsAny(['#composer','form#composer','[data-role="composer"]','form[action="/chat"]']);
  const input   = qsAny(['#msg','#input','textarea#input','textarea[name="message"]','[data-role="input"]','textarea']);
  const sendBtn = qsAny(['#send','#sendBtn','button#sendBtn','button[type="submit"]','[data-role="send"]']);
  const status  = qsAny(['#statusText','.status-text','[data-role="status"]']);
  const newBtn  = qsAny(['#newChatBtn','.new-chat','[data-role="new-chat"]']);

  // Pick a scroller: prefer the chat log; otherwise the page itself
  const pageScroller = () => document.scrollingElement || document.documentElement || document.body;
  const scroller = log || pageScroller();
  const isLog    = !!log;

  const setStatus = (text, state='') => {
    if (!status) return;
    status.textContent = text;
    status.dataset.state = state; // ready|sending|slow
  };

  const setSending = (flag) => {
    if (sendBtn) {
      sendBtn.disabled = flag;
      sendBtn.classList.toggle('is-disabled', flag);
      sendBtn.setAttribute('aria-busy', String(flag));
    }
    if (form) form.classList.toggle('is-sending', flag);
  };

  const nearBottom = () => {
    if (isLog) {
      return scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 120;
    }
    const ps = pageScroller();
    return ps.scrollTop + window.innerHeight >= ps.scrollHeight - 120;
    };

  const scrollToBottom = (force=false) => {
    if (!scroller) return;
    if (!force && !nearBottom()) return;
    requestAnimationFrame(() => {
      if (isLog) {
        scroller.scrollTop = scroller.scrollHeight;
      } else {
        const ps = pageScroller();
        window.scrollTo({ top: ps.scrollHeight, behavior: 'instant' in window ? 'instant' : 'auto' });
      }
    });
  };

  // Mutation observer: autoscroll when new bubbles appear
  if (log) {
    const mo = new MutationObserver(() => scrollToBottom(false));
    mo.observe(log, { childList: true, subtree: true });
  }

  const escapeHtml = (s) => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  const bubbleUser = (t) => `<div class="msg right"><div class="bubble user">${escapeHtml(t)}</div></div>`;
  const bubbleBot  = (t) => `<div class="msg left"><div class="bubble bot"><pre>${escapeHtml(t)}</pre></div></div>`;

  const appendHTML = (html) => {
    const wasNear = nearBottom();
    const wrap = document.createElement('div');
    wrap.innerHTML = html;
    const parent = log || document.body;
    while (wrap.firstChild) parent.appendChild(wrap.firstChild);
    if (wasNear) scrollToBottom(true);
  };

  const postChat = async (message) => {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({ message })
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  };

  const SLOW_MS = 1800;

  const send = async (e) => {
    if (e) e.preventDefault();
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;

    appendHTML(bubbleUser(text));
    input.value = '';
    input.style.height = 'auto';

    setSending(true);
    setStatus('sending…', 'sending');
    const slowTimer = setTimeout(() => setStatus('sending… (slow)', 'slow'), SLOW_MS);

    try {
      const data = await postChat(text);
      clearTimeout(slowTimer);
      setStatus('ready','ready');
      const reply = (data && data.text) ? String(data.text) : '[no response]';
      appendHTML(bubbleBot(reply));
    } catch (err) {
      clearTimeout(slowTimer);
      setStatus('ready','ready');
      appendHTML(bubbleBot(`[error] ${err.message || err}`));
    } finally {
      setSending(false);
      scrollToBottom(true);
      input.focus();
    }
  };

  if (form) form.addEventListener('submit', send);
  if (sendBtn) sendBtn.addEventListener('click', send);

  // Shift+Enter = send, Enter = newline
  if (input) {
    input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && ev.shiftKey) {
        ev.preventDefault();
        send();
      }
    });
    // auto-grow
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = `${Math.min(240, input.scrollHeight)}px`;
      scrollToBottom();
    });
  }

  // New chat clears the log
  if (newBtn) {
    newBtn.addEventListener('click', () => {
      if (log) log.innerHTML = '';
      setStatus('ready','ready');
      if (input) { input.value=''; input.style.height='auto'; input.focus(); }
      scrollToBottom(true);
    });
  }

  // Initial status/layout
  setStatus('ready','ready');
  scrollToBottom(true);
})();
