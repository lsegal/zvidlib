// Page chrome: tab switching and a small syntax highlighter for the code samples. The demo
// itself lives in demo.js.

for (const tablist of document.querySelectorAll('[role="tablist"]')) {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  const select = (tab) => {
    for (const other of tabs) {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      other.tabIndex = selected ? 0 : -1;
      document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
    }
    tab.dispatchEvent(new CustomEvent("tabselected", { bubbles: true }));
  };
  tablist.addEventListener("click", (event) => {
    const tab = event.target.closest('[role="tab"]');
    if (tab) select(tab);
  });
  tablist.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    const next = tabs[(tabs.indexOf(document.activeElement) + step + tabs.length) % tabs.length];
    next.focus();
    select(next);
  });
}

const KEYWORDS = {
  js: /\b(import|from|await|const|let|for|new|return|async|function|if|else|try|catch)\b/,
  rust: /\b(let|mut|fn|use|pub|impl|for|in|return|match|if|else|Ok|Err|Some|None|unwrap)\b/,
};

// Comments, strings and numbers first, so a keyword inside one is left alone.
function highlight(code, language) {
  const tokens = new RegExp(
    [
      /(\/\/[^\n]*)/.source,
      /("(?:[^"\\]|\\.)*")/.source,
      /(\b\d[\d_]*n?\b)/.source,
      `(${KEYWORDS[language].source})`,
      /(\b[A-Z][A-Za-z0-9]+\b)/.source,
    ].join("|"),
    "g",
  );
  const escape = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let html = "";
  let last = 0;
  for (const match of code.matchAll(tokens)) {
    html += escape(code.slice(last, match.index));
    const kind = match[1] ? "c" : match[2] ? "s" : match[3] ? "n" : match[4] ? "k" : "t";
    html += `<span class="tok-${kind}">${escape(match[0])}</span>`;
    last = match.index + match[0].length;
  }
  return html + escape(code.slice(last));
}

for (const block of document.querySelectorAll("code.lang-js, code.lang-rust")) {
  block.innerHTML = highlight(block.textContent, block.classList.contains("lang-js") ? "js" : "rust");
}
