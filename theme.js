// Light / dark theme toggle. Loaded in <head> so the saved theme applies before first paint.
(() => {
    const KEY = 'theme';
    const root = document.documentElement;
    let saved = null;
    try { saved = localStorage.getItem(KEY); } catch (e) {}
    if (saved === 'dark') root.setAttribute('data-theme', 'dark');

    const SUN = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
    const MOON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

    function isDark() { return root.getAttribute('data-theme') === 'dark'; }
    function paint(btn) {
        btn.innerHTML = isDark() ? SUN : MOON;
        btn.setAttribute('aria-label', isDark() ? 'Switch to light mode' : 'Switch to dark mode');
        btn.title = isDark() ? 'Light mode' : 'Dark mode';
    }
    function set(dark) {
        if (dark) root.setAttribute('data-theme', 'dark'); else root.removeAttribute('data-theme');
        try { localStorage.setItem(KEY, dark ? 'dark' : 'light'); } catch (e) {}
        window.dispatchEvent(new CustomEvent('themechange', { detail: { dark } }));
    }

    document.addEventListener('DOMContentLoaded', () => {
        const host = document.querySelector('.header-nav') || document.getElementById('header');
        if (!host) return;
        const btn = document.createElement('button');
        btn.className = 'theme-toggle';
        btn.type = 'button';
        paint(btn);
        btn.addEventListener('click', () => { set(!isDark()); paint(btn); });
        host.appendChild(btn);
    });
})();
