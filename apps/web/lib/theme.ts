// Theme preference (light / dark / system). Only the preference is stored; it is not sensitive.
export type ThemePreference = "light" | "dark" | "system";

const KEY = "vf-theme";
const listeners = new Set<() => void>();

export function readTheme(): ThemePreference {
  try {
    const value = window.localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

function prefersDark(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function applyTheme(preference: ThemePreference): void {
  const dark = preference === "dark" || (preference === "system" && prefersDark());
  document.documentElement.classList.toggle("dark", dark);
}

export function writeTheme(preference: ThemePreference): void {
  try {
    if (preference === "system") window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, preference);
  } catch {
    // Storage unavailable (private mode): the choice still applies to this page.
  }
  applyTheme(preference);
  listeners.forEach((listener) => listener());
}

export function subscribeTheme(listener: () => void): () => void {
  listeners.add(listener);
  const media = typeof window.matchMedia === "function" ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  const onSystemChange = () => {
    if (readTheme() === "system") applyTheme("system");
    listener();
  };
  media?.addEventListener("change", onSystemChange);
  return () => {
    listeners.delete(listener);
    media?.removeEventListener("change", onSystemChange);
  };
}
