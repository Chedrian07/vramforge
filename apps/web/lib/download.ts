// Saves a file the page fetched itself: a POST answer cannot be a plain <a href download>.
const REVOKE_AFTER_MS = 10_000;

export function saveFile(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Revoked later: some browsers start reading the URL only after the click handler returned.
  setTimeout(() => URL.revokeObjectURL(url), REVOKE_AFTER_MS);
}
