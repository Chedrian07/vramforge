// Date/time display (Korean locale, the viewer's time zone).
const dateTime = new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" });

/** ISO timestamp -> "2026. 10. 12. 오후 2:00"; null when missing or unreadable. */
export function formatDateTime(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : dateTime.format(date);
}
