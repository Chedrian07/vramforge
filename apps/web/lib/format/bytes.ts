// Display formatting only: results stay in integer bytes; nothing is rounded before display
// (plan.md §3.2 "화면 1자리 소수, tooltip 원본 bytes").
export const GIB = 1_073_741_824;
export const MIB = 1_048_576;
export const KIB = 1_024;

const oneDecimal = new Intl.NumberFormat("ko-KR", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const integer = new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 });

/** GiB value with one decimal, no unit (e.g. "12.3"). */
export function gibNumber(bytes: number): string {
  return oneDecimal.format(bytes / GIB);
}

/** "12.3 GiB"; null stays null so callers render "산정 불가" instead of a number. */
export function formatGiB(bytes: number | null | undefined): string | null {
  return bytes == null ? null : `${gibNumber(bytes)} GiB`;
}

/**
 * Binary-unit size with one decimal. Small non-zero values switch to MiB/KiB/B so that a real
 * allocation never reads as "0.0 GiB".
 */
export function formatSize(bytes: number | null | undefined): string | null {
  if (bytes == null) return null;
  const abs = Math.abs(bytes);
  if (abs === 0) return "0 B";
  if (abs >= 0.1 * GIB) return `${oneDecimal.format(bytes / GIB)} GiB`;
  if (abs >= 0.1 * MIB) return `${oneDecimal.format(bytes / MIB)} MiB`;
  if (abs >= 0.1 * KIB) return `${oneDecimal.format(bytes / KIB)} KiB`;
  return `${integer.format(bytes)} B`;
}

/** "12.3 – 15.8 GiB" (or a single value when low == high). Null when either side is unknown. */
export function formatGiBRange(
  low: number | null | undefined,
  high: number | null | undefined,
): string | null {
  if (low == null || high == null) return null;
  if (gibNumber(low) === gibNumber(high)) return `${gibNumber(high)} GiB`;
  return `${gibNumber(low)} – ${gibNumber(high)} GiB`;
}

/** Same as formatGiBRange but with adaptive units per side. */
export function formatSizeRange(
  low: number | null | undefined,
  high: number | null | undefined,
): string | null {
  if (low == null || high == null) return null;
  const lo = formatSize(low);
  const hi = formatSize(high);
  return lo === hi ? hi : `${lo} – ${hi}`;
}

/** Exact integer bytes for tooltips and tables: "13,207,862,272 bytes". */
export function exactBytes(bytes: number | null | undefined): string | null {
  return bytes == null ? null : `${integer.format(bytes)} bytes`;
}

export function exactBytesRange(
  low: number | null | undefined,
  high: number | null | undefined,
): string | null {
  if (low == null || high == null) return null;
  return low === high
    ? `${integer.format(high)} bytes`
    : `${integer.format(low)} – ${integer.format(high)} bytes`;
}

export function formatCount(n: number | null | undefined): string | null {
  return n == null ? null : integer.format(n);
}

/** Ratio 1.264 -> "126%". Kept as text even above 100% (plan.md §3.4). */
export function formatPercent(ratio: number | null | undefined): string | null {
  return ratio == null ? null : `${integer.format(Math.round(ratio * 100))}%`;
}

export function formatNumber(n: number | null | undefined, digits = 1): string | null {
  if (n == null) return null;
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: digits }).format(n);
}

/** GiB text typed by a user -> integer bytes (null when empty or invalid). */
export function gibInputToBytes(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value < 0) return null;
  return Math.round(value * GIB);
}

/** Integer bytes -> compact GiB input text (up to 3 decimals, no grouping). */
export function bytesToGibInput(bytes: number | null | undefined): string {
  if (bytes == null) return "";
  return String(Math.round((bytes / GIB) * 1000) / 1000);
}

/** Shortens digests/ids for display; the full value stays available via copy/tooltip. */
export function shortDigest(value: string | null | undefined, keep = 12): string | null {
  if (value == null) return null;
  const v = value.startsWith("sha256:") ? value.slice(7) : value;
  return v.length > keep ? `${v.slice(0, keep)}…` : v;
}
