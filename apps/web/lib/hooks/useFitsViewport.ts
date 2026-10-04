"use client";

import { useEffect, useState, type RefObject } from "react";

/**
 * Whether an element fits in the viewport. The sticky summary uses it to drop `sticky` when the
 * card is taller than the screen instead of forcing an inner scroll (plan.md §3.1).
 */
export function useFitsViewport(ref: RefObject<HTMLElement | null>, offset = 48): boolean {
  const [fits, setFits] = useState(true);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () => setFits(element.offsetHeight + offset <= window.innerHeight);
    const frame = requestAnimationFrame(update);
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    observer?.observe(element);
    window.addEventListener("resize", update);
    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [ref, offset]);
  return fits;
}
