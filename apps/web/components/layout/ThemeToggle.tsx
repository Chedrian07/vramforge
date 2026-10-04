"use client";

import { useSyncExternalStore } from "react";

import { Segmented } from "@/components/ui/controls";
import { readTheme, subscribeTheme, writeTheme, type ThemePreference } from "@/lib/theme";

export function ThemeToggle() {
  const theme = useSyncExternalStore<ThemePreference>(subscribeTheme, readTheme, () => "system");
  return (
    <Segmented
      label="테마"
      size="sm"
      value={theme}
      onValueChange={writeTheme}
      options={[
        { value: "light", label: "라이트" },
        { value: "dark", label: "다크" },
        { value: "system", label: "시스템" },
      ]}
    />
  );
}
