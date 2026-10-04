"use client";

import { HelpDialog } from "./HelpDialog";
import { ThemeToggle } from "./ThemeToggle";

export function Header() {
  return (
    <header className="mx-auto flex w-full max-w-[1320px] flex-col gap-4 px-4 pt-6 pb-4 sm:flex-row sm:items-start sm:justify-between sm:px-6 sm:pt-8">
      <div className="flex min-w-0 items-start gap-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/icon.svg" alt="" width={36} height={36} className="mt-0.5 shrink-0" />
        <div className="min-w-0">
          <p className="text-[12px] font-semibold tracking-wide text-accent uppercase">VRAMForge</p>
          <h1 className="text-xl font-semibold tracking-tight text-ink sm:text-2xl">Fine-Tuning VRAM Calculator</h1>
          <p className="text-[14px] text-muted">데이터셋을 자르지 않고 학습할 때 필요한 GPU 메모리</p>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <ThemeToggle />
        <HelpDialog />
      </div>
    </header>
  );
}
