"use client";

import { AlertDialog } from "radix-ui";
import { useState } from "react";

import { Button } from "@/components/ui/primitives";
import { ApiError, type ApiClient } from "@/lib/api/client";

/** Deletes the stored analysis and its artifacts after confirmation (plan.md §16.4). */
export function DeleteAnalysis({
  api,
  analysisId,
  disabled,
  onDeleted,
}: {
  api: Pick<ApiClient, "deleteAnalysis">;
  analysisId: string | null;
  disabled?: boolean;
  onDeleted: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!analysisId) return null;

  const confirm = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.deleteAnalysis(analysisId);
      setOpen(false);
      onDeleted();
    } catch (err) {
      setError(err instanceof ApiError ? err.issue.user_message : "삭제하지 못했습니다.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AlertDialog.Root open={open} onOpenChange={setOpen}>
      <AlertDialog.Trigger asChild>
        <Button variant="ghost" size="sm" disabled={disabled}>
          결과 삭제
        </Button>
      </AlertDialog.Trigger>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className="fixed inset-0 z-40 bg-black/40" />
        <AlertDialog.Content className="fixed top-1/2 left-1/2 z-50 w-[min(26rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-line bg-surface p-6 shadow-lg">
          <AlertDialog.Title className="text-lg font-semibold text-ink">분석 결과를 삭제할까요?</AlertDialog.Title>
          <AlertDialog.Description className="mt-2 text-[13px] leading-snug text-ink-2">
            서버에 저장된 결과와 길이 artifact, 연결된 업로드 참조를 삭제합니다. 되돌릴 수 없습니다.
          </AlertDialog.Description>
          {error ? (
            <p role="alert" className="mt-3 text-[13px] text-err">
              {error}
            </p>
          ) : null}
          <div className="mt-5 flex justify-end gap-2">
            <AlertDialog.Cancel asChild>
              <Button>취소</Button>
            </AlertDialog.Cancel>
            <Button variant="danger" onClick={() => void confirm()} disabled={busy}>
              {busy ? "삭제 중…" : "삭제"}
            </Button>
          </div>
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}
