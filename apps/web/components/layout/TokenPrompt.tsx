"use client";

import { useQueryClient } from "@tanstack/react-query";
import { Dialog } from "radix-ui";
import { useState, type FormEvent } from "react";

import { Button, Field, TextInput } from "@/components/ui/primitives";
import { ApiError } from "@/lib/api/client";
import { useApiEnvironment, useAuthPromptOpen } from "@/lib/api/context";

/** Shown on HTTP 401: posts the access token to /api/v1/session; the token is never stored. */
export function TokenPrompt() {
  const { api, authGate } = useApiEnvironment();
  const open = useAuthPromptOpen();
  const queryClient = useQueryClient();
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!token.trim()) {
      setError("접근 토큰을 입력하세요.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.createSession(token.trim());
      setToken("");
      authGate.close();
      await queryClient.invalidateQueries();
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) setError("이 서버는 토큰 세션을 제공하지 않습니다. 관리자에게 문의하세요.");
      else if (err instanceof ApiError && err.status === 401) setError("토큰이 올바르지 않습니다.");
      else setError(err instanceof ApiError ? err.issue.user_message : "토큰을 확인하지 못했습니다.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog.Root
      open={open}
      onOpenChange={(next) => {
        if (!next) {
          setToken("");
          authGate.close();
        }
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/40" />
        <Dialog.Content className="fixed top-1/2 left-1/2 z-50 w-[min(26rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-line bg-surface p-6 shadow-lg">
          <Dialog.Title className="text-lg font-semibold text-ink">접근 토큰 필요</Dialog.Title>
          <Dialog.Description className="mt-1 text-[13px] text-muted">
            이 서버는 접근 토큰을 요구합니다. 토큰은 세션 쿠키 발급에만 쓰고 브라우저에 저장하지 않습니다.
          </Dialog.Description>
          <form onSubmit={submit} className="mt-4 flex flex-col gap-4">
            <Field label="접근 토큰" htmlFor="access-token" error={error ?? undefined}>
              <TextInput
                id="access-token"
                type="password"
                autoComplete="off"
                value={token}
                onChange={(event) => setToken(event.target.value)}
                aria-invalid={error ? true : undefined}
                aria-describedby={error ? "access-token-error" : undefined}
              />
            </Field>
            <div className="flex justify-end gap-2">
              <Dialog.Close asChild>
                <Button>취소</Button>
              </Dialog.Close>
              <Button type="submit" variant="primary" disabled={busy}>
                {busy ? "확인 중…" : "확인"}
              </Button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
