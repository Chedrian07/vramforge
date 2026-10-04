"use client";

import { createContext, useContext, useSyncExternalStore, type ReactNode } from "react";

import type { ApiClient } from "./client";
import type { EventSourceFactory } from "./stream";

/** Opens the access-token prompt when any request returns 401 (docs/architecture.md §6). */
export interface AuthGate {
  request: () => void;
  close: () => void;
  subscribe: (listener: () => void) => () => void;
  isOpen: () => boolean;
}

export function createAuthGate(): AuthGate {
  let open = false;
  const listeners = new Set<() => void>();
  const emit = () => listeners.forEach((l) => l());
  return {
    request: () => {
      if (!open) {
        open = true;
        emit();
      }
    },
    close: () => {
      if (open) {
        open = false;
        emit();
      }
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    isOpen: () => open,
  };
}

export interface ApiEnvironment {
  api: ApiClient;
  eventSourceFactory: EventSourceFactory;
  authGate: AuthGate;
  /** Non-null in NEXT_PUBLIC_VF_DEV_MOCKS=1 mode: shown as a banner, never in production. */
  devMockLabel: string | null;
}

const ApiContext = createContext<ApiEnvironment | null>(null);

export function ApiProvider({ value, children }: { value: ApiEnvironment; children: ReactNode }) {
  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
}

export function useApiEnvironment(): ApiEnvironment {
  const value = useContext(ApiContext);
  if (!value) throw new Error("ApiProvider is missing");
  return value;
}

export function useApi(): ApiClient {
  return useApiEnvironment().api;
}

export function useAuthPromptOpen(): boolean {
  const { authGate } = useApiEnvironment();
  return useSyncExternalStore(authGate.subscribe, authGate.isOpen, () => false);
}
