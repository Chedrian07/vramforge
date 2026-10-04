import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderOptions } from "@testing-library/react";
import { Tooltip } from "radix-ui";
import type { ReactElement, ReactNode } from "react";

import type { ApiClient } from "@/lib/api/client";
import { ApiProvider, createAuthGate, type ApiEnvironment } from "@/lib/api/context";

import { fakeEventSourceFactory } from "./fake-event-source";
import { createFakeApi } from "./fake-api";

export function makeEnvironment(api: Partial<ApiClient> = {}): ApiEnvironment {
  return {
    api: createFakeApi(api),
    eventSourceFactory: fakeEventSourceFactory,
    authGate: createAuthGate(),
    devMockLabel: null,
  };
}

export function renderWithProviders(
  ui: ReactElement,
  {
    environment = makeEnvironment(),
    client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity }, mutations: { retry: false } } }),
    ...options
  }: { environment?: ApiEnvironment; client?: QueryClient } & RenderOptions = {},
) {
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>
      <ApiProvider value={environment}>
        <Tooltip.Provider delayDuration={0}>{children}</Tooltip.Provider>
      </ApiProvider>
    </QueryClientProvider>
  );
  return { ...render(ui, { wrapper: Wrapper, ...options }), environment, client };
}
