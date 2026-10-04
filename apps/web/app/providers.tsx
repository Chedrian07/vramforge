"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { devMocks } from "@vf/dev-mocks";
import { Tooltip } from "radix-ui";
import { useState, type ReactNode } from "react";

import { createHttpClient } from "@/lib/api/client";
import { ApiProvider, createAuthGate, type ApiEnvironment } from "@/lib/api/context";
import { browserEventSourceFactory } from "@/lib/api/stream";

function createEnvironment(): ApiEnvironment {
  const authGate = createAuthGate();
  if (devMocks) {
    return {
      api: devMocks.api,
      eventSourceFactory: devMocks.eventSourceFactory,
      authGate,
      devMockLabel: devMocks.label,
    };
  }
  return {
    api: createHttpClient({ onUnauthorized: authGate.request }),
    eventSourceFactory: browserEventSourceFactory,
    authGate,
    devMockLabel: null,
  };
}

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () => new QueryClient({ defaultOptions: { queries: { staleTime: 5_000, retry: 1, refetchOnWindowFocus: false } } }),
  );
  const [environment] = useState(createEnvironment);
  return (
    <QueryClientProvider client={client}>
      <ApiProvider value={environment}>
        <Tooltip.Provider delayDuration={300}>{children}</Tooltip.Provider>
      </ApiProvider>
    </QueryClientProvider>
  );
}
