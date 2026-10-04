"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Tooltip } from "radix-ui";
import { useState, type ReactNode } from "react";

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () => new QueryClient({ defaultOptions: { queries: { staleTime: 5_000, retry: 1 } } }),
  );
  return (
    <QueryClientProvider client={client}>
      <Tooltip.Provider delayDuration={300}>{children}</Tooltip.Provider>
    </QueryClientProvider>
  );
}
