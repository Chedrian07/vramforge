import type { ApiClient } from "@/lib/api/client";
import type { EventSourceFactory } from "@/lib/api/stream";

/** Fixture-backed API used only with NEXT_PUBLIC_VF_DEV_MOCKS=1 (see next.config.ts). */
export interface DevMocks {
  api: ApiClient;
  eventSourceFactory: EventSourceFactory;
  label: string;
}
