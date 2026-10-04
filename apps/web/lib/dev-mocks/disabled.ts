// Default target of the `@vf/dev-mocks` alias: production and normal dev builds get no mocks.
import type { DevMocks } from "./types";

export const devMocks: DevMocks | null = null;
