import { defineConfig } from "@playwright/test";

const requestedPort = Number.parseInt(process.env.RECALL_E2E_PORT || "4173", 10);
const testPort =
  Number.isInteger(requestedPort) && requestedPort >= 1 && requestedPort <= 65_535
    ? requestedPort
    : 4173;

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 20_000,
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  use: {
    baseURL: `http://127.0.0.1:${testPort}`,
    channel: "chrome",
    headless: true,
  },
  webServer: {
    command: `RECALL_E2E_PORT=${testPort} node tests/e2e/server.mjs`,
    url: `http://127.0.0.1:${testPort}`,
    reuseExistingServer: false,
    timeout: 10_000,
  },
});
