#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const forbiddenLicense = /\b(?:AGPL|GPL|LGPL|SSPL|BUSL|Commons-Clause|NonCommercial)\b/i;
const reviewedIdentifiers = new Set([
  "0BSD",
  "Apache-2.0",
  "BSD-2-Clause",
  "BSD-3-Clause",
  "BSL-1.0",
  "CC0-1.0",
  "CDLA-Permissive-2.0",
  "ISC",
  "LLVM-exception",
  "MIT",
  "MIT-0",
  "MPL-2.0",
  "Unicode-3.0",
  "Unlicense",
  "Zlib",
]);

function fail(message) {
  console.error("License audit failed: " + message);
  process.exitCode = 1;
}

function readJson(relativePath) {
  return JSON.parse(fs.readFileSync(path.join(root, relativePath), "utf8"));
}

function auditLicenseExpression(owner, expression) {
  if (forbiddenLicense.test(expression)) {
    fail(owner + " uses a blocked license expression: " + expression);
  }
  const identifiers = String(expression).match(/[A-Za-z0-9.-]+/g) || [];
  const unknown = identifiers.filter(
    (identifier) => !["AND", "OR", "WITH"].includes(identifier) && !reviewedIdentifiers.has(identifier),
  );
  if (unknown.length) {
    fail(owner + " introduces unreviewed license identifier(s): " + unknown.join(", "));
  }
}

const packageJson = readJson("package.json");
if (packageJson.license !== "MIT OR Apache-2.0") {
  fail("package.json must declare MIT OR Apache-2.0");
}

for (const required of ["LICENSE-MIT", "LICENSE-APACHE", "THIRD_PARTY_NOTICES.md"]) {
  if (!fs.existsSync(path.join(root, required))) fail(required + " is missing");
}

const packageLock = readJson("package-lock.json");
const npmPackages = Object.entries(packageLock.packages || {}).filter(([name]) => name);
for (const [name, metadata] of npmPackages) {
  if (!metadata.license) fail("npm dependency has no declared license: " + name);
  auditLicenseExpression("npm dependency " + name, metadata.license || "");
}

const rustVersion = execFileSync("rustc", ["-vV"], { encoding: "utf8" });
const rustHost = rustVersion.match(/^host:\s*(.+)$/m)?.[1];
if (!rustHost) throw new Error("Could not determine the Rust host target");

const metadata = JSON.parse(
  execFileSync(
    "cargo",
    ["metadata", "--offline", "--format-version", "1", "--filter-platform", rustHost],
    { cwd: path.join(root, "src-tauri"), encoding: "utf8", maxBuffer: 128 * 1024 * 1024 },
  ),
);
const resolvedIds = new Set((metadata.resolve?.nodes || []).map((node) => node.id));
const rustPackages = metadata.packages.filter((item) => resolvedIds.has(item.id));
for (const item of rustPackages) {
  if (!item.license && !item.license_file) {
    fail("Rust dependency has no declared license: " + item.name + " " + item.version);
  }
  if (item.license) {
    auditLicenseExpression("Rust dependency " + item.name + " " + item.version, item.license);
  }
}

const notices = fs.readFileSync(path.join(root, "THIRD_PARTY_NOTICES.md"), "utf8");
if (!/WeSpeaker ECAPA-TDNN-512/.test(notices) || !/CC BY 4\.0/.test(notices)) {
  fail("the bundled WeSpeaker model and CC BY 4.0 terms are not documented");
}
if (!/Silero voice activity detector/.test(notices) || !/MIT\s+License/.test(notices)) {
  fail("the bundled Silero VAD model and MIT terms are not documented");
}

if (!process.exitCode) {
  const rustLicenseFamilies = new Set(rustPackages.map((item) => item.license || "license-file"));
  console.log(
    `License audit passed: Recall is MIT OR Apache-2.0; ${npmPackages.length} npm packages and ${rustPackages.length} Rust packages use reviewed license identifiers.`,
  );
  console.log(
    `Rust dependency graph contains ${rustLicenseFamilies.size} declared license expressions; model attribution remains separately documented as CC BY 4.0.`,
  );
}
