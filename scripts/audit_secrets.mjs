#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.resolve(process.argv[2] || scriptRoot);
const selfPath = "scripts/audit_secrets.mjs";
const binaryExtensions = new Set([
  ".icns",
  ".ico",
  ".jpg",
  ".jpeg",
  ".onnx",
  ".png",
  ".wav",
  ".m4a",
]);
const detectors = [
  ["private key", /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/],
  ["OpenAI key", /\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b/],
  ["GitHub token", /\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b/],
  ["AWS access key", /\bAKIA[0-9A-Z]{16}\b/],
  ["Azure storage key", /AccountKey=[A-Za-z0-9+/=]{20,}/i],
  [
    "credential assignment",
    /(?:api[_ -]?key|access[_ -]?key|client[_ -]?secret|password|auth[_ -]?token)\s*[:=]\s*["']?[A-Za-z0-9+/_=-]{24,}/i,
  ],
  ["Soniox-like 64-hex value", /\b[a-f0-9]{64}\b/i],
];

function isAllowedMatch(kind, line) {
  if (kind === "Soniox-like 64-hex value" && /sha(?:-?256)?|checksum|digest|hash|integrity/i.test(line)) {
    return true;
  }
  return /(?:YOUR|EXAMPLE|PLACEHOLDER|REPLACE_ME|APP_STORE_CONNECT|TEAMID|ORGANIZATION|\/absolute\/path|<[^>]+>)/i.test(
    line,
  );
}

function findingsInText(text, location, onlyPatchLines = false) {
  const findings = [];
  const lines = text.split(/\r?\n/);
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (onlyPatchLines && (!/^[+-]/.test(line) || /^\+\+\+|^---/.test(line))) continue;
    for (const [kind, detector] of detectors) {
      if (detector.test(line) && !isAllowedMatch(kind, line)) {
        findings.push({ location, line: index + 1, kind });
      }
    }
  }
  return findings;
}

const listed = spawnSync("git", ["ls-files", "-z", "--cached", "--others", "--exclude-standard"], {
  cwd: root,
  encoding: "utf8",
});
if (listed.status !== 0) throw new Error(listed.stderr || "git ls-files failed");

const findings = [];
for (const relativePath of listed.stdout.split("\0").filter(Boolean)) {
  if (
    (root === scriptRoot && relativePath === selfPath) ||
    binaryExtensions.has(path.extname(relativePath).toLowerCase())
  ) {
    continue;
  }
  const absolutePath = path.join(root, relativePath);
  const stats = fs.statSync(absolutePath);
  if (!stats.isFile() || stats.size > 5 * 1024 * 1024) continue;
  const contents = fs.readFileSync(absolutePath, "utf8");
  if (contents.includes("\0")) continue;
  findings.push(...findingsInText(contents, relativePath));
}

const history = spawnSync("git", ["log", "-p", "--all", "--no-ext-diff", "--", "."], {
  cwd: root,
  encoding: "utf8",
  maxBuffer: 256 * 1024 * 1024,
});
if (history.status !== 0) throw new Error(history.stderr || "git history scan failed");

let commit = "history";
let changedPath = "history";
const historyLines = history.stdout.split(/\r?\n/);
for (let index = 0; index < historyLines.length; index += 1) {
  const line = historyLines[index];
  if (line.startsWith("commit ")) commit = line.slice(7, 19);
  const diffMatch = line.match(/^diff --git a\/(.+) b\/(.+)$/);
  if (diffMatch) changedPath = diffMatch[2];
  if (
    (root === scriptRoot && changedPath === selfPath) ||
    !/^[+-]/.test(line) ||
    /^\+\+\+|^---/.test(line)
  ) {
    continue;
  }
  for (const [kind, detector] of detectors) {
    if (detector.test(line) && !isAllowedMatch(kind, line)) {
      findings.push({ location: commit + ":" + changedPath, line: index + 1, kind });
    }
  }
}

const unique = Array.from(
  new Map(findings.map((finding) => [finding.location + "|" + finding.line + "|" + finding.kind, finding])).values(),
);
if (unique.length) {
  console.error(`Secret audit failed with ${unique.length} redacted finding(s):`);
  for (const finding of unique) {
    console.error(`- ${finding.kind} at ${finding.location}:${finding.line}`);
  }
  console.error("No credential values were printed. Inspect and rotate any confirmed secret before publishing.");
  process.exit(1);
}

console.log("Secret audit passed: no credential patterns found in tracked/untracked source or Git patch history.");
