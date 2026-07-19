import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const html = fs.readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
const javascript = fs.readFileSync(new URL("../src/main.js", import.meta.url), "utf8");
const rustMain = fs.readFileSync(new URL("../src-tauri/src/main.rs", import.meta.url), "utf8");
const rustSoniox = fs.readFileSync(new URL("../src-tauri/src/soniox.rs", import.meta.url), "utf8");
const tauriConfig = fs.readFileSync(
  new URL("../src-tauri/tauri.conf.json", import.meta.url),
  "utf8",
);
const packageJson = fs.readFileSync(new URL("../package.json", import.meta.url), "utf8");

function matches(source, expression) {
  return Array.from(source.matchAll(expression), (match) => match[1]);
}

test("every element requested by the UI controller exists exactly once", () => {
  const htmlIds = matches(html, /\bid="([^"]+)"/g);
  const requestedIds = matches(javascript, /getElementById\("([^"]+)"\)/g);
  assert.equal(new Set(htmlIds).size, htmlIds.length, "HTML contains duplicate element IDs");
  const missing = requestedIds.filter((id) => !htmlIds.includes(id));
  assert.deepEqual(missing, []);
});

test("every native command invoked by the UI is registered by Tauri", () => {
  const invoked = new Set(matches(javascript, /invoke\("([^"]+)"/g));
  const handlerBlock = rustMain.match(/generate_handler!\[([\s\S]*?)\]\)/);
  assert(handlerBlock, "Tauri command registration block not found");
  const registered = new Set(
    handlerBlock[1]
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean),
  );
  assert.deepEqual(
    Array.from(invoked).filter((command) => !registered.has(command)),
    [],
  );
});

test("every native event listened for by the UI is emitted by the app", () => {
  const listened = new Set(matches(javascript, /listen\("([^"]+)"/g));
  const emitted = new Set([
    ...matches(rustMain, /\.emit\(\s*"([^"]+)"/g),
    ...matches(rustSoniox, /\.emit\(\s*"([^"]+)"/g),
  ]);
  assert.deepEqual(
    Array.from(listened).filter((event) => !emitted.has(event)),
    [],
  );
});

test("active desktop code has no localhost API or Azure dependency", () => {
  for (const source of [html, javascript, rustMain, rustSoniox, tauriConfig, packageJson]) {
    assert.doesNotMatch(source, /localhost:\d+|api base url|azure/i);
  }
  assert.doesNotMatch(packageJson, /http-server|npm-run-all/i);
});
