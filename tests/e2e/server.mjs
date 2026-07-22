import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const sourceRoot = resolve(fileURLToPath(new URL("../../src", import.meta.url)));
const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".png": "image/png",
};

createServer(async (request, response) => {
  const pathname = new URL(request.url || "/", "http://127.0.0.1").pathname;
  const relative = pathname === "/" ? "index.html" : pathname.slice(1);
  const path = resolve(sourceRoot, relative);
  if (!path.startsWith(sourceRoot + "/") && path !== sourceRoot) {
    response.writeHead(403).end("Forbidden");
    return;
  }
  try {
    const content = await readFile(path);
    response.writeHead(200, {
      "cache-control": "no-store",
      "content-type": contentTypes[extname(path)] || "application/octet-stream",
    });
    response.end(content);
  } catch {
    response.writeHead(404).end("Not found");
  }
}).listen(4173, "127.0.0.1");
