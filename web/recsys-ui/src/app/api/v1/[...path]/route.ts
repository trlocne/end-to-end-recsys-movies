import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

function getRecsysBase(): string {
  const raw = process.env.RECSYS_API_BASE_URL?.trim().replace(/\/$/, "");
  if (raw) return raw;
  if (process.env.NODE_ENV === "development") return "http://127.0.0.1:8001";
  throw new Error(
    [
      "Missing RECSYS_API_BASE_URL (FastAPI base URL, no trailing slash).",
      "Production: set e.g. http://serving:8000 (same namespace as UI) or",
      "http://serving.serving.svc.cluster.local:8000 (cluster DNS).",
      "Local Next dev only: NODE_ENV=development falls back to http://127.0.0.1:8001.",
    ].join(" "),
  );
}

async function proxy(req: NextRequest, pathParts: string[]) {
  const base = getRecsysBase();
  const subpath = pathParts.join("/");
  const target = `${base}/v1/${subpath}${req.nextUrl.search}`;

  const headers: Record<string, string> = {
    Accept: req.headers.get("accept") ?? "application/json",
  };
  const ct = req.headers.get("content-type");
  if (ct) headers["Content-Type"] = ct;

  const init: RequestInit = {
    method: req.method,
    headers,
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }

  const res = await fetch(target, init);
  const text = await res.text();
  const out = new NextResponse(text, { status: res.status });
  const outCt = res.headers.get("content-type");
  if (outCt) out.headers.set("content-type", outCt);
  return out;
}

type Ctx = { params: Promise<{ path: string[] }> };

/** FastAPI chỉ có POST — GET trong browser không phải trang đăng nhập. */
const POST_ONLY_AUTH_PATHS = new Set(["auth/login", "auth/register"]);

export async function GET(req: NextRequest, ctx: Ctx) {
  const { path: segments } = await ctx.params;
  const joined = segments.join("/");
  if (POST_ONLY_AUTH_PATHS.has(joined)) {
    return NextResponse.json(
      {
        detail:
          'POST only: send JSON {"user_id": <number>}. The sign-in UI is at /login (same origin), not this API path.',
      },
      { status: 405 },
    );
  }
  return proxy(req, segments);
}

export async function POST(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function PUT(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function PATCH(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function DELETE(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}
