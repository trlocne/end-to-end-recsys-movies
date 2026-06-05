"use client";

import { useState } from "react";
import Link from "next/link";
import { setStoredUserId } from "@/lib/session";

export default function RegisterPage() {
  const [userId, setUserId] = useState("");
  const [metadata, setMetadata] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setMsg(null);
    const id = parseInt(userId, 10);
    if (Number.isNaN(id)) {
      setErr("User id must be a number.");
      return;
    }
    setLoading(true);
    try {
      let meta: Record<string, unknown> | undefined;
      if (metadata.trim()) {
        try {
          meta = JSON.parse(metadata) as Record<string, unknown>;
        } catch {
          setErr("Metadata must be valid JSON object.");
          setLoading(false);
          return;
        }
      }
      const res = await fetch("/api/v1/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: id, metadata: meta }),
      });
      const text = await res.text();
      if (!res.ok) {
        let detail = text;
        try {
          const j = JSON.parse(text) as { detail?: string };
          if (j.detail) detail = j.detail;
        } catch {
          /* use raw */
        }
        setErr(detail);
        return;
      }
      setStoredUserId(String(id));
      setMsg("Registered. You can search or open For you.");
      setUserId("");
      setMetadata("");
    } catch (ce) {
      setErr(ce instanceof Error ? ce.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-md">
      <p className="text-sm font-medium uppercase tracking-widest text-ink">Sign up</p>
      <h1 className="mt-2 font-display text-3xl font-medium text-ink md:text-4xl">
        Join SeanMovies
      </h1>
      <p className="mt-3 text-body">
        Use a numeric user id. This mirrors the demo API — there is no password yet.
      </p>
      <form onSubmit={onSubmit} className="card-auth mt-8 space-y-6 border border-mute/30">
        <div>
          <label htmlFor="user_id" className="mb-2 block text-sm font-semibold text-ink">
            User id
          </label>
          <input
            id="user_id"
            className="input-brand"
            inputMode="numeric"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="e.g. 1001"
            required
          />
        </div>
        <div>
          <label htmlFor="metadata" className="mb-2 block text-sm font-semibold text-ink">
            Metadata (optional JSON)
          </label>
          <textarea
            id="metadata"
            className="input-brand min-h-[100px] font-mono text-sm"
            value={metadata}
            onChange={(e) => setMetadata(e.target.value)}
            placeholder='{"source":"web"}'
          />
        </div>
        {err && (
          <p className="rounded-input border border-primary/50 bg-canvas px-3 py-2 text-sm text-ink">
            {err}
          </p>
        )}
        {msg && (
          <p className="rounded-input border border-mute bg-canvasSoft px-3 py-2 text-sm text-ink">
            {msg}{" "}
            <Link href="/search" className="font-semibold text-primary underline">
              Search
            </Link>
          </p>
        )}
        <button type="submit" className="btn-primary w-full" disabled={loading}>
          {loading ? "Working…" : "Create account"}
        </button>
      </form>
      <p className="mt-6 text-center text-body">
        Already have an id?{" "}
        <Link href="/login" className="font-semibold text-ink underline">
          Log in
        </Link>
      </p>
    </div>
  );
}
