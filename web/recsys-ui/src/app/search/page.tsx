"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  AUTH_CHANGE_EVENT,
  getStoredUserId,
} from "@/lib/session";
import { fetchMoviesByIds } from "@/lib/movies";
import { MovieResultGrid } from "@/components/MovieResultGrid";
import type { MovieMeta, RecItem, RecResponse } from "@/lib/rec-types";

export default function SearchPage() {
  const [q, setQ] = useState("");
  const [sessionUser, setSessionUser] = useState<string | null>(null);
  const [numItems, setNumItems] = useState("20");
  const [items, setItems] = useState<RecItem[]>([]);
  const [metaById, setMetaById] = useState<Map<number, MovieMeta>>(new Map());
  const [metaPending, setMetaPending] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const [rawJson, setRawJson] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const sync = () => setSessionUser(getStoredUserId());
    sync();
    window.addEventListener(AUTH_CHANGE_EVENT, sync);
    return () => window.removeEventListener(AUTH_CHANGE_EVENT, sync);
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setItems([]);
    setMetaById(new Map());
    setMetaPending(false);
    setRawJson(null);
    setLatency(null);
    const raw = getStoredUserId();
    if (!raw?.trim()) {
      setErr("Please log in first — your user id is taken from your session (no id field on this page).");
      return;
    }
    const uid = parseInt(raw, 10);
    if (Number.isNaN(uid)) {
      setErr("Stored session is invalid. Log in again.");
      return;
    }
    const n = parseInt(numItems, 10);
    const top = Number.isNaN(n) ? 20 : Math.min(Math.max(n, 1), 100);
    setLoading(true);
    try {
      const params = new URLSearchParams({ q, user_id: String(uid), num_items: String(top) });
      const res = await fetch(`/api/v1/recommend/search?${params}`);
      const text = await res.text();
      if (!res.ok) {
        setErr(text);
        return;
      }
      let data: RecResponse;
      try {
        data = JSON.parse(text) as RecResponse;
      } catch {
        setErr("Invalid JSON from search API");
        return;
      }
      setRawJson(JSON.stringify(data, null, 2));
      setLatency(typeof data.latency_ms === "number" ? data.latency_ms : null);
      const list = data.items ?? [];
      setItems(list);
      setMetaPending(list.length > 0);
      const ids = list.map((i) => i.item_id);
      if (ids.length === 0) {
        setMetaPending(false);
      } else {
        void fetchMoviesByIds(ids, (partial) => setMetaById(new Map(partial)))
          .then((final) => setMetaById(new Map(final)))
          .catch(() => {
            /* keep placeholders */
          })
          .finally(() => setMetaPending(false));
      }
    } catch (ce) {
      setErr(ce instanceof Error ? ce.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const uidNum = parseInt(sessionUser ?? "", 10);

  return (
    <div className="mx-auto max-w-6xl">
      <p className="text-sm font-medium uppercase tracking-widest text-ink">Search</p>
      <h1 className="mt-2 font-display text-3xl font-medium text-ink md:text-4xl">
        Find movies
      </h1>
      <p className="mt-3 max-w-2xl text-body">
        Recommendations use your <strong>logged-in user id</strong> automatically (no id in the URL or
        form). Results show movie ids first; titles load from the catalog afterward.
      </p>
      {sessionUser && (
        <p className="mt-2 text-sm text-bodyMid">
          Signed in as user <span className="font-mono font-semibold text-ink">{sessionUser}</span>
        </p>
      )}

      <form onSubmit={onSubmit} className="card-auth mt-8 space-y-4 border border-mute/30">
        <div>
          <label htmlFor="q" className="mb-2 block text-sm font-semibold text-ink">
            Query
          </label>
          <input
            id="q"
            className="input-brand"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="e.g. sci fi animation"
            required
          />
        </div>
        <div>
          <label htmlFor="n" className="mb-2 block text-sm font-semibold text-ink">
            Number of items
          </label>
          <input
            id="n"
            className="input-brand max-w-xs"
            inputMode="numeric"
            value={numItems}
            onChange={(e) => setNumItems(e.target.value)}
          />
        </div>
        {err && (
          <p className="rounded-input border border-primary/50 bg-canvas px-3 py-2 text-sm text-ink">
            {err}
          </p>
        )}
        <button type="submit" className="btn-primary" disabled={loading || !sessionUser}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {items.length > 0 && !Number.isNaN(uidNum) && (
        <>
          {latency != null && (
            <p className="mt-4 text-sm text-bodyMid">{latency.toFixed(0)} ms</p>
          )}
          <MovieResultGrid
            items={items}
            metaById={metaById}
            metaPending={metaPending}
            title="Results"
          />
        </>
      )}

      {rawJson && (
        <div className="mt-10">
          <button
            type="button"
            onClick={() => setShowRaw((s) => !s)}
            className="text-sm font-semibold text-ink underline"
          >
            {showRaw ? "Hide" : "Show"} raw JSON
          </button>
          {showRaw && (
            <pre className="card-content mt-4 max-h-[360px] overflow-auto whitespace-pre-wrap font-mono text-xs text-body">
              {rawJson}
            </pre>
          )}
        </div>
      )}

      <p className="mt-10 text-center text-body">
        {!sessionUser ? (
          <>
            <Link href="/login" className="font-semibold text-ink underline">
              Log in
            </Link>
            {" · "}
            <Link href="/register" className="font-semibold text-ink underline">
              Sign up
            </Link>
          </>
        ) : null}
      </p>
    </div>
  );
}
