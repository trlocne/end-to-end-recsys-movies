"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { AUTH_CHANGE_EVENT, getStoredUserId } from "@/lib/session";
import { fetchMoviesByIds } from "@/lib/movies";
import { MovieCard } from "@/components/MovieCard";
import type { MovieMeta, RecItem } from "@/lib/rec-types";

type RecentPayload = {
  user_id: number;
  recent_movie_ids: number[];
  recent_ratings: number[];
};

export default function RecentPage() {
  const [sessionUser, setSessionUser] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [rows, setRows] = useState<{ item: RecItem; rating: number }[]>([]);
  const [metaById, setMetaById] = useState<Map<number, MovieMeta>>(new Map());
  const [metaPending, setMetaPending] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const sync = () => setSessionUser(getStoredUserId());
    sync();
    window.addEventListener(AUTH_CHANGE_EVENT, sync);
    return () => window.removeEventListener(AUTH_CHANGE_EVENT, sync);
  }, []);

  const load = useCallback(async () => {
    setErr(null);
    setRows([]);
    setMetaById(new Map());
    setMetaPending(false);
    const raw = getStoredUserId();
    if (!raw?.trim()) {
      setErr("Log in to see your last five ratings from Postgres.");
      return;
    }
    const uid = parseInt(raw, 10);
    if (Number.isNaN(uid)) {
      setErr("Stored session is invalid. Log in again.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`/api/v1/feedback/recent/${uid}`);
      const text = await res.text();
      if (!res.ok) {
        setErr(text);
        return;
      }
      let data: RecentPayload;
      try {
        data = JSON.parse(text) as RecentPayload;
      } catch {
        setErr("Invalid JSON from recent API");
        return;
      }
      const ids = data.recent_movie_ids ?? [];
      const ratings = data.recent_ratings ?? [];
      const paired: { item: RecItem; rating: number }[] = ids.map((item_id, i) => ({
        item: { item_id },
        rating: typeof ratings[i] === "number" ? ratings[i] : Number(ratings[i]) || 0,
      }));
      setRows(paired);
      if (ids.length === 0) return;

      setMetaPending(true);
      void fetchMoviesByIds(ids, (partial) => setMetaById(new Map(partial)))
        .then((final) => setMetaById(new Map(final)))
        .catch(() => {})
        .finally(() => setMetaPending(false));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="mx-auto max-w-6xl">
      <p className="text-sm font-medium uppercase tracking-widest text-ink">History</p>
      <h1 className="mt-2 font-display text-3xl font-medium text-ink md:text-4xl">
        Recent movies
      </h1>
      <p className="mt-3 max-w-2xl text-body">
        Last up to five interactions from{" "}
        <code className="rounded bg-canvasSoft px-1">GET /v1/feedback/recent/&lt;user_id&gt;</code>{" "}
        (Postgres). Titles load after via{" "}
        <code className="rounded bg-canvasSoft px-1">/v1/movies/batch</code> (Postgres{" "}
        <code className="rounded bg-canvasSoft px-1">items</code>, then catalog parquet).
      </p>
      {sessionUser && (
        <p className="mt-2 text-sm text-bodyMid">
          Signed in as <span className="font-mono font-semibold text-ink">{sessionUser}</span>
        </p>
      )}

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="button"
          className="btn-secondary text-base"
          onClick={() => void load()}
          disabled={loading || !sessionUser}
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
        <Link href="/home" className="text-sm font-semibold text-ink underline">
          For you
        </Link>
      </div>

      {err && (
        <p className="mt-6 rounded-input border border-primary/50 bg-canvas px-3 py-2 text-sm text-ink">
          {err}
        </p>
      )}

      {!err && !loading && rows.length === 0 && sessionUser && (
        <p className="mt-8 text-bodyMid">No recent interactions in the database yet.</p>
      )}

      {rows.length > 0 && (
        <section className="mt-10">
          {metaPending ? (
            <p className="text-sm text-bodyMid">Loading titles from catalog…</p>
          ) : null}
          <div
            className={`grid gap-4 sm:grid-cols-2 lg:grid-cols-3 ${metaPending ? "mt-4" : "mt-6"}`}
          >
            {rows.map(({ item, rating }) => (
              <MovieCard
                key={item.item_id}
                item={item}
                meta={metaById.get(item.item_id)}
                metaPending={metaPending}
                secondaryLabel={`Your rating: ${Number.isInteger(rating) ? String(rating) : rating.toFixed(1)}`}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
