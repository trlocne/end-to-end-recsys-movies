"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { AUTH_CHANGE_EVENT, getStoredUserId } from "@/lib/session";
import { fetchMoviesByIds } from "@/lib/movies";
import { MovieResultGrid } from "@/components/MovieResultGrid";
import type { MovieMeta, RecItem, RecResponse } from "@/lib/rec-types";

export function MovieDetailClient() {
  const params = useParams();
  const itemIdRaw = params?.itemId as string;
  const itemId = parseInt(itemIdRaw, 10);

  const [sessionUser, setSessionUser] = useState<string | null>(null);
  const [meta, setMeta] = useState<MovieMeta | null>(null);
  const [metaHeaderPending, setMetaHeaderPending] = useState(true);
  const [similar, setSimilar] = useState<RecItem[]>([]);
  const [metaById, setMetaById] = useState<Map<number, MovieMeta>>(new Map());
  const [similarMetaPending, setSimilarMetaPending] = useState(false);
  const [rating, setRating] = useState("4");
  const [feedbackMsg, setFeedbackMsg] = useState<string | null>(null);
  const [feedbackErr, setFeedbackErr] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [loadingSimilar, setLoadingSimilar] = useState(false);
  const [ratingSubmitPending, setRatingSubmitPending] = useState(false);

  useEffect(() => {
    const sync = () => setSessionUser(getStoredUserId());
    sync();
    window.addEventListener(AUTH_CHANGE_EVENT, sync);
    return () => window.removeEventListener(AUTH_CHANGE_EVENT, sync);
  }, []);

  const loadMeta = useCallback(async () => {
    if (Number.isNaN(itemId)) return;
    setMeta(null);
    setMetaHeaderPending(true);
    try {
      await fetchMoviesByIds([itemId], (partial) => {
        const m = partial.get(itemId);
        if (m) setMeta(m);
      });
    } catch {
      /* keep fallback title */
    } finally {
      setMetaHeaderPending(false);
    }
  }, [itemId]);

  const loadSimilar = useCallback(async () => {
    if (Number.isNaN(itemId)) return;
    const raw = sessionUser?.trim();
    if (!raw) {
      setSimilar([]);
      setMetaById(new Map());
      setSimilarMetaPending(false);
      return;
    }
    const uid = parseInt(raw, 10);
    if (Number.isNaN(uid)) {
      setSimilar([]);
      setMetaById(new Map());
      setSimilarMetaPending(false);
      return;
    }
    setLoadingSimilar(true);
    setLoadErr(null);
    try {
      const q = new URLSearchParams({
        item_id: String(itemId),
        user_id: String(uid),
        num_items: "20",
      });
      const res = await fetch(`/api/v1/recommend/item?${q}`);
      const text = await res.text();
      if (!res.ok) {
        setLoadErr(text);
        setSimilar([]);
        setSimilarMetaPending(false);
        return;
      }
      const data = JSON.parse(text) as RecResponse;
      const list = data.items ?? [];
      setSimilar(list);
      setMetaById(new Map());
      const ids = list.map((i) => i.item_id);
      if (ids.length === 0) {
        setSimilarMetaPending(false);
      } else {
        setSimilarMetaPending(true);
        void fetchMoviesByIds(ids, (partial) => setMetaById(new Map(partial)))
          .then((final) => setMetaById(new Map(final)))
          .catch(() => {})
          .finally(() => setSimilarMetaPending(false));
      }
    } catch (e) {
      setLoadErr(e instanceof Error ? e.message : "Failed to load similar");
      setSimilar([]);
      setSimilarMetaPending(false);
    } finally {
      setLoadingSimilar(false);
    }
  }, [itemId, sessionUser]);

  useEffect(() => {
    loadMeta();
  }, [loadMeta]);

  useEffect(() => {
    loadSimilar();
  }, [loadSimilar]);

  async function submitRating(e: React.FormEvent) {
    e.preventDefault();
    setFeedbackMsg(null);
    setFeedbackErr(null);
    const raw = sessionUser?.trim();
    const uid = raw ? parseInt(raw, 10) : NaN;
    const r = parseFloat(rating);
    if (!raw || Number.isNaN(uid)) {
      setFeedbackErr("Log in from the header — ratings use your session user id.");
      return;
    }
    if (Number.isNaN(r) || r < 1 || r > 5) {
      setFeedbackErr("Rating must be between 1 and 5.");
      return;
    }
    setRatingSubmitPending(true);
    try {
      const res = await fetch("/api/v1/feedback/click", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: uid,
          item_id: itemId,
          event_type: "click",
          rating: r,
        }),
      });
      const text = await res.text();
      if (!res.ok) {
        setFeedbackErr(text);
        return;
      }
      setFeedbackMsg("Rating saved. Recommendations may refresh on your next request.");
      void loadSimilar();
    } catch (err) {
      setFeedbackErr(err instanceof Error ? err.message : "Submit failed");
    } finally {
      setRatingSubmitPending(false);
    }
  }

  if (Number.isNaN(itemId)) {
    return (
      <div className="mx-auto max-w-xl text-body">
        <p>Invalid movie id.</p>
        <Link href="/search" className="mt-4 inline-block font-semibold text-ink underline">
          Back to search
        </Link>
      </div>
    );
  }

  const titleTrim = meta?.title?.trim();
  const headerTitle = titleTrim
    ? titleTrim
    : metaHeaderPending
      ? "Loading title…"
      : `Movie #${itemId}`;

  return (
    <div className="mx-auto max-w-6xl">
      <Link
        href="/search"
        className="text-sm font-semibold text-body underline hover:text-ink"
      >
        ← Search
      </Link>
      <p className="mt-4 text-sm font-medium uppercase tracking-widest text-ink">Movie</p>
      <p className="mt-2 font-mono text-lg font-semibold text-ink">{itemId}</p>
      <h1
        className={`mt-2 font-display text-3xl font-medium md:text-5xl ${
          metaHeaderPending && !titleTrim ? "text-bodyMid italic" : "text-ink"
        }`}
      >
        {headerTitle}
      </h1>
      {(meta?.genres || meta?.tag) && (
        <p className="mt-4 max-w-3xl text-lg text-body">
          {[meta?.genres, meta?.tag].filter(Boolean).join(" · ")}
        </p>
      )}

      <div className="card-auth mt-8 max-w-xl border border-mute/30">
        <h2 className="font-display text-lg font-semibold text-ink">Your rating</h2>
        <p className="mt-2 text-sm text-body">
          Uses your logged-in user from the header (same session as Home / Search). Posts to{" "}
          <code className="rounded bg-canvas px-1">POST /v1/feedback/click</code> (Postgres when configured).
        </p>
        {sessionUser && (
          <p className="mt-2 text-sm text-bodyMid">
            Signed in as{" "}
            <span className="font-mono font-semibold text-ink">{sessionUser}</span>
          </p>
        )}
        <form onSubmit={submitRating} className="mt-6 space-y-4">
          <div>
            <label htmlFor="rating" className="mb-2 block text-sm font-semibold text-ink">
              Rating (1–5)
            </label>
            <input
              id="rating"
              type="number"
              min={1}
              max={5}
              step={0.5}
              className="input-brand w-40"
              value={rating}
              onChange={(e) => setRating(e.target.value)}
            />
          </div>
          {feedbackErr && (
            <p className="text-sm text-primary">{feedbackErr}</p>
          )}
          {feedbackMsg && (
            <p className="text-sm text-ink">{feedbackMsg}</p>
          )}
          <button
            type="submit"
            className="btn-primary inline-flex items-center justify-center gap-2"
            disabled={ratingSubmitPending || !sessionUser?.trim()}
          >
            {ratingSubmitPending && (
              <span
                className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-onPrimary/30 border-t-onPrimary"
                aria-hidden
              />
            )}
            {ratingSubmitPending ? "Saving…" : "Submit rating"}
          </button>
        </form>
      </div>

      <section className="mt-12">
        <h2 className="font-display text-2xl font-medium text-ink">Similar titles</h2>
        <p className="mt-2 text-body">
          From <code className="rounded bg-canvasSoft px-1">GET /v1/recommend/item</code> using this
          movie and your logged-in session.
        </p>
        {loadingSimilar && <p className="mt-4 text-bodyMid">Loading…</p>}
        {loadErr && (
          <p className="mt-4 rounded-input border border-primary/50 bg-canvas px-3 py-2 text-sm text-ink">
            {loadErr}
          </p>
        )}
        {(() => {
          const uid = parseInt(sessionUser ?? "", 10);
          if (Number.isNaN(uid)) {
            return (
              <p className="mt-4 text-bodyMid">
                Log in from the header so your user id is in session — needed for similar titles and
                ratings.
              </p>
            );
          }
          if (similar.length === 0 && !loadingSimilar && !loadErr) {
            return (
              <p className="mt-4 text-bodyMid">No similar titles returned.</p>
            );
          }
          if (similar.length > 0) {
            return (
              <MovieResultGrid
                items={similar}
                metaById={metaById}
                metaPending={similarMetaPending}
                title=""
              />
            );
          }
          return null;
        })()}
      </section>
    </div>
  );
}
