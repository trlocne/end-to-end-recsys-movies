"use client";

import { useState } from "react";
import Link from "next/link";
import { setStoredUserId } from "@/lib/session";

export default function LoginPage() {
  const [userId, setUserId] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    const id = parseInt(userId, 10);
    if (Number.isNaN(id)) {
      setErr("User id must be a number.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: id }),
      });
      const text = await res.text();
      if (!res.ok) {
        let detail = text;
        try {
          const j = JSON.parse(text) as { detail?: string };
          if (j.detail) detail = j.detail;
        } catch {
          /* raw */
        }
        setErr(detail);
        return;
      }
      setStoredUserId(String(id));
      window.location.href = "/search";
    } catch (ce) {
      setErr(ce instanceof Error ? ce.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-md">
      <p className="text-sm font-medium uppercase tracking-widest text-ink">Log in</p>
      <h1 className="mt-2 font-display text-3xl font-medium text-ink md:text-4xl">
        Welcome back
      </h1>
      <p className="mt-3 text-body">
        Enter your numeric user id. The API only checks that the user exists.
      </p>
      <form onSubmit={onSubmit} className="card-auth mt-8 space-y-6 border border-mute/30">
        <div>
          <label htmlFor="uid" className="mb-2 block text-sm font-semibold text-ink">
            User id
          </label>
          <input
            id="uid"
            className="input-brand"
            inputMode="numeric"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="e.g. 1001"
            required
          />
        </div>
        {err && (
          <p className="rounded-input border border-primary/50 bg-canvas px-3 py-2 text-sm text-ink">
            {err}
          </p>
        )}
        <button type="submit" className="btn-primary w-full" disabled={loading}>
          {loading ? "Checking…" : "Continue"}
        </button>
      </form>
      <p className="mt-6 text-center text-body">
        New here?{" "}
        <Link href="/register" className="font-semibold text-ink underline">
          Sign up
        </Link>
      </p>
    </div>
  );
}
