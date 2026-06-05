"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  AUTH_CHANGE_EVENT,
  clearStoredUserId,
  getStoredUserId,
} from "@/lib/session";

export function NavBar() {
  const [userId, setUserId] = useState<string | null>(null);

  useEffect(() => {
    const sync = () => setUserId(getStoredUserId());
    sync();
    window.addEventListener("storage", sync);
    window.addEventListener(AUTH_CHANGE_EVENT, sync);
    return () => {
      window.removeEventListener("storage", sync);
      window.removeEventListener(AUTH_CHANGE_EVENT, sync);
    };
  }, []);

  function logout() {
    clearStoredUserId();
  }

  return (
    <header className="border-b border-mute/40 bg-canvas px-6 py-3 md:px-8">
      <nav className="mx-auto flex max-w-container flex-wrap items-center justify-between gap-4">
        <Link
          href="/"
          className="font-display text-xl font-bold tracking-tight text-ink md:text-2xl"
        >
          SeanMovies
        </Link>
        <ul className="flex flex-wrap items-center gap-3 text-base text-ink md:gap-6">
          <li>
            <Link href="/recent" className="rounded-brand px-2 py-1 hover:bg-canvasSoft">
              Recent
            </Link>
          </li>
          <li>
            <Link href="/search" className="rounded-brand px-2 py-1 hover:bg-canvasSoft">
              Search
            </Link>
          </li>
          <li>
            <Link href="/home" className="rounded-brand px-2 py-1 hover:bg-canvasSoft">
              For you
            </Link>
          </li>
          {userId ? (
            <>
              <li className="flex items-center gap-2">
                <span
                  className="flex h-10 min-w-10 max-w-[5rem] shrink-0 items-center justify-center rounded-full bg-ink px-2 text-center text-xs font-bold leading-tight text-onPrimary"
                  title={`User id ${userId}`}
                >
                  {userId}
                </span>
                <button
                  type="button"
                  onClick={logout}
                  className="rounded-brand px-2 py-1 text-sm font-semibold text-ink hover:bg-canvasSoft"
                >
                  Log out
                </button>
              </li>
            </>
          ) : (
            <>
              <li>
                <Link href="/login" className="rounded-brand px-2 py-1 hover:bg-canvasSoft">
                  Log in
                </Link>
              </li>
              <li>
                <Link
                  href="/register"
                  className="rounded-brand bg-primary px-3 py-2 text-sm font-semibold text-onPrimary hover:opacity-90"
                >
                  Sign up
                </Link>
              </li>
            </>
          )}
        </ul>
      </nav>
    </header>
  );
}
