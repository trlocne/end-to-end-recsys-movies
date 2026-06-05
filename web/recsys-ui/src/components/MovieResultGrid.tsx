"use client";

import { MovieCard } from "@/components/MovieCard";
import type { MovieMeta, RecItem } from "@/lib/rec-types";

type Props = {
  items: RecItem[];
  metaById: Map<number, MovieMeta>;
  metaPending?: boolean;
  title?: string;
};

export function MovieResultGrid({
  items,
  metaById,
  metaPending = false,
  title = "Results",
}: Props) {
  if (items.length === 0) return null;

  return (
    <section className="mt-10">
      {title ? (
        <h2 className="font-display text-2xl font-medium text-ink">{title}</h2>
      ) : null}
      {metaPending ? (
        <p className="mt-2 text-sm text-bodyMid">Loading titles from catalog…</p>
      ) : null}
      <div>
        <div
          className={`grid gap-4 sm:grid-cols-2 lg:grid-cols-3 ${title || metaPending ? "mt-6" : "mt-0"}`}
        >
          {items.map((item) => (
            <MovieCard
              key={item.item_id}
              item={item}
              meta={metaById.get(item.item_id)}
              metaPending={metaPending}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
