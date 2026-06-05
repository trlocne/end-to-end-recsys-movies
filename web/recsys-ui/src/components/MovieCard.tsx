import Link from "next/link";
import type { MovieMeta } from "@/lib/rec-types";
import type { RecItem } from "@/lib/rec-types";

type Props = {
  item: RecItem;
  meta?: MovieMeta;
  /** True while /v1/movies/batch is in flight for this result set */
  metaPending?: boolean;
  /** Shown below the card when set (e.g. user rating on recent list) */
  secondaryLabel?: string;
};

function scoreLabel(item: RecItem): string {
  if (item.rerank_score != null && !Number.isNaN(Number(item.rerank_score))) {
    return `Score ${Number(item.rerank_score).toFixed(2)}`;
  }
  if (item.ann_score != null) {
    return `ANN ${Number(item.ann_score).toFixed(3)}`;
  }
  return "";
}

export function MovieCard({
  item,
  meta,
  metaPending = false,
  secondaryLabel,
}: Props) {
  const hasTitle = Boolean(meta?.title?.trim());
  const titleLine = hasTitle
    ? meta!.title!.trim()
    : metaPending
      ? "Loading title…"
      : `Movie #${item.item_id}`;

  const sub = hasTitle
    ? [meta?.genres, meta?.tag].filter(Boolean).join(" · ") || item.source || ""
    : metaPending
      ? ""
      : item.source || "";

  return (
    <Link
      href={`/movie/${item.item_id}`}
      className="group block rounded-brand border border-mute/40 bg-canvasSoft p-5 transition-shadow hover:border-ink/30 hover:shadow-md"
    >
      <p className="text-xs font-semibold uppercase tracking-wider text-bodyMid">Movie id</p>
      <p className="mt-1 font-mono text-base font-semibold text-ink">{item.item_id}</p>
      <h3
        className={`mt-3 font-display text-lg font-semibold line-clamp-2 ${
          metaPending && !hasTitle
            ? "text-bodyMid italic"
            : "text-ink group-hover:text-primary"
        }`}
      >
        {titleLine}
      </h3>
      {sub ? <p className="mt-2 line-clamp-2 text-sm text-body">{sub}</p> : null}
      {(secondaryLabel?.trim() || scoreLabel(item)) && (
        <p className="mt-3 text-sm font-medium text-inkSoft">
          {secondaryLabel?.trim() || scoreLabel(item)}
        </p>
      )}
    </Link>
  );
}
