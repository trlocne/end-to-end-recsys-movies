import Link from "next/link";

export default function HomePage() {
  return (
    <div className="space-y-16">
      <section className="rounded-brand bg-canvas px-4 py-16 md:py-20">
        <p className="mb-3 text-sm font-medium uppercase tracking-widest text-ink">
          Welcome
        </p>
        <h1 className="max-w-2xl font-display text-4xl font-medium leading-tight text-ink md:text-6xl md:leading-none">
          Find your next watch with SeanMovies
        </h1>
        <p className="mt-6 max-w-xl text-xl leading-relaxed text-body">
          Sign in with your user id, search the catalog, and get personalized picks
          from the same API that powers our recommendations backend.
        </p>
        <div className="mt-10 flex flex-wrap gap-4">
          <Link href="/register" className="btn-primary inline-block text-center">
            Create account
          </Link>
          <Link href="/search" className="btn-secondary inline-block text-center">
            Search movies
          </Link>
        </div>
      </section>

      <section className="rounded-brand bg-canvasSoft px-4 py-14 md:py-16">
        <h2 className="font-display text-3xl font-medium text-ink md:text-5xl">
          How it works
        </h2>
        <ul className="mt-8 grid gap-6 md:grid-cols-3">
          <li className="card-content">
            <span className="text-sm font-semibold uppercase tracking-wide text-bodyMid">
              Step 1
            </span>
            <p className="mt-2 font-semibold text-ink">Register your numeric user id</p>
            <p className="mt-2 text-body">No password yet — same as the demo API.</p>
          </li>
          <li className="card-content">
            <span className="text-sm font-semibold uppercase tracking-wide text-bodyMid">
              Step 2
            </span>
            <p className="mt-2 font-semibold text-ink">Search or open For you</p>
            <p className="mt-2 text-body">Queries go through this site, then to the internal API.</p>
          </li>
          <li className="card-content">
            <span className="text-sm font-semibold uppercase tracking-wide text-bodyMid">
              Step 3
            </span>
            <p className="mt-2 font-semibold text-ink">Tune results with more interactions</p>
            <p className="mt-2 text-body">The more you use the platform API, the better it gets.</p>
          </li>
        </ul>
      </section>
    </div>
  );
}
