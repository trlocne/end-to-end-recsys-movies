export function Footer() {
  return (
    <footer className="mt-auto bg-ink px-6 py-12 text-base text-canvasSoft md:px-8">
      <div className="mx-auto max-w-container">
        <p className="font-semibold text-canvas">SeanMovies</p>
        <p className="mt-2 max-w-md text-canvasSoft">
          A simple demo UI for movie discovery and recommendations.
        </p>
        <p className="mt-6 text-sm text-mute">&copy; {new Date().getFullYear()} SeanMovies</p>
      </div>
    </footer>
  );
}
