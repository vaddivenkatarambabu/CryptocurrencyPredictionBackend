import { useLocation, Link } from "react-router-dom";
import { useEffect } from "react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-6">
      <div className="glass-card rounded-xl p-10 text-center max-w-md w-full border border-border/50">
        <h1 className="text-5xl font-bold mb-4 text-gradient-neon">404</h1>

        <p className="text-muted-foreground mb-6 text-lg">
          The page you’re looking for doesn’t exist.
        </p>

        <Link
          to="/"
          className="inline-block px-6 py-2 rounded-lg bg-primary text-primary-foreground font-semibold transition hover:opacity-90"
        >
          Return to Home
        </Link>
      </div>
    </div>
  );
};

export default NotFound;