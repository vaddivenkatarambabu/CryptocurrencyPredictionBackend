import { AlertTriangle, RefreshCcw } from "lucide-react";

interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

const ErrorState = ({ message, onRetry }: ErrorStateProps) => {
  const safeMessage =
    message && message.length < 200
      ? message
      : "Something went wrong while fetching data.";

  return (
    <div className="glass-card rounded-xl p-8 text-center glow-cyan animate-slide-up">
      <AlertTriangle className="mx-auto h-10 w-10 text-neon-red mb-3" />

      <p className="text-foreground font-semibold mb-1">
        Failed to load data
      </p>

      <p className="text-sm text-muted-foreground mb-4">
        {safeMessage}
      </p>

      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition"
        >
          <RefreshCcw className="h-4 w-4" />
          Retry
        </button>
      )}
    </div>
  );
};

export default ErrorState;