import { AlertTriangle } from "lucide-react";

interface ErrorStateProps {
  message?: string;
}

const ErrorState = ({ message }: ErrorStateProps) => {
  const safeMessage =
    typeof message === "string" && message.length > 0
      ? message.slice(0, 200)
      : "Something went wrong while fetching data.";

  return (
    <div className="glass-card rounded-xl p-8 text-center glow-cyan animate-slide-up">
      <AlertTriangle className="mx-auto h-10 w-10 text-neon-red mb-3" />
      <p className="text-foreground font-semibold mb-1">
        Failed to load data
      </p>
      <p className="text-sm text-muted-foreground break-words">
        {safeMessage}
      </p>
    </div>
  );
};

export default ErrorState;