import { Search, X } from "lucide-react";
import { useEffect, useState } from "react";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  isLoading?: boolean;
}

const SearchBar = ({ value, onChange, isLoading }: SearchBarProps) => {
  const [internalValue, setInternalValue] = useState(value);

  // Debounce effect
  useEffect(() => {
    const timer = setTimeout(() => {
      onChange(internalValue);
    }, 300);

    return () => clearTimeout(timer);
  }, [internalValue, onChange]);

  useEffect(() => {
    setInternalValue(value);
  }, [value]);

  return (
    <div className="relative w-full max-w-md">
      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />

      <input
        type="text"
        placeholder="Search coins..."
        value={internalValue}
        onChange={(e) => setInternalValue(e.target.value)}
        className="w-full pl-10 pr-10 py-2.5 rounded-lg bg-muted border border-border text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 transition-all text-sm"
      />

      {internalValue && (
        <button
          onClick={() => {
            setInternalValue("");
            onChange("");
          }}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition"
        >
          <X className="h-4 w-4" />
        </button>
      )}

      {isLoading && (
        <div className="absolute right-10 top-1/2 -translate-y-1/2">
          <div className="h-3 w-3 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      )}
    </div>
  );
};

export default SearchBar;