import { useMemo, useState } from "react";
import type { ChangeEvent } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Coin } from "@/services/coingecko";

interface PredictionFormProps {
  coins: Coin[];
  selectedCoin: Coin | null;
  onCoinChange: (coin: Coin | null) => void;
  days: number;
  onDaysChange: (days: number) => void;
  onPredict: () => void;
  isLoading: boolean;
}

const MAX_DAYS = 365;
const PRESET_DAYS = [1, 7, 30, 90, 180, 365];

const PredictionForm = ({
  coins,
  selectedCoin,
  onCoinChange,
  days,
  onDaysChange,
  onPredict,
  isLoading,
}: PredictionFormProps) => {
  const [searchText, setSearchText] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);

  const filteredCoins = useMemo(() => {
    const query = searchText.trim().toLowerCase();
    if (!query) return coins.slice(0, 50);

    return coins
      .filter(
        (coin) =>
          coin.name.toLowerCase().includes(query) ||
          coin.symbol.toLowerCase().includes(query)
      )
      .slice(0, 50);
  }, [coins, searchText]);

  const handleCoinSelect = (value: string) => {
    const coin = coins.find((item) => item.id === value);
    onCoinChange(coin ?? null);
  };

  const handleDaysInput = (event: ChangeEvent<HTMLInputElement>) => {
    const value = Number(event.target.value);

    if (!Number.isInteger(value)) {
      setInputError("Enter a valid whole number.");
      return;
    }

    if (value < 1 || value > MAX_DAYS) {
      setInputError(`Days must be between 1 and ${MAX_DAYS}.`);
      return;
    }

    setInputError(null);
    onDaysChange(value);
  };

  const handlePredictClick = () => {
    if (!selectedCoin || inputError || days < 1 || days > MAX_DAYS) return;
    onPredict();
  };

  return (
    <div className="glass-card rounded-xl p-6 space-y-5 animate-slide-up">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="space-y-2">
          <Label className="text-foreground text-sm font-semibold">
            Select Cryptocurrency
          </Label>

          <Input
            placeholder="Search coin name or symbol..."
            value={searchText}
            onChange={(event) => setSearchText(event.target.value)}
            className="bg-muted/30 border-border/60 text-foreground mb-2"
          />

          <Select value={selectedCoin?.id ?? ""} onValueChange={handleCoinSelect}>
            <SelectTrigger className="bg-muted/30 border-border/60">
              <SelectValue placeholder="Choose a coin..." />
            </SelectTrigger>

            <SelectContent className="bg-card border-border max-h-64">
              {filteredCoins.map((coin) => (
                <SelectItem key={coin.id} value={coin.id}>
                  <div className="flex items-center gap-2">
                    <img
                      src={coin.image}
                      alt={coin.name}
                      className="w-5 h-5 rounded-full"
                      onError={(event) => {
                        event.currentTarget.src = "https://via.placeholder.com/20";
                      }}
                    />
                    <span>{coin.name}</span>
                    <span className="text-muted-foreground uppercase text-xs">
                      {coin.symbol}
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {selectedCoin && (
            <div className="flex items-center gap-2 mt-2 px-2 py-1.5 rounded-lg bg-primary/10 border border-primary/20">
              <img
                src={selectedCoin.image}
                alt={selectedCoin.name}
                className="w-5 h-5 rounded-full"
              />
              <span className="text-primary text-sm font-semibold">
                {selectedCoin.name}
              </span>
              <span className="text-muted-foreground text-xs uppercase">
                {selectedCoin.symbol}
              </span>
              <span className="ml-auto text-foreground text-sm font-mono">
                ${selectedCoin.current_price.toLocaleString()}
              </span>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <Label className="text-foreground text-sm font-semibold">
            Prediction Days (1-365)
          </Label>

          <Input
            type="number"
            min={1}
            max={MAX_DAYS}
            value={days}
            onChange={handleDaysInput}
            className="bg-muted/30 border-border/60 font-mono"
          />

          {inputError && <p className="text-neon-red text-xs">{inputError}</p>}

          <div className="flex flex-wrap gap-2 mt-2">
            {PRESET_DAYS.map((day) => (
              <button
                key={day}
                type="button"
                onClick={() => {
                  setInputError(null);
                  onDaysChange(day);
                }}
                className={`px-3 py-1 rounded-md text-xs font-semibold border transition-colors ${
                  days === day
                    ? "bg-primary/20 border-primary text-primary"
                    : "bg-muted/30 border-border/50 text-muted-foreground hover:text-foreground"
                }`}
              >
                {day}d
              </button>
            ))}
          </div>
        </div>
      </div>

      <Button
        onClick={handlePredictClick}
        disabled={!selectedCoin || isLoading || !!inputError}
        className="w-full font-bold bg-primary text-primary-foreground hover:bg-primary/80 disabled:opacity-50"
        size="lg"
      >
        {isLoading ? "Analyzing Market..." : "Predict Price"}
      </Button>
    </div>
  );
};

export default PredictionForm;
