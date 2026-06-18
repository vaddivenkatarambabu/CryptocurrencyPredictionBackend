import * as React from "react";

const MOBILE_BREAKPOINT = 768;

export function useIsMobile() {
  const getInitialValue = () => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(
      `(max-width: ${MOBILE_BREAKPOINT - 1}px)`
    ).matches;
  };

  const [isMobile, setIsMobile] = React.useState<boolean>(getInitialValue);

  React.useEffect(() => {
    if (typeof window === "undefined") return;

    const mql = window.matchMedia(
      `(max-width: ${MOBILE_BREAKPOINT - 1}px)`
    );

    const handler = (event: MediaQueryListEvent) => {
      setIsMobile(event.matches);
    };

    // Set initial state
    setIsMobile(mql.matches);

    mql.addEventListener("change", handler);

    return () => {
      mql.removeEventListener("change", handler);
    };
  }, []);

  return isMobile;
}