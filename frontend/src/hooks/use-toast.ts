import * as React from "react";
import type { ToastActionElement, ToastProps } from "@/components/ui/toast";

const TOAST_LIMIT = 3;
const TOAST_REMOVE_DELAY = 4000;

type ToasterToast = ToastProps & {
  id: string;
  title?: React.ReactNode;
  description?: React.ReactNode;
  action?: ToastActionElement;
};

let count = 0;
const genId = () => (++count).toString();

type State = {
  toasts: ToasterToast[];
};

type Action =
  | { type: "ADD"; toast: ToasterToast }
  | { type: "UPDATE"; toast: Partial<ToasterToast> }
  | { type: "DISMISS"; id?: string }
  | { type: "REMOVE"; id?: string };

const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case "ADD":
      return {
        ...state,
        toasts: [action.toast, ...state.toasts].slice(0, TOAST_LIMIT),
      };

    case "UPDATE":
      return {
        ...state,
        toasts: state.toasts.map((t) =>
          t.id === action.toast.id ? { ...t, ...action.toast } : t
        ),
      };

    case "DISMISS":
      return {
        ...state,
        toasts: state.toasts.map((t) =>
          action.id === undefined || t.id === action.id
            ? { ...t, open: false }
            : t
        ),
      };

    case "REMOVE":
      return {
        ...state,
        toasts:
          action.id === undefined
            ? []
            : state.toasts.filter((t) => t.id !== action.id),
      };
  }
};

let memoryState: State = { toasts: [] };
const listeners: Array<(state: State) => void> = [];

function dispatch(action: Action) {
  memoryState = reducer(memoryState, action);
  listeners.forEach((l) => l(memoryState));
}

function scheduleRemove(id: string) {
  setTimeout(() => {
    dispatch({ type: "REMOVE", id });
  }, TOAST_REMOVE_DELAY);
}

export function toast(props: Omit<ToasterToast, "id">) {
  const id = genId();

  dispatch({
    type: "ADD",
    toast: {
      ...props,
      id,
      open: true,
      onOpenChange: (open) => {
        if (!open) {
          dispatch({ type: "DISMISS", id });
          scheduleRemove(id);
        }
      },
    },
  });

  scheduleRemove(id);

  return {
    id,
    dismiss: () => dispatch({ type: "DISMISS", id }),
    update: (newProps: Partial<ToasterToast>) =>
      dispatch({ type: "UPDATE", toast: { ...newProps, id } }),
  };
}

export function useToast() {
  const [state, setState] = React.useState<State>(memoryState);

  React.useEffect(() => {
    listeners.push(setState);
    return () => {
      const index = listeners.indexOf(setState);
      if (index > -1) listeners.splice(index, 1);
    };
  }, []); // 🔥 fixed dependency

  return {
    ...state,
    toast,
    dismiss: (id?: string) => dispatch({ type: "DISMISS", id }),
  };
}