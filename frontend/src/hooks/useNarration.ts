/**
 * The stage scripts of one world: everything written so far on mount, then each live script
 * as the narrator finishes it. The narrator runs behind the step loop, so a step's script can
 * arrive after the step itself, or never (narration off, a failed call).
 */

import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { StepNarration } from "../types";

export interface NarrationStore {
  /** Every script known, in step order. */
  scripts: StepNarration[];
  /** Call from the WS onFrame handler for each live narration frame. */
  onLiveScript: (script: StepNarration) => void;
}

const byStep = (a: StepNarration, b: StepNarration) => a.step - b.step;

export function useNarration(worldId: string): NarrationStore {
  const [scripts, setScripts] = useState<StepNarration[]>([]);

  const onLiveScript = useCallback((script: StepNarration) => {
    setScripts((prev) => [...prev.filter((s) => s.step !== script.step), script].sort(byStep));
  }, []);

  useEffect(() => {
    let live = true;
    api
      .listNarration(worldId)
      .then((saved) => {
        if (!live) return;
        // A live script may have landed while the list loaded: keep it.
        setScripts((prev) => {
          const steps = new Set(prev.map((s) => s.step));
          return [...prev, ...saved.filter((s) => !steps.has(s.step))].sort(byStep);
        });
      })
      .catch(() => {
        /* no scripts is a valid state: the record view still works */
      });
    return () => {
      live = false;
    };
  }, [worldId]);

  return { scripts, onLiveScript };
}
