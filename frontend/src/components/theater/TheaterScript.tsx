/**
 * The theater view: the stage narrator's scripts read as a play, in step order.
 *
 * Structure (acts, scenes, continuation) comes from the backend; this only lays it out. Lines
 * and asides are the characters' own words, quoted verbatim by the narrator, so they print with
 * the speaker's identity color like the record view's dialogue.
 */

import { useEffect, useRef } from "react";

import { actTitle, sceneTitle } from "../../lib/theater";
import type { NarrationBeat, NarrationScene, StepNarration } from "../../types";

function Beat({ beat, colorOf }: { beat: NarrationBeat; colorOf: (agentId: string) => string | undefined }) {
  const speaker = (
    <span className="font-semibold" style={{ color: colorOf(beat.speaker_id) }}>
      {beat.speaker || "某人"}
    </span>
  );
  if (beat.kind === "line") {
    return (
      <p className="text-[14px] leading-relaxed text-slate-100">
        {speaker}：「{beat.text}」
      </p>
    );
  }
  if (beat.kind === "aside") {
    return (
      <p className="text-[13px] italic leading-relaxed text-slate-400">
        {speaker}
        <span className="not-italic text-slate-600">（独白）</span>
        {beat.text}
      </p>
    );
  }
  return <p className="text-[13px] leading-relaxed text-slate-300">{beat.text}</p>;
}

function Scene({
  scene,
  timeLabel,
  colorOf,
  onSpotlight,
  lit,
}: {
  scene: NarrationScene;
  timeLabel: string;
  colorOf: (agentId: string) => string | undefined;
  onSpotlight: (scene: NarrationScene) => void;
  lit: boolean;
}) {
  return (
    <section className={`rounded-xl border px-3 py-2.5 ${lit ? "border-amber-500/50 bg-amber-950/10" : "border-slate-800/70 bg-slate-950/40"}`}>
      {/* The heading spotlights the scene's place on the map. */}
      <button
        className="w-full text-left flex items-baseline gap-2 group"
        onClick={() => onSpotlight(scene)}
        title="在地图上追光这一场"
      >
        <span className="text-xs font-semibold text-amber-300/90">
          {sceneTitle(scene.scene)}
          {scene.continues && <span className="text-amber-300/50">（续）</span>}
        </span>
        <span className="text-xs text-slate-300 group-hover:text-white">{scene.location_name}</span>
        <span className="ml-auto text-[10px] text-slate-500">{timeLabel}</span>
      </button>
      {scene.stage_direction && (
        <p className="mt-1.5 text-[12px] italic leading-relaxed text-slate-500">（{scene.stage_direction}）</p>
      )}
      <div className="mt-1.5 space-y-1.5">
        {scene.beats.map((beat, i) => (
          <Beat key={i} beat={beat} colorOf={colorOf} />
        ))}
      </div>
    </section>
  );
}

export default function TheaterScript({
  scripts,
  colorOf,
  onSpotlight,
  litPlace,
  follow,
}: {
  /** Scripts to show, in step order (already cut to the current point by the caller). */
  scripts: StepNarration[];
  colorOf: (agentId: string) => string | undefined;
  onSpotlight: (scene: NarrationScene) => void;
  /** The place currently spotlit on the map, so its scene can be marked. */
  litPlace: string | null;
  /** Keep the newest script in view as scripts arrive. */
  follow: boolean;
}) {
  const end = useRef<HTMLDivElement | null>(null);
  const last = scripts[scripts.length - 1]?.step;
  useEffect(() => {
    if (follow) end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [last, follow]);

  if (!scripts.length) {
    return (
      <p className="text-slate-500 text-sm">
        还没有剧本。说书人在每一步结束后落笔，比叙事流稍慢一拍；若这个世界关闭了说书人，这里会一直为空。
      </p>
    );
  }
  return (
    <div className="space-y-3">
      {scripts.map((script, i) => (
        <div key={script.step} className="space-y-2">
          {script.opens_act && (
            <header className="pt-2">
              <h4 className="text-center text-sm font-semibold tracking-[0.3em] text-amber-200/90">
                {actTitle(script.act)}
              </h4>
              {/* The synopsis written before this act opened: what the audience carries in. */}
              {i > 0 && scripts[i - 1].memo && (
                <p className="mt-1 text-center text-[11px] leading-relaxed text-slate-500">
                  前情：{scripts[i - 1].memo}
                </p>
              )}
            </header>
          )}
          {script.scenes.map((scene) => (
            <Scene
              key={`${script.step}-${scene.location_id}`}
              scene={scene}
              timeLabel={script.time_label}
              colorOf={colorOf}
              onSpotlight={onSpotlight}
              lit={litPlace === scene.location_id}
            />
          ))}
        </div>
      ))}
      <div ref={end} />
    </div>
  );
}
