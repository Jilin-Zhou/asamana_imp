/**
 * A check in words, for the observer: the three enums' labels and the roll's arithmetic.
 * Total tables, like the contract copies they key on.
 */

import type { CheckView } from "../types";
import {
  CheckTier, type CheckTierValue, Difficulty, type DifficultyValue, Edge, type EdgeValue,
} from "./contract";

export const TIER_LABEL: Record<CheckTierValue, string> = {
  [CheckTier.critical_success]: "大成功",
  [CheckTier.success]:          "成功",
  [CheckTier.costly_success]:   "有代价的成功",
  [CheckTier.failure]:          "失败",
  [CheckTier.critical_failure]: "大失败",
};

export const DIFFICULTY_LABEL: Record<DifficultyValue, string> = {
  [Difficulty.easy]:      "容易",
  [Difficulty.normal]:    "普通",
  [Difficulty.hard]:      "困难",
  [Difficulty.very_hard]: "极难",
};

export const EDGE_LABEL: Record<EdgeValue, string> = {
  [Edge.much_worse]:  "明显劣势",
  [Edge.worse]:       "略处劣势",
  [Edge.even]:        "势均力敌",
  [Edge.better]:      "略占优势",
  [Edge.much_better]: "明显优势",
};

/** The label for a wire value; an unknown value reads as itself rather than vanishing. */
function label<K extends string>(table: Record<K, string>, value: string): string {
  return (table as Record<string, string>)[value] ?? value;
}

export const tierLabel = (check: CheckView): string => label(TIER_LABEL, check.tier);

/** What the roll was against: difficulty, then the edge (or that luck alone decided). */
export function checkStakes(check: CheckView): string {
  const odds = label(DIFFICULTY_LABEL, check.difficulty);
  return check.pure_chance ? `${odds} · 全凭运气` : `${odds} · ${label(EDGE_LABEL, check.edge)}`;
}

/** The roll itself: "4+3+3 +2 = 12 / 11". A zero modifier is left out, a negative one signed. */
export function checkArithmetic(check: CheckView): string {
  const dice = check.dice.join("+");
  const mod = check.modifier === 0 ? "" : ` ${check.modifier > 0 ? "+" : "−"}${Math.abs(check.modifier)}`;
  return `${dice}${mod} = ${check.total} / ${check.dc}`;
}
