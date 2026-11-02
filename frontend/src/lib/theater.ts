/**
 * Framework-free helpers for the theater view: how acts and scenes are named.
 */

const DIGITS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];

/** 1 → 一, 10 → 十, 12 → 十二, 20 → 二十, 105 → 一百零五. Non-positive numbers read as digits. */
export function chineseNumeral(n: number): string {
  if (!Number.isInteger(n) || n <= 0) return String(n);
  if (n < 10) return DIGITS[n];
  if (n < 20) return `十${n % 10 ? DIGITS[n % 10] : ""}`;
  if (n < 100) return `${DIGITS[Math.floor(n / 10)]}十${n % 10 ? DIGITS[n % 10] : ""}`;
  const rest = n % 100;
  const head = `${DIGITS[Math.floor(n / 100)]}百`;
  if (rest === 0) return head;
  return rest < 10 ? `${head}零${DIGITS[rest]}` : `${head}${rest < 20 ? "一" : ""}${chineseNumeral(rest)}`;
}

export const actTitle = (act: number) => `第${chineseNumeral(act)}幕`;
export const sceneTitle = (scene: number) => `第${chineseNumeral(scene)}场`;
