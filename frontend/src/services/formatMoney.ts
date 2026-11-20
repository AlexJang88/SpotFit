/** Format an amount stored in 만원, rounding once before splitting into 억/만원. */
export function formatMoney(amount: number): string {
  if (!Number.isFinite(amount)) return "가격 정보 없음";
  const rounded = Math.round(Math.abs(amount));
  const sign = amount < 0 && rounded !== 0 ? "−" : "";
  const eok = Math.floor(rounded / 10000);
  const remainder = rounded % 10000;
  if (!eok) return `${sign}${remainder.toLocaleString("ko-KR")}만원`;
  return remainder
    ? `${sign}${eok.toLocaleString("ko-KR")}억 ${remainder.toLocaleString("ko-KR")}만원`
    : `${sign}${eok.toLocaleString("ko-KR")}억원`;
}
