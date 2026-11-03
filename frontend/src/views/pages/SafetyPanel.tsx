import type { CrimeSafetyResponse } from "./crimeSafetyApi";
import "./safety-panel.css";

export function SafetyPanel({ data, loading, error }: {
  data: CrimeSafetyResponse | null;
  loading: boolean;
  error: string;
}) {
  return <section className="safety-panel" aria-label="서울 자치구 지역안전지수 예측">
    <h2>지역안전지수 예상등급</h2>
    {loading && <p role="status">안전등급을 불러오는 중입니다.</p>}
    {error && <p className="safety-error" role="alert">{error}</p>}
    {data && <>
      <div className="safety-meta">
        <span>입력 기준 <strong>{data.data_year}년</strong></span>
        <span>예측 대상 <strong>{data.prediction_year}년</strong></span>
      </div>
    </>}
  </section>;
}

