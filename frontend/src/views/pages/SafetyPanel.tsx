import type { CrimeSafetyResponse } from "./crimeSafetyApi";
import "./safety-panel.css";

export function SafetyPanel({ data: _data, loading, error }: {
  data: CrimeSafetyResponse | null;
  loading: boolean;
  error: string;
}) {
  return <section className="safety-panel" aria-label="지역점수 등급 색상 안내">
    {loading && <p role="status">안전등급을 불러오는 중입니다.</p>}
    {error && <p className="safety-error" role="alert">{error}</p>}
    <h3>등급 색상 안내</h3>
    <div className="grade-guide">
      <span><i className="grade grade-a">A</i>상위 20%</span>
      <span><i className="grade grade-b">B</i>상위 20~40%</span>
      <span><i className="grade grade-c">C</i>상위 40~60%</span>
      <span><i className="grade grade-d">D</i>상위 60~80%</span>
      <span><i className="grade grade-e">E</i>하위 20%</span>
    </div>
  </section>;
}

