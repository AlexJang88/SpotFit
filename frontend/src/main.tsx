import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { HomePage } from "./views/pages/HomePage";

// index.html의 root 요소 하나를 React의 진입점으로 사용한다.
// !는 TypeScript에 요소가 있음을 알리는 단언이며 실제 존재 여부를 검사하는 코드는 아니다.
// StrictMode는 개발 중 effect의 설정/정리를 다시 실행해 정리 누락을 드러낸다.
// 따라서 지도 effect는 이벤트 해제·요청 취소·지도 제거가 가능한 형태로 작성한다.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <HomePage />
  </StrictMode>,
);
