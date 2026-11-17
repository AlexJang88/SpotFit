"""Explain the planned final service using real LAN screenshots and the working template."""
from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.shapes.autoshape import CT_Shape

from build_spotfit_project_deck import change, text, rect, panel, table, GREEN, GREY, DARK, PALE, MEDIUM

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'SPOT_FIT_범죄모델_개선과정_2년3년_20261111.pptx'
CAP = ROOT / 'artifacts/presentations/final_service_capture'
OUT = ROOT / 'SPOT_FIT_최종서비스_예산추천_행정동검색_3장.pptx'


def main():
    prs = Presentation(SOURCE)
    original = list(prs.slides._sldIdLst)
    template = prs.slides[0]

    def fresh(title, subtitle):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        for shape in list(template.shapes)[:5]:
            slide.shapes._spTree.insert_element_before(deepcopy(shape._element), 'p:extLst')
        change(slide.shapes[3], '06')
        change(slide.shapes[4], title)
        rect(slide, 1.25, 3.25, 17.5, .8, PALE)
        text(slide, subtitle, 1.5, 3.4, 17, .5, 24, DARK, MEDIUM)
        return slide

    def pic(slide, name, x, y, w):
        slide.shapes.add_picture(str(CAP / name), Inches(x), Inches(y), width=Inches(w))

    def footer(slide, value):
        text(slide, value, 1.25, 10.2, 17.5, .42, 17, GREY)

    def notes(slide, value):
        frame = slide.notes_slide.notes_text_frame
        if frame is None:
            shapes = slide.notes_slide.shapes
            ident = max([shape.shape_id for shape in shapes] + [1]) + 1
            element = CT_Shape.new_textbox_sp(ident, 'Service notes', Inches(1), Inches(1), Inches(6), Inches(6))
            shapes._spTree.insert_element_before(element, 'p:extLst')
            frame = shapes[-1].text_frame
        frame.text = value + (
            '\n화면 출처: http://192.168.0.3:8080 · 실제 브라우저 캡처. '
            '현재 화면의 영역 선택, 1년/3년 표기, 3영역 종합점수는 기존 프로토타입 상태임. '
            '본 자료는 사용자가 요청한 최종 서비스 설계이며 새 기능 구현을 주장하지 않음. '
            '사이트와 학습모델은 수정하지 않음.'
        )

    s = fresh('최종 서비스 ① · 예산과 동네점수를 함께 비교',
              '서비스 목표  |  예산 안에서 아동범죄노출·교통안전 점수가 높은 동네 찾기')
    text(s, '현재 서비스 화면 · 192.168.0.3:8080', 1.25, 4.3, 9.7, .4, 19, GREEN, MEDIUM)
    pic(s, '01_home.png', 1.85, 4.75, 8.5)
    panel(s, '사용자가 입력하는 정보', '부동산 예산과 거래유형\n관심 행정동 또는 추천 지역\n확인할 예측연도: 2027 / 2028',
          11.35, 4.3, 7.4, 2.45)
    panel(s, '서비스가 함께 제공하는 정보', '예산에 맞는 추천 동네 최대 5곳\n행정동의 아파트 예측금액\n해당 자치구의 동네점수',
          11.35, 7.0, 7.4, 2.45)
    text(s, '아동범죄노출 + 교통안전: 두 영역 자동 반영', 11.5, 9.62, 7.1, .4, 20, GREEN, MEDIUM)
    footer(s, '캡처는 현재 화면입니다. 영역 선택 버튼은 제거하고 두 영역을 고정 반영하는 최종 서비스로 개편할 예정입니다.')
    notes(s, '지역점수는 아동범죄노출과 교통안전 두 영역을 자동 반영하며 사용자가 선택하거나 해제하지 않는다. '
          '육아환경은 본 추천점수 범위에 포함하지 않는다. 예산·행정동 중심 UX를 제시한다. '
          '캡처의 기존 33.3% 점수와 순위는 새 두 영역 추천 결과로 재사용하지 않는다.')

    s = fresh('최종 서비스 ② · 예산에 맞는 추천 동네 TOP 5',
              '예산 입력 → 가격 조건 필터 → 동네점수 정렬 → 추천 결과 최대 5개 제공')
    panel(s, '01  예산 입력', '거래유형: 매매 / 전세\n희망 예산: 최대 금액 입력\n예측연도: 2027 / 2028', 1.25, 4.4, 5.55, 2.35)
    panel(s, '02  후보 비교', '예측금액이 예산 이하인 동네\n소속 자치구의 점수 연결\n동네점수가 높은 순으로 정렬', 7.20, 4.4, 5.55, 2.35)
    panel(s, '03  추천 제공', '상위 동네 최대 5개 표시\n지도 위치와 추천 목록 연결\n선택하면 가격·점수 상세 조회', 13.15, 4.4, 5.60, 2.35)
    # Use the original screenshot's map as a visual anchor, cropped natively in PPT.
    image = s.shapes.add_picture(str(CAP / '03_dong.png'), Inches(1.25), Inches(7.12), Inches(6.1), Inches(2.55))
    image.crop_left = .23
    image.crop_top = .23
    image.crop_bottom = .20
    text(s, '현재 행정동 지도 · 추천 위치 표시의 기반', 1.25, 9.76, 6.2, .3, 16, GREY)
    table(s, [
        ['추천 카드 항목', '최종 화면에 표시할 내용'],
        ['동네명 · 위치', '행정동명 / 소속 자치구 / 지도 위치'],
        ['가격 · 예산', '아파트 예측금액 / 예산 충족 여부'],
        ['점수 · 추천 이유', '소속 구의 동네점수 / 범죄·교통 세부점수'],
    ], 7.75, 7.12, 11, 2.55, [3.2, 7.8], 20)
    footer(s, '추천 단위는 행정동 기준안입니다. 후보가 5개 미만이면 있는 만큼 제공하며, 가격은 거래유형별 총액으로 비교합니다.')
    notes(s, '추천 단위는 사용자의 행정동 검색 요구와 맞춘 행정동 기준안이며 개별 아파트 매물 추천으로 단정하지 않는다. '
          '총액 예산과 평당가를 직접 비교하지 않는다. 평당 추정값만 있을 경우 면적을 추가 입력받아 동일 기준 총액으로 환산해야 한다. '
          '아동범죄노출과 교통안전은 낮은 위험일수록 높은 점수가 되도록 방향과 척도를 맞춘 뒤 고정 결합한다. '
          '결합 비중과 동점 처리 규칙은 구현 전 확정 사항이다. 동일 자치구 소속 행정동은 동일한 구 점수를 갖는다. '
          '이미지 부분은 실제 행정동 지도 캡처의 일부이며 새로운 추천 결과를 그린 것이 아니다.')

    s = fresh('최종 서비스 ③ · 행정동 검색으로 가격과 점수 조회',
              '행정동의 아파트 예측금액과 소속 자치구의 동네점수를 한 화면에 표시')
    text(s, '현재 검색 화면 · “역삼” 검색 결과', 1.25, 4.3, 9.2, .4, 19, GREEN, MEDIUM)
    pic(s, '02_search.png', 1.6, 4.82, 8.25)
    panel(s, '조회 흐름 · 최종 서비스 계획', '행정동 검색 및 선택\n예: 역삼1동 → 소속 구: 강남구\n선택 지역의 가격과 구 점수를 연결', 10.85, 4.3, 7.9, 2.35)
    table(s, [
        ['정보', '표시 기준'],
        ['아파트 예측금액', '검색한 행정동 / 거래유형 / 면적'],
        ['동네점수', '해당 자치구 점수임을 명시'],
        ['세부점수', '아동범죄노출 + 교통안전'],
        ['예측 시점', '기준 데이터 연도 / 목표연도'],
    ], 10.85, 6.95, 7.9, 2.85, [2.2, 5.7], 19)
    footer(s, '같은 자치구의 행정동은 같은 구 점수를 표시합니다. 실제 아동 피해율·행정동별 안전점수로 해석하지 않습니다.')
    notes(s, '가격의 공간단위는 행정동, 범죄노출·교통안전 점수의 공간단위는 자치구임을 사용자에게 표시한다. '
          '역삼1동은 강남구에 속한다는 지역 연결 예시만 사용하며 캡처의 기존 강남구 점수나 예시 가격을 새 서비스 결과로 인용하지 않는다. '
          '현재 가격 출력은 최신 거래 특성 기준 대표가격 추정일 수 있으므로, 최종 서비스의 목표연도 가격 제공 전 모델 계약과 예측기간 확인이 필요하다. '
          '공간 조인 시 행정동 코드와 자치구 코드를 사용하고 이름만으로 동명이동을 연결하지 않는다.')

    for slide_id in original:
        prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)
    prs.save(OUT)
    assert len(Presentation(OUT).slides) == 3
    print(OUT)


if __name__ == '__main__':
    main()
