from copy import deepcopy
from pathlib import Path
import re

from pptx import Presentation
from pptx.util import Inches


SOURCE = Path(r"C:\Users\M\Desktop\발표자료\SPOT_FIT_4차중간발표.pptx")
OUTPUT = Path(r"C:\Users\M\Desktop\JUS\SpotFit\artifacts\SPOT_FIT_4차중간발표_페이지번호추가.pptx")
PAGE_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")


def is_page_number(shape) -> bool:
    return bool(getattr(shape, "has_text_frame", False) and PAGE_RE.fullmatch(shape.text or ""))


def copy_footer(slide, template, page: int):
    element = deepcopy(template._element)
    slide.shapes._spTree.insert_element_before(element, "p:extLst")
    footer = next(shape for shape in slide.shapes if shape._element is element)
    footer.text_frame.paragraphs[0].runs[0].text = f"- {page} -"
    return footer


def fit_bottom_tables(slide, footer_top):
    """Keep the existing table layout but reserve a clear footer band."""
    safe_bottom = footer_top - Inches(0.14)
    for shape in slide.shapes:
        if not getattr(shape, "has_table", False) or shape.top + shape.height <= safe_bottom:
            continue
        old_top = shape.top
        old_height = shape.height
        new_top = max(Inches(0.05), old_top - Inches(0.06))
        new_height = safe_bottom - new_top
        ratio = new_height / old_height
        shape.top = new_top
        shape.height = new_height
        for row in shape.table.rows:
            row.height = int(row.height * ratio)


def main():
    prs = Presentation(SOURCE)
    # Slide 3 contains the deck's established footer formatting.
    template = next(shape for shape in prs.slides[2].shapes if is_page_number(shape))
    footer_top = template.top

    # Cover, contents and the final thank-you slide intentionally remain unnumbered.
    for page, slide in enumerate(list(prs.slides)[2:16], start=1):
        existing = next((shape for shape in slide.shapes if is_page_number(shape)), None)
        if existing is None:
            fit_bottom_tables(slide, footer_top)
            existing = copy_footer(slide, template, page)
        else:
            existing.text_frame.paragraphs[0].runs[0].text = f"- {page} -"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
