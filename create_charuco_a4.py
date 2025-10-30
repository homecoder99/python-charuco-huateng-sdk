# main.py
# deps: opencv-contrib-python>=4.7, pillow, reportlab, numpy
import cv2
import numpy as np
from PIL import Image, ImageDraw

def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))

def make_charuco_a4(
    squaresX=10, squaresY=12,            # 칸수 (cols, rows)
    squareLength_mm=20.0,                # 한 칸(mm)
    markerLength_mm=14.0,                # 마커(mm) = 0.7 * square 권장
    dpi=300,
    dictionary_name=cv2.aruco.DICT_5X5_1000,
    page_margin_mm=5.0,
    out_png="charuco_A4_300dpi.png",
    out_pdf="charuco_A4_300dpi.pdf",
):
    # A4(px)
    A4_W_MM, A4_H_MM = 210.0, 297.0
    page_w_px = mm_to_px(A4_W_MM, dpi)
    page_h_px = mm_to_px(A4_H_MM, dpi)

    # 보드 실제 크기(mm→px)
    board_w_mm = squaresX * squareLength_mm
    board_h_mm = squaresY * squareLength_mm
    board_w_px = mm_to_px(board_w_mm, dpi)
    board_h_px = mm_to_px(board_h_mm, dpi)

    # 여백 고려 스케일(필요 시 축소)
    margin_px = mm_to_px(page_margin_mm, dpi)
    max_w = page_w_px - 2 * margin_px
    max_h = page_h_px - 2 * margin_px
    scale = min(max_w / board_w_px, max_h / board_h_px, 1.0)
    board_w_px = int(round(board_w_px * scale))
    board_h_px = int(round(board_h_px * scale))

    # 사전 & 보드(★ 신규 API: tuple로 크기, 단위는 미터)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_name)
    board = cv2.aruco.CharucoBoard(
        (squaresX, squaresY),
        squareLength_mm / 1000.0,
        markerLength_mm / 1000.0,
        aruco_dict,
    )

    # 보드 렌더(★ 신규 API: generateImage)
    board_img = board.generateImage(
        (board_w_px, board_h_px), marginSize=0, borderBits=1
    )  # uint8

    # A4 페이지에 중앙 배치
    page = Image.new("L", (page_w_px, page_h_px), 255)
    board_pil = Image.fromarray(board_img)
    x = (page_w_px - board_w_px) // 2
    y = (page_h_px - board_h_px) // 2
    page.paste(board_pil, (x, y))

    # 가이드(외곽선/스케일바/텍스트)
    draw = ImageDraw.Draw(page)
    border_w = max(mm_to_px(0.5, dpi), 1)  # 0.5 mm 라인
    draw.rectangle([x, y, x + board_w_px - 1, y + board_h_px - 1], outline=0, width=border_w)

    bar_len_mm = 20.0
    bar_len_px = mm_to_px(bar_len_mm, dpi)
    gap_px = mm_to_px(5.0, dpi)
    bar_x0 = x
    bar_y0 = min(y + board_h_px + gap_px, page_h_px - gap_px - border_w*2)
    bar_x1 = bar_x0 + bar_len_px
    bar_y1 = bar_y0 + border_w*2
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=0)
    label = f"{int(bar_len_mm)} mm | {squaresX}×{squaresY}, square={squareLength_mm} mm, marker={markerLength_mm} mm"
    draw.text((bar_x1 + mm_to_px(3, dpi), bar_y0 - mm_to_px(1, dpi)), label, fill=0)
    draw.text((x, y - mm_to_px(6, dpi)), "Print at 100% (Actual size). Do NOT scale.", fill=0)

    # 저장: PNG
    page.save(out_png, dpi=(dpi, dpi))
    print(f"Saved: {out_png}")

    # 저장: PDF
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        c = canvas.Canvas(out_pdf, pagesize=A4)
        c.drawImage(ImageReader(page.convert("RGB")), 0, 0, width=210*mm, height=297*mm,
                    preserveAspectRatio=False, mask='auto')
        c.showPage(); c.save()
        print(f"Saved: {out_pdf}")
    except Exception as e:
        print(f"(PDF 생략) reportlab 오류: {e}")

if __name__ == "__main__":
    print("OpenCV:", cv2.__version__)
    make_charuco_a4()
