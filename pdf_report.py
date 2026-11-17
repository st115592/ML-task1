"""Typeset the generated Servo Markdown report on A4 with ReportLab."""

from collections.abc import Iterator
from io import BytesIO
import os
from pathlib import Path
import re
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.doctemplate import IndexingFlowable
from reportlab.platypus.tableofcontents import TableOfContents

LEFT, RIGHT, VERTICAL = 30 * mm, 15 * mm, 20 * mm
TEXT_WIDTH = A4[0] - LEFT - RIGHT
GOST_URL = "https://protect.gost.ru/gost/details/7d280e43-7036-4a69-8e6e-d15867028343"
INLINE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]+)\]\((https?://[^)]+)\)")


def register_fonts() -> None:
    """Load the local copies of Times New Roman without downloading fonts."""
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    folder = Path(os.environ.get("SERVO_FONT_DIR", windows_fonts))
    for name, filename in (("Times", "times.ttf"), ("Times-Bold", "timesbd.ttf")):
        font = folder / filename
        if not font.is_file():
            raise FileNotFoundError(
                f"Нет шрифта {font}. Укажите папку через SERVO_FONT_DIR."
            )
        pdfmetrics.registerFont(TTFont(name, str(font)))
    pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold")


def styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        "Body",
        fontName="Times",
        fontSize=12,
        leading=18,
        alignment=TA_JUSTIFY,
        firstLineIndent=12.5 * mm,
        spaceAfter=5,
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        "body": body,
        "section": ParagraphStyle(
            "Section",
            parent=body,
            fontName="Times-Bold",
            spaceAfter=10,
            keepWithNext=True,
        ),
        "structural": ParagraphStyle(
            "Structural",
            parent=body,
            fontName="Times-Bold",
            firstLineIndent=0,
            alignment=TA_CENTER,
            spaceAfter=12,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=body,
            fontName="Times-Bold",
            spaceBefore=8,
            keepWithNext=True,
        ),
        "center": ParagraphStyle(
            "Center", parent=body, firstLineIndent=0, alignment=TA_CENTER
        ),
        "title": ParagraphStyle(
            "Title",
            parent=body,
            firstLineIndent=0,
            alignment=TA_CENTER,
            fontName="Times-Bold",
            fontSize=16,
            leading=24,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=body,
            firstLineIndent=0,
            alignment=TA_CENTER,
            fontSize=11,
            leading=14,
        ),
        "table_title": ParagraphStyle(
            "TableTitle",
            parent=body,
            firstLineIndent=0,
            alignment=TA_LEFT,
            fontSize=11,
            leading=14,
            keepWithNext=True,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=body,
            firstLineIndent=0,
            alignment=TA_LEFT,
            fontSize=10,
            leading=12,
            spaceAfter=0,
        ),
        "number": ParagraphStyle(
            "Number",
            parent=body,
            firstLineIndent=0,
            alignment=TA_CENTER,
            fontSize=10,
            leading=12,
            spaceAfter=0,
        ),
        "toc": ParagraphStyle("Toc", parent=body, firstLineIndent=0, leading=23),
    }


def markup(text: str) -> str:
    """Convert the inline notation used by the generated Markdown to PDF text."""
    parts = []
    start = 0
    for match in INLINE.finditer(text):
        parts.append(escape(text[start : match.start()]))
        strong, code, label, url = match.groups()
        if strong is not None:
            parts.append(f"<b>{escape(strong)}</b>")
        elif code is not None:
            parts.append(escape(code))
        else:
            parts.append(f"<link href={quoteattr(url)}>{escape(label)}</link>")
        start = match.end()
    parts.append(escape(text[start:]))
    return "".join(parts)


def blocks(markdown: str) -> Iterator[list[str]]:
    """Split the source report at blank lines, retaining table and list rows."""
    for block in re.split(r"\n\s*\n", markdown.strip()):
        yield block.splitlines()


class Document(BaseDocTemplate):
    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name in {
            "Section",
            "Structural",
        }:
            title = flowable.getPlainText()
            if title not in {"РЕФЕРАТ", "СОДЕРЖАНИЕ"}:
                self.notify("TOCEntry", (0, title, self.page))


class PageExtent(IndexingFlowable):
    """Insert the page count after the contents and pagination stabilize."""

    def __init__(self, style: ParagraphStyle, images: int, tables: int) -> None:
        super().__init__()
        self.style = style
        self.images = images
        self.tables = tables
        self.pages = self.previous_pages = 0

    def beforeBuild(self) -> None:
        self.previous_pages, self.pages = self.pages, 0

    def notify(self, kind: str, stuff: Any) -> None:
        if kind == "PageCount":
            self.pages = stuff

    def isSatisfied(self) -> bool:
        return self.pages == self.previous_pages

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:
        self.paragraph = Paragraph(
            f"Отчёт {self.previous_pages} с., {self.images} рис., {self.tables} табл., 2 источника, 1 прил.",
            self.style,
        )
        return self.paragraph.wrap(availWidth, availHeight)

    def draw(self) -> None:
        self.paragraph.drawOn(self.canv, 0, 0)


def footer(canvas: Canvas, doc: Document) -> None:
    doc.notify("PageCount", doc.page)
    if doc.page > 1:
        canvas.setFont("Times", 12)
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))


def invariant_canvas(*args: Any, **kwargs: Any) -> Canvas:
    """Fix PDF creation metadata so subsequent renders are byte-reproducible."""
    kwargs["invariant"] = True
    return Canvas(*args, **kwargs)


def fitted_image(source: Path | BytesIO, max_width: float, max_height: float) -> Image:
    stream = str(source) if isinstance(source, Path) else source
    width, height = ImageReader(stream).getSize()
    scale = min(max_width / width, max_height / height)
    if isinstance(stream, BytesIO):
        stream.seek(0)
    return Image(stream, width=width * scale, height=height * scale, hAlign="CENTER")


def read_table(lines: list[str], style: dict[str, ParagraphStyle]) -> LongTable:
    rows = [
        [cell.strip() for cell in row.strip().strip("|").split("|")]
        for row in [lines[0], *lines[2:]]
    ]
    columns = len(rows[0])
    # Give descriptive columns more room than the numeric results.
    weights = {
        3: [0.38, 0.31, 0.31],
        4: [0.42, 0.20, 0.20, 0.18],
        5: [0.32, 0.16, 0.18, 0.18, 0.16],
        6: [0.35, 0.16, 0.13, 0.12, 0.12, 0.12],
    }[columns]
    if rows[0][0] == "Оценка":
        weights = [0.46, 0.18, 0.18, 0.18]
    cells = [
        [
            Paragraph(
                f"<b>{markup(cell)}</b>" if i == 0 else markup(cell),
                style["cell" if j == 0 else "number"],
            )
            for j, cell in enumerate(row)
        ]
        for i, row in enumerate(rows)
    ]
    table = LongTable(
        cells,
        colWidths=[w * TEXT_WIDTH for w in weights],
        repeatRows=1,
        hAlign="LEFT",
        spaceAfter=8,
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


TABLE_TITLES = (
    "Входные величины и время нарастания",
    "Ошибка внешней перекрёстной проверки",
    "Оценки итоговой модели",
    "Разброс оценок коэффициентов",
    "Проверка переноса по усилению",
    "Метрики при обучении на исходном времени",
    "Метрики при обучении на логарифме времени",
    "Поправки для разных двигателей и передач",
)

FIGURE_REFERENCES = (
    "На рисунке 1 сопоставлены ошибки при разных степенях полинома и величинах штрафа.",
    "На рисунке 2 прогнозы сопоставлены с исходными значениями времени.",
    "На рисунке 3 показаны остатки в зависимости от прогноза и усилений.",
    "Карта на рисунке 4 показывает среднюю ошибку для каждой наблюдаемой пары усилений.",
    "На рисунке 5 показаны диапазоны повторных оценок коэффициентов.",
)

TEXT_REFERENCES = (
    ("набор Servo из репозитория UCI:", "набор Servo из репозитория UCI [1]. В нём"),
    ("в систему СИ.", "в систему СИ. Обозначения и диапазоны приведены в таблице 1."),
    ("В таблице приведены ошибки", "В таблице 2 приведены ошибки"),
    ("Получены следующие оценки:", "Оценки приведены в таблице 3."),
    ("Таблица показывает", "В таблице 4 приведены"),
    (
        "Для сравнения приведена основная модель,",
        "В таблице 5 приведена и основная модель,",
    ),
    ("В следующих таблицах приведены", "В таблицах А.1 и А.2 приведены"),
    (
        "она имеет вид a + bp + cv:",
        "она имеет вид a + bp + cv. Коэффициенты приведены в таблице А.3.",
    ),
)


def report_body(
    root: Path, source: list[list[str]], style: dict[str, ParagraphStyle]
) -> list[Flowable]:
    """Convert only the block types used in the report; retain all its content."""
    story: list[Flowable] = []
    figures = tables = equations = appendix_tables = subsection = 0
    appendix = False
    for lines in source:
        line = lines[0]
        if line.startswith("## "):
            heading = line[3:]
            story.append(PageBreak())
            if heading == "8. Выводы":
                story.append(Paragraph("ЗАКЛЮЧЕНИЕ", style["structural"]))
            elif heading.startswith("Приложение."):
                appendix = True
                story.append(Paragraph("ПРИЛОЖЕНИЕ А", style["structural"]))
                story.append(
                    Paragraph(escape(heading.split(". ", 1)[1]), style["center"])
                )
            else:
                story.append(
                    Paragraph(
                        markup(re.sub(r"^(\d+)\. ", r"\1 ", heading)), style["section"]
                    )
                )
        elif line.startswith("### "):
            subsection += 1
            if subsection > 1:
                story.append(PageBreak())
            story.append(
                Paragraph(f"А.{subsection} {markup(line[4:])}", style["subsection"])
            )
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", line)
            if match is None:
                raise ValueError(f"Некорректная ссылка на рисунок: {line}")
            title, path = match.groups()
            figures += 1
            story.append(
                KeepTogether(
                    [
                        Paragraph(FIGURE_REFERENCES[figures - 1], style["body"]),
                        fitted_image(root / path, TEXT_WIDTH, 105 * mm),
                        Paragraph(
                            f"Рисунок {figures} - {markup(title)}", style["caption"]
                        ),
                    ]
                )
            )
        elif line.startswith("|"):
            title = TABLE_TITLES[tables]
            tables += 1
            if appendix:
                appendix_tables += 1
            label = f"А.{appendix_tables}" if appendix else str(tables)
            story.append(Paragraph(f"Таблица {label} - {title}", style["table_title"]))
            story.append(read_table(lines, style))
        elif line == "$$":
            equations += 1
            buffer = BytesIO()
            formula = "".join(lines[1:-1]).rstrip(".")
            math_to_image(
                f"${formula}$",
                buffer,
                dpi=300,
                format="png",
                prop=FontProperties(size=13, math_fontfamily="stix"),
            )
            buffer.seek(0)
            label = f"А.{equations - 1}" if appendix else str(equations)
            row = Table(
                [
                    [
                        fitted_image(buffer, TEXT_WIDTH - 45, 10 * mm),
                        Paragraph(f"({label})", style["center"]),
                    ]
                ],
                colWidths=[TEXT_WIDTH - 40, 40],
            )
            row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            story.append(row)
        elif re.match(r"^(\* |\d+\. )", line):
            items = []
            for part in lines:
                entry = re.match(r"^(\*|\d+\.) (.*)$", part)
                if entry:
                    marker, content = entry.groups()
                    bullet = "-" if marker == "*" else f"{marker[:-1]})"
                    items.append(f"{bullet} {content}")
                else:
                    items[-1] += " " + part.strip()
            story.extend(Paragraph(markup(item), style["body"]) for item in items)
        elif line.startswith("**Источник данных:"):
            continue
        else:
            paragraph = " ".join(lines)
            for old, new in TEXT_REFERENCES:
                paragraph = paragraph.replace(old, new)
            story.append(Paragraph(markup(paragraph), style["body"]))
    return story


def assemble_pdf(root: Path) -> None:
    """Build the PDF atomically from the just-generated report and figures."""
    register_fonts()
    style = styles()
    source = list(blocks((root / "REPORT.md").read_text(encoding="utf-8")))
    title = source[0][0].removeprefix("# ")
    intro = source[2]
    conclusion_start = next(
        i for i, group in enumerate(source) if group[0] == "## 8. Выводы"
    )
    conclusion = " ".join(source[conclusion_start + 1])
    summary = " ".join(re.split(r"(?<=\.) ", conclusion)[:2])
    limitation = " ".join(
        re.split(r"(?<=\.) ", " ".join(source[conclusion_start + 2]))[:2]
    )
    body_source = [
        *source[1:2],
        *source[3:],
    ]  # the opening paragraph becomes the introduction
    figures = sum(group[0].startswith("![") for group in body_source)
    tables = sum(group[0].startswith("|") for group in body_source)
    toc = TableOfContents()
    toc.levelStyles = [style["toc"]]
    toc.dotsMinLevel = 0
    story = [
        Spacer(1, 35 * mm),
        Paragraph("ОТЧЁТ ПО УЧЕБНОЙ РАБОТЕ", style["center"]),
        Paragraph("Курс «Машинное обучение в механике»", style["center"]),
        Spacer(1, 38 * mm),
        Paragraph(markup(title), style["title"]),
        Spacer(1, 130 * mm),
        Paragraph("2026", style["center"]),
        PageBreak(),
        Paragraph("РЕФЕРАТ", style["structural"]),
        PageExtent(style["body"], figures, tables),
        Paragraph(
            "СЕРВОПРИВОД, ВРЕМЯ НАРАСТАНИЯ, ЛИНЕЙНАЯ РЕГРЕССИЯ, РЕГУЛЯРИЗАЦИЯ, ПЕРЕКРЁСТНАЯ ПРОВЕРКА",
            style["body"],
        ),
        Paragraph(
            "Время нарастания сервопривода предсказано по двум усилениям, типу двигателя и типу передачи. Модели сопоставлены на повторных разбиениях и отложенных данных. В отчёте приведены оценки точности, результаты проверки коэффициентов и границы применимости модели.",
            style["body"],
        ),
        Paragraph(markup(summary + " " + limitation), style["body"]),
        Paragraph("При оформлении использован ГОСТ 7.32-2017 [2].", style["body"]),
        PageBreak(),
        Paragraph("СОДЕРЖАНИЕ", style["structural"]),
        toc,
        PageBreak(),
        Paragraph("ВВЕДЕНИЕ", style["structural"]),
        Paragraph(markup(" ".join(intro)), style["body"]),
    ]
    rendered = report_body(root, body_source, style)
    appendix = next(
        i
        for i, item in enumerate(rendered)
        if isinstance(item, Paragraph) and item.getPlainText() == "ПРИЛОЖЕНИЕ А"
    )
    story.extend(rendered[: appendix - 1])
    story.extend(
        [
            PageBreak(),
            Paragraph("СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", style["structural"]),
            Paragraph(
                markup(
                    "1 Ulrich K. [Servo](https://archive.ics.uci.edu/dataset/87/servo). UCI Machine Learning Repository. 1986. DOI 10.24432/C5Q30F. Лицензия CC BY 4.0."
                ),
                style["body"],
            ),
            Paragraph(
                markup(
                    f"2 [ГОСТ 7.32-2017]({GOST_URL}). Отчёт о научно-исследовательской работе. Структура и правила оформления. Росстандарт."
                ),
                style["body"],
            ),
        ]
    )
    story.extend(rendered[appendix - 1 :])
    temporary = root / ".report.tmp.pdf"
    document = Document(
        str(temporary),
        pagesize=A4,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=VERTICAL,
        bottomMargin=VERTICAL,
        title=title,
        pageCompression=1,
    )
    frame = Frame(
        LEFT,
        VERTICAL,
        TEXT_WIDTH,
        A4[1] - 2 * VERTICAL,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document.addPageTemplates(PageTemplate(id="A4", frames=[frame], onPage=footer))
    try:
        document.multiBuild(story, canvasmaker=invariant_canvas)
        temporary.replace(root / "report.pdf")
    finally:
        temporary.unlink(missing_ok=True)
