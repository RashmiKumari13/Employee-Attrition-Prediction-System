from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 42


def _draw_wrapped_text(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    font_name: str = "Helvetica",
    font_size: int = 10,
    line_height: int = 14,
    color: colors.Color = colors.black,
) -> float:
    lines = simpleSplit(text or "", font_name, font_size, max_width)
    pdf.setFont(font_name, font_size)
    pdf.setFillColor(color)

    for line in lines:
        pdf.drawString(x, y, line)
        y -= line_height

    return y


def _draw_shap_graph(
    pdf: canvas.Canvas,
    contributors: list[dict[str, Any]],
    x: float,
    y: float,
    width: float,
) -> float:
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor("#3B2B24"))
    pdf.drawString(x, y, "SHAP Contributor Graph")
    y -= 20

    if not contributors:
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#6E5F57"))
        pdf.drawString(x, y, "No SHAP contributors found for this prediction.")
        return y - 14

    chart_x = x + 152
    chart_width = width - 210
    center_x = chart_x + chart_width / 2
    row_height = 18
    max_abs = max(abs(float(item.get("shap_value", 0.0))) for item in contributors[:8]) or 1.0

    graph_bottom = y - (row_height * len(contributors[:8])) - 6
    pdf.setStrokeColor(colors.HexColor("#C8B7AE"))
    pdf.setLineWidth(1)
    pdf.line(center_x, graph_bottom, center_x, y + 3)

    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(colors.HexColor("#6E5F57"))
    pdf.drawString(chart_x, y + 8, "Risk Down")
    pdf.drawRightString(chart_x + chart_width, y + 8, "Risk Up")

    for index, item in enumerate(contributors[:8]):
        shap_value = float(item.get("shap_value", 0.0))
        feature = str(item.get("feature", "Unknown"))
        feature_label = feature[:23]
        label_y = y - (index * row_height) - 3
        bar_y = label_y - 7
        half_width = (chart_width / 2) - 6
        bar_width = max((abs(shap_value) / max_abs) * half_width, 4)

        pdf.setFont("Helvetica", 9)
        pdf.setFillColor(colors.HexColor("#4C3E36"))
        pdf.drawRightString(chart_x - 8, label_y, feature_label)

        fill_color = colors.HexColor("#B3261E") if shap_value >= 0 else colors.HexColor("#2F7F5F")
        start_x = center_x if shap_value >= 0 else center_x - bar_width
        pdf.setFillColor(fill_color)
        pdf.roundRect(start_x, bar_y, bar_width, 8, 2, stroke=0, fill=1)

        pdf.setFont("Helvetica", 8)
        pdf.setFillColor(colors.HexColor("#4C3E36"))
        prefix = "+" if shap_value > 0 else ""
        pdf.drawString(chart_x + chart_width + 8, label_y - 1, f"{prefix}{shap_value:.4f}")

    return graph_bottom - 16


def build_explainability_pdf(
    prediction: dict[str, Any],
    employee_name: str | None = None,
) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    title = "Employee Attrition Explainability Report"
    employee_label = employee_name.strip() if employee_name else "Employee Profile"
    risk_percent = prediction.get("attrition_percent", 0)
    risk_label = prediction.get("risk_label", "Unknown")
    summary = prediction.get("summary", "No summary generated.")
    recommendations = prediction.get("recommendations", [])
    contributors = prediction.get("explainability", {}).get("top_contributors", [])

    # Header
    pdf.setFillColor(colors.HexColor("#402218"))
    pdf.roundRect(MARGIN_X, PAGE_HEIGHT - 104, PAGE_WIDTH - (MARGIN_X * 2), 72, 10, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(MARGIN_X + 14, PAGE_HEIGHT - 60, title)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(MARGIN_X + 14, PAGE_HEIGHT - 78, f"Generated: {timestamp}")
    pdf.drawRightString(PAGE_WIDTH - MARGIN_X - 14, PAGE_HEIGHT - 78, employee_label)

    current_y = PAGE_HEIGHT - 132

    # Risk section
    pdf.setFillColor(colors.HexColor("#FFF3E8"))
    pdf.roundRect(MARGIN_X, current_y - 62, PAGE_WIDTH - (MARGIN_X * 2), 56, 8, stroke=0, fill=1)
    pdf.setFillColor(colors.HexColor("#3B2B24"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN_X + 12, current_y - 26, "Risk Score")
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(MARGIN_X + 12, current_y - 52, f"{risk_percent}%")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(MARGIN_X + 120, current_y - 46, f"({risk_label})")
    current_y -= 82

    # Summary section
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor("#3B2B24"))
    pdf.drawString(MARGIN_X, current_y, "Explainability Summary")
    current_y -= 16
    current_y = _draw_wrapped_text(
        pdf=pdf,
        text=summary,
        x=MARGIN_X,
        y=current_y,
        max_width=PAGE_WIDTH - (MARGIN_X * 2),
        font_name="Helvetica",
        font_size=10,
        line_height=14,
        color=colors.HexColor("#4C3E36"),
    )
    current_y -= 8

    # Top factors
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor("#3B2B24"))
    pdf.drawString(MARGIN_X, current_y, "Top Contributing Factors")
    current_y -= 16

    if contributors:
        for factor in contributors[:6]:
            feature = str(factor.get("feature", "Unknown"))
            direction = str(factor.get("direction", "unknown")).replace("_", " ")
            value = float(factor.get("shap_value", 0.0))
            prefix = "+" if value > 0 else ""
            bullet = f"- {feature}: {prefix}{value:.4f} ({direction})"
            current_y = _draw_wrapped_text(
                pdf=pdf,
                text=bullet,
                x=MARGIN_X + 4,
                y=current_y,
                max_width=PAGE_WIDTH - (MARGIN_X * 2) - 8,
                font_name="Helvetica",
                font_size=10,
                line_height=13,
                color=colors.HexColor("#4C3E36"),
            )
            current_y -= 1
    else:
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#6E5F57"))
        pdf.drawString(MARGIN_X + 4, current_y, "- No contributors available.")
        current_y -= 14

    current_y -= 8
    current_y = _draw_shap_graph(
        pdf=pdf,
        contributors=contributors,
        x=MARGIN_X,
        y=current_y,
        width=PAGE_WIDTH - (MARGIN_X * 2),
    )

    # Recommendations
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor("#3B2B24"))
    pdf.drawString(MARGIN_X, current_y, "Retention Recommendations")
    current_y -= 16

    for recommendation in recommendations[:6]:
        current_y = _draw_wrapped_text(
            pdf=pdf,
            text=f"- {recommendation}",
            x=MARGIN_X + 4,
            y=current_y,
            max_width=PAGE_WIDTH - (MARGIN_X * 2) - 8,
            font_name="Helvetica",
            font_size=10,
            line_height=13,
            color=colors.HexColor("#4C3E36"),
        )
        current_y -= 1

    if not recommendations:
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#6E5F57"))
        pdf.drawString(MARGIN_X + 4, current_y, "- Continue monitoring engagement drivers.")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

