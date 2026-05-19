from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import os


def generate_report(
    helmet_status,
    risk_score,
    emotion,
    signal,
    plate_number,
    accident_risk
):

    # =========================
    # DATE & TIME FORMAT
    # =========================
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # =========================
    # CREATE REPORT FOLDER
    # =========================
    folder = "reports"
    os.makedirs(folder, exist_ok=True)

    # =========================
    # AUTO-NAMED FILE
    # =========================
    filename = f"{plate_number}_{timestamp}.pdf"
    filepath = os.path.join(folder, filename)

    # =========================
    # PDF DOCUMENT
    # =========================
    doc = SimpleDocTemplate(filepath)

    styles = getSampleStyleSheet()
    elements = []

    # =========================
    # TITLE
    # =========================
    title = Paragraph(
        "AI Helmet Traffic System Report",
        styles['Title']
    )

    elements.append(title)
    elements.append(Spacer(1, 20))

    # =========================
    # REPORT DATA
    # =========================
    report_data = [
        f"Date & Time: {now}",
        f"Helmet Status: {helmet_status}",
        f"Risk Score: {risk_score}",
        f"Emotion: {emotion}",
        f"Signal Status: {signal}",
        f"Number Plate: {plate_number}",
        f"Accident Risk: {accident_risk}"
    ]

    for item in report_data:
        p = Paragraph(item, styles['BodyText'])
        elements.append(p)
        elements.append(Spacer(1, 10))

    # =========================
    # BUILD PDF
    # =========================
    doc.build(elements)

    print(f"PDF Report Saved: {filepath}")