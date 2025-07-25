# report.py
from flask import Blueprint, request, send_file, render_template
import pdfkit
from io import BytesIO
from datetime import datetime
import json

report_bp = Blueprint("report", __name__, url_prefix="/api/report")

def _json(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except: return {}
    return x or {}

@report_bp.post("/pdf")
def make_pdf():
    data = request.get_json(force=True) or {}
    ctx = {
        "today": datetime.now().strftime("%d %b %Y %H:%M"),
        "filename": data.get("filename","(none)"),
        "meta": data.get("meta"),
        "ai": _json(data.get("ai")),
        "vc": data.get("vc") or {},
        "top_corr": data.get("top_corr") or [],
        "pca": data.get("pca") or {},
        "kmeans": data.get("kmeans") or {},
        "assoc": data.get("assoc") or [],
        "summary": data.get("summary") or {},
        "charts": data.get("charts") or {}
    }
    html = render_template("report.html", **ctx)
    pdf_bytes = pdfkit.from_string(html, False, options={
        "quiet": "",
        "enable-local-file-access": None
    })
    return send_file(BytesIO(pdf_bytes),
                     mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"{ctx['filename']}_report.pdf")
