import os
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OptimizeRequest(BaseModel):
    resume_text: str
    job_text: str


class OptimizeResponse(BaseModel):
    optimized_text: str
    summary: str
    keywords: List[str]
    bullets: List[str]
    match_score: int


class ExportRequest(BaseModel):
    format: str  # txt | md | pdf | docx
    optimized_text: str
    filename: Optional[str] = "optimized_resume"


# ---- Simple NLP helpers (heuristic) ----

def _tokenize(text: str) -> List[str]:
    import re

    return [
        w
        for w in re.sub(r"[^a-z0-9+\s]", " ", (text or "").lower()).split()
        if w
    ]


def _unique(words: List[str]) -> List[str]:
    return list(dict.fromkeys(words))


def _match_score(resume: str, job: str) -> int:
    r = set(_unique(_tokenize(resume)))
    j = _unique(_tokenize(job))
    if not j:
        return 0
    overlap = sum(1 for w in j if w in r)
    return round((overlap / len(j)) * 100)


def _missing_keywords(resume: str, job: str, limit: int = 12) -> List[str]:
    r = set(_tokenize(resume))
    j = _unique(_tokenize(job))
    miss = [w for w in j if len(w) > 2 and w not in r]
    return miss[:limit]


def _derive_suggestions(resume: str, job: str):
    keywords = _missing_keywords(resume, job, 12)
    bullets = [
        f"• Led {kw} initiative delivering 12–25% improvement in a core KPI within two quarters."
        for kw in keywords[:8]
    ]
    focus = ", ".join(keywords[:4]) or "role-aligned capabilities"
    summary = (
        f"Results-driven professional with strengths in {focus}. "
        f"Proven record of shipping impact, collaborating cross-functionally, and quantifying outcomes."
    )
    return summary, keywords, bullets


def _assemble_resume(original: str, summary: str, bullets: List[str], keywords: List[str]) -> str:
    header = "Optimized Resume\n=================\n"
    advice = (
        "\nKeywords to weave in: " + (", ".join(keywords) if keywords else "N/A") + "\n\n"
    )
    impact = "Key Impact Bullets:\n" + ("\n".join(bullets) if bullets else "• Quantified achievement here") + "\n\n"
    body = original.strip()
    return f"{header}{summary}\n\n{advice}{impact}{body}\n"


@app.get("/")
def read_root():
    return {"message": "AI Resume Optimizer Backend"}


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(payload: OptimizeRequest):
    resume = payload.resume_text or ""
    job = payload.job_text or ""
    summary, keywords, bullets = _derive_suggestions(resume, job)
    optimized = _assemble_resume(resume, summary, bullets, keywords)
    score = _match_score(resume, job)
    return OptimizeResponse(
        optimized_text=optimized,
        summary=summary,
        keywords=keywords,
        bullets=bullets,
        match_score=score,
    )


@app.post("/export")
async def export_file(req: ExportRequest):
    fmt = (req.format or "txt").lower()
    text = req.optimized_text or ""
    name = (req.filename or "optimized_resume").replace(" ", "_")

    if fmt == "txt":
        data = text.encode("utf-8")
        return StreamingResponse(
            BytesIO(data),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={name}.txt"},
        )

    if fmt == "md":
        md = f"# {name.replace('_', ' ').title()}\n\n" + text
        data = md.encode("utf-8")
        return StreamingResponse(
            BytesIO(data),
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={name}.md"},
        )

    if fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF engine not available: {e}")

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        x_margin, y_margin = 0.75 * inch, 0.75 * inch
        max_width = width - 2 * x_margin
        text_obj = c.beginText(x_margin, height - y_margin)
        text_obj.setFont("Helvetica", 10)

        # simple word-wrap
        for line in text.split("\n"):
            words = line.split(" ")
            current = ""
            for w in words:
                test = (current + " " + w).strip()
                if c.stringWidth(test, "Helvetica", 10) <= max_width:
                    current = test
                else:
                    text_obj.textLine(current)
                    current = w
            text_obj.textLine(current)
        c.drawText(text_obj)
        c.showPage()
        c.save()
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={name}.pdf"},
        )

    if fmt == "docx":
        try:
            from docx import Document
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DOCX engine not available: {e}")

        document = Document()
        for line in text.split("\n"):
            document.add_paragraph(line)
        out = BytesIO()
        document.save(out)
        out.seek(0)
        return StreamingResponse(
            out,
            media_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            headers={"Content-Disposition": f"attachment; filename={name}.docx"},
        )

    raise HTTPException(status_code=400, detail="Unsupported format. Use txt|md|pdf|docx.")


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        from database import db  # type: ignore

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = getattr(db, "name", "✅ Connected")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os

    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
