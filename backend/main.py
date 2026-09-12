import uuid
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.database import fetch_all_logs, init_db, save_audit_log
from backend.engine import (
    FontSizeChecker,
    LegalMetrologyAuditor,
    MultimodalExtractor,
)

app = FastAPI(
    title="Legal Metrology Compliance Engine",
    version="3.0.1",
    description="Automated Compliance Inspector for PCR 2011",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vlm_extractor = MultimodalExtractor()
auditor = LegalMetrologyAuditor()
font_checker = FontSizeChecker()


@app.on_event("startup")
def startup_db_client() -> None:
    init_db()


@app.post("/api/v1/audit")
async def run_compliance_audit(
    file: UploadFile = File(...),
    package_shape: Optional[str] = Form(None),
    height_cm: Optional[float] = Form(None),
    width_cm: Optional[float] = Form(None),
    circumference_cm: Optional[float] = Form(None),
    detected_font_height_mm: Optional[float] = Form(None),
    is_molded: Optional[bool] = Form(False),
) -> Dict[str, Any]:

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a valid image file (JPG/PNG)."
        )

    scan_id = str(uuid.uuid4())
    image_bytes = await file.read()

    # Step 1: Multimodal VLM Extraction
    try:
        extracted, confidence, ocr_source = vlm_extractor.extract_with_vision(image_bytes)
    except Exception as exc:
        print("\n" + "=" * 60)
        print("[CRITICAL BACKEND ERROR] VLM Extraction Failed:")
        traceback.print_exc()
        print("=" * 60 + "\n")
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Multimodal VLM Processing Error: {str(exc)}. Please check your terminal console for the full exception trace."
            }
        )

    # Step 2: Rule 7 Font Size Measurement Check
    font_size_result = None
    if package_shape and detected_font_height_mm:
        dimensions = {
            "height_cm": height_cm or 0.0,
            "width_cm": width_cm or 0.0,
            "circumference_cm": circumference_cm or 0.0,
        }
        font_size_result = font_checker.check(
            package_shape=package_shape,
            dimensions=dimensions,
            detected_font_height_mm=detected_font_height_mm,
            is_molded=is_molded or False,
        )

    # Step 3: Legal Audit Verification
    audit_results = auditor.audit(extracted, font_size_result)

    # Step 4: Persist Log Record
    save_audit_log(
        scan_id=scan_id,
        filename=file.filename or "package_image.jpg",
        compliance_status=audit_results["compliance_status"],
        confidence=confidence,
        violations=audit_results["violations"],
        warnings=audit_results["warnings"],
        audit_trail=audit_results["audit_trail"],
    )

    return {
        "scan_id": scan_id,
        "status": audit_results["compliance_status"],
        "confidence": round(confidence * 100, 2),
        "source": ocr_source,
        "fields": extracted,
        "font_size_check": font_size_result,
        "violations": audit_results["violations"],
        "warnings": audit_results["warnings"],
        "audit_trail": audit_results["audit_trail"],
    }


@app.get("/api/v1/logs")
async def get_logs() -> List[Dict[str, Any]]:
    return fetch_all_logs()