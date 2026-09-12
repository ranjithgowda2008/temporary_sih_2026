import io
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()


# =====================================================================
# RULE 7: PRINCIPAL DISPLAY PANEL (PDP) & FONT SIZE STATUTORY LOGIC
# =====================================================================

FONT_SIZE_TABLE = [
    {"min_area_cm2": 0, "max_area_cm2": 50, "normal_mm": 1.0, "molded_mm": 1.5},
    {"min_area_cm2": 50, "max_area_cm2": 100, "normal_mm": 1.5, "molded_mm": 3.0},
    {"min_area_cm2": 100, "max_area_cm2": 500, "normal_mm": 2.5, "molded_mm": 4.0},
    {"min_area_cm2": 500, "max_area_cm2": 2500, "normal_mm": 4.0, "molded_mm": 6.0},
    {"min_area_cm2": 2500, "max_area_cm2": None, "normal_mm": 6.0, "molded_mm": 6.0},
]


def calculate_panel_area(shape: str, **dimensions) -> float:
    """Calculates Principal Display Panel (PDP) area (cm^2) per Rule 7(4)."""
    shape = str(shape).lower()
    if shape == "rectangular":
        return dimensions.get("height_cm", 0.0) * dimensions.get("width_cm", 0.0)
    elif shape == "cylindrical":
        return 0.40 * dimensions.get("height_cm", 0.0) * dimensions.get("circumference_cm", 0.0)
    elif shape == "other":
        return 0.40 * dimensions.get("total_surface_area_cm2", 0.0)
    return 0.0


def get_minimum_font_height(panel_area_cm2: float, is_molded: bool = False) -> float:
    """Returns minimum required font height (mm) per Rule 7(2)."""
    for row in FONT_SIZE_TABLE:
        min_a, max_a = row["min_area_cm2"], row["max_area_cm2"]
        if max_a is None:
            if panel_area_cm2 >= min_a:
                return row["molded_mm"] if is_molded else row["normal_mm"]
        elif min_a <= panel_area_cm2 < max_a:
            return row["molded_mm"] if is_molded else row["normal_mm"]
    return 1.0


class FontSizeChecker:
    """Checks detected font height against statutory Rule 7 requirements."""

    def check(
        self,
        package_shape: str,
        dimensions: dict,
        detected_font_height_mm: float,
        is_molded: bool = False,
    ) -> dict:
        panel_area = calculate_panel_area(package_shape, **dimensions)
        min_required = get_minimum_font_height(panel_area, is_molded)
        compliant = detected_font_height_mm >= min_required

        issue = None
        if not compliant:
            issue = (
                f"Detected font height {detected_font_height_mm}mm is below the "
                f"{min_required}mm minimum required for a {panel_area:.1f} cm² panel."
            )

        return {
            "value": f"{detected_font_height_mm} mm",
            "issue": issue,
            "panel_area_cm2": round(panel_area, 1),
            "minimum_required_mm": min_required,
            "compliant": compliant,
        }


# =====================================================================
# MULTIMODAL VISION EXTRACTION PIPELINE
# =====================================================================

class MultimodalExtractor:
    """Multimodal Vision Extraction Engine using Gemini VLM."""

    @staticmethod
    def preprocess_image(image_bytes: bytes) -> Image.Image:
        """Applies CLAHE contrast enhancement to improve OCR on reflective or curved surfaces."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes.")

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        rgb_img = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_img)

    def extract_with_vision(self, image_bytes: bytes) -> Tuple[Dict[str, Any], float, str]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing or empty in your .env file.")

        client = genai.Client(api_key=api_key)
        processed_pil = self.preprocess_image(image_bytes)

        prompt = """
        You are an official Indian Legal Metrology Inspector auditing packaged goods under the Legal Metrology (Packaged Commodities) Rules, 2011.
        Analyze the packaging image thoroughly and extract statutory disclosures.

        Return ONLY a raw JSON object with NO markdown code block styling or extra text.

        Target Schema:
        {
          "net_quantity": {
            "value": string or null (e.g., "50 g", "1 kg", "200 ml"),
            "confidence": float (0.0 to 1.0)
          },
          "mrp": {
            "value": float or null (numeric value only, e.g., 20.0),
            "raw": string or null (e.g., "Rs. 20.00"),
            "inclusive_of_taxes": boolean,
            "confidence": float (0.0 to 1.0)
          },
          "batch_details": {
            "value": string or null,
            "confidence": float (0.0 to 1.0)
          },
          "manufacturer": {
            "value": string or null,
            "confidence": float (0.0 to 1.0)
          },
          "mfg_date": {
            "value": string or null (e.g., "05/2026", "MAY 2026"),
            "confidence": float (0.0 to 1.0)
          },
          "consumer_care": {
            "value": string or null,
            "confidence": float (0.0 to 1.0)
          }
        }
        """

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[processed_pil, prompt],
        )

        clean_text = response.text.strip()
        clean_text = re.sub(r"^```json\s*", "", clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r"^```\s*", "", clean_text, flags=re.MULTILINE).strip()

        data = json.loads(clean_text)

        conf_scores = [
            v.get("confidence", 0.0)
            for k, v in data.items()
            if isinstance(v, dict)
        ]
        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.95

        return data, avg_conf, "Hybrid Multimodal Vision Engine"


# =====================================================================
# STATUTORY COMPLIANCE AUDITOR ENGINE
# =====================================================================

class LegalMetrologyAuditor:
    """Enforces statutory checks mapped explicitly to PCR 2011."""

    @staticmethod
    def audit(fields: dict, font_size_result: dict = None) -> dict:
        violations = []
        warnings = []
        audit_trail = []
        step_num = 1

        # Check 1: Rule 6(1)(e) - MRP & Tax Declaration
        mrp_info = fields.get("mrp", {})
        mrp_val = mrp_info.get("value")
        has_tax = mrp_info.get("inclusive_of_taxes", False)

        if not mrp_val:
            violations.append("Rule 6(1)(e): Maximum Retail Price (MRP) is missing.")
            audit_trail.append({"step": step_num, "field": "MRP", "result": "FAIL", "reason": "MRP not declared."})
        elif not has_tax:
            violations.append("Rule 6(1)(e): Mandatory declaration 'Incl. of all taxes' is missing alongside MRP.")
            audit_trail.append({"step": step_num, "field": "MRP", "result": "FAIL", "reason": "Tax inclusion clause missing."})
        else:
            audit_trail.append({"step": step_num, "field": "MRP", "result": "PASS", "reason": f"Declared: Rs. {mrp_val} (Incl. of all taxes)"})
        step_num += 1

        # Check 2: Rule 6(1)(c) - Net Quantity & Standard Metric Units
        net_info = fields.get("net_quantity", {})
        net_val = net_info.get("value")
        valid_units = ["g", "gm", "kg", "ml", "l", "ltr", "liter", "litres", "n", "units", "pcs"]

        if not net_val:
            violations.append("Rule 6(1)(c): Net quantity declaration is missing.")
            audit_trail.append({"step": step_num, "field": "Net Quantity", "result": "FAIL", "reason": "Net quantity not declared."})
        else:
            unit_found = any(u in str(net_val).lower() for u in valid_units)
            if not unit_found:
                warnings.append(f"Rule 6(1)(c): Quantity unit in '{net_val}' should conform strictly to standard metric units (g, kg, ml, l, N).")
                audit_trail.append({"step": step_num, "field": "Net Quantity", "result": "WARNING", "reason": "Non-standard unit."})
            else:
                audit_trail.append({"step": step_num, "field": "Net Quantity", "result": "PASS", "reason": f"Declared: {net_val}"})
        step_num += 1

        # Check 3: Rule 6(1)(a) - Manufacturer Details
        mfg_info = fields.get("manufacturer", {})
        mfg_val = mfg_info.get("value")
        if not mfg_val:
            violations.append("Rule 6(1)(a): Name and address of manufacturer/packer is missing.")
            audit_trail.append({"step": step_num, "field": "Manufacturer", "result": "FAIL", "reason": "Manufacturer details missing."})
        else:
            audit_trail.append({"step": step_num, "field": "Manufacturer", "result": "PASS", "reason": f"Declared: {mfg_val}"})
        step_num += 1

        # Check 4: Rule 6(1)(d) - Date of Manufacture
        date_info = fields.get("mfg_date", {})
        date_val = date_info.get("value")
        if not date_val:
            violations.append("Rule 6(1)(d): Month and year of manufacture/packing is missing.")
            audit_trail.append({"step": step_num, "field": "Mfg Date", "result": "FAIL", "reason": "Date of manufacture missing."})
        else:
            audit_trail.append({"step": step_num, "field": "Mfg Date", "result": "PASS", "reason": f"Declared: {date_val}"})
        step_num += 1

        # Check 5: Rule 6(2) - Consumer Care Mechanism
        care_info = fields.get("consumer_care", {})
        care_val = care_info.get("value")
        if not care_val:
            warnings.append("Rule 6(2): Dedicated consumer care contact details were not detected.")
            audit_trail.append({"step": step_num, "field": "Consumer Care", "result": "WARNING", "reason": "Consumer care contact missing."})
        else:
            audit_trail.append({"step": step_num, "field": "Consumer Care", "result": "PASS", "reason": f"Declared: {care_val}"})
        step_num += 1

        # Check 6: Rule 7 - Font Size Compliance (Optional)
        if font_size_result is not None:
            if not font_size_result.get("compliant", False):
                violations.append(f"Rule 7 Font Size: {font_size_result['issue']}")
                audit_trail.append({"step": step_num, "field": "Font Size", "result": "FAIL", "reason": font_size_result["issue"]})
            else:
                audit_trail.append({"step": step_num, "field": "Font Size", "result": "PASS", "reason": f"Font height meets {font_size_result['minimum_required_mm']}mm requirement."})

        # Final Status Determination
        if violations:
            status = "NON-COMPLIANT"
        elif warnings:
            status = "COMPLIANT WITH WARNINGS"
        else:
            status = "COMPLIANT"

        return {
            "compliance_status": status,
            "violations": violations,
            "warnings": warnings,
            "audit_trail": audit_trail,
        }