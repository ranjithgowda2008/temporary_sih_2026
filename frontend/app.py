import requests
from PIL import Image
import streamlit as st

BACKEND_URL = "http://localhost:8000/api/v1/audit"

st.set_page_config(
    page_title="Legal Metrology Compliance Inspector",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Legal Metrology Compliance Inspector")
st.caption("Automated Statutory Verification Engine — Packaged Commodities Rules, 2011")

# Sidebar Database Audit Trail
st.sidebar.header("📜 Audit Logs Database")
if st.sidebar.button("Refresh Database Logs"):
    try:
        res = requests.get("http://localhost:8000/api/v1/logs")
        if res.status_code == 200:
            logs = res.json()
            st.sidebar.success(f"Total Scans: {len(logs)}")
            for log in logs:
                st.sidebar.markdown(f"**Scan ID:** `{log['scan_id'][:8]}...`")
                st.sidebar.markdown(f"**Status:** {log['compliance_status']}")
                st.sidebar.markdown(f"**File:** {log['filename']}")
                st.sidebar.divider()
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Package Label Ingestion")
    uploaded_file = st.file_uploader("Upload Product Package Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Package Label", use_container_width=True)

    st.markdown("---")
    st.subheader("📐 Rule 7 PDP & Font Metrics (Optional)")
    enable_font_check = st.checkbox("Enable Rule 7 Minimum Font Check")

    package_shape = None
    height_cm = 0.0
    width_cm = 0.0
    circumference_cm = 0.0
    detected_font_mm = 0.0
    is_molded = False

    if enable_font_check:
        package_shape = st.selectbox("Package Geometry", ["rectangular", "cylindrical", "other"])
        c1, c2 = st.columns(2)
        with c1:
            height_cm = st.number_input("Height (cm)", min_value=0.0, value=12.0)
            width_cm = st.number_input("Width (cm)", min_value=0.0, value=8.0)
        with c2:
            circumference_cm = st.number_input("Circumference (cm)", min_value=0.0, value=0.0)
            detected_font_mm = st.number_input("Detected Font Height (mm)", min_value=0.0, value=2.0)
        is_molded = st.checkbox("Molded / Embossed Package Label")

    analyze_btn = st.button("⚡ Run Statutory Audit", type="primary", use_container_width=True)

with col2:
    st.subheader("🔍 Compliance Audit Dashboard")

    if uploaded_file and analyze_btn:
        with st.spinner("Analyzing Packaging Disclosures & Verifying Rules..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                payload = {}
                if enable_font_check:
                    payload = {
                        "package_shape": package_shape,
                        "height_cm": height_cm,
                        "width_cm": width_cm,
                        "circumference_cm": circumference_cm,
                        "detected_font_height_mm": detected_font_mm,
                        "is_molded": is_molded,
                    }

                response = requests.post(BACKEND_URL, files=files, data=payload)

                if response.status_code == 200:
                    data = response.json()

                    # Metrics Overview
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Engine Confidence", f"{data['confidence']}%")
                    m2.metric("Extraction Source", "Hybrid Vision Engine")
                    m3.metric("Legal Status", data["status"])

                    st.markdown("---")

                    # Parsed Disclosures Table
                    st.subheader("📋 Parsed Disclosures")
                    fields = data["fields"]
                    table_rows = []
                    for key, obj in fields.items():
                        val = obj.get("value") if isinstance(obj, dict) else obj
                        table_rows.append({
                            "Parameter": key.replace("_", " ").upper(),
                            "Extracted Value": str(val) if val is not None else "NOT DETECTED",
                        })
                    st.table(table_rows)

                    # Rule 7 Panel Verification Result
                    if data.get("font_size_check"):
                        st.subheader("📏 Rule 7 Font Size Verification")
                        f_info = data["font_size_check"]
                        f_col1, f_col2 = st.columns(2)
                        f_col1.metric("Panel Area", f"{f_info['panel_area_cm2']} cm²")
                        f_col2.metric("Min Required Height", f"{f_info['minimum_required_mm']} mm")
                        if not f_info["compliant"]:
                            st.error(f_info["issue"])
                        else:
                            st.success("Font size complies with Rule 7 minimum height standards.")

                    # Violations & Warnings Display
                    st.subheader("❌ Rule Violations & Warnings")
                    if data["violations"]:
                        for v in data["violations"]:
                            st.error(v)
                    else:
                        st.success("Zero statutory rule violations detected under PCR 2011.")

                    if data["warnings"]:
                        for w in data["warnings"]:
                            st.warning(w)

                    with st.expander("🏛️ View Government Inspection Audit Trail"):
                        st.json(data["audit_trail"])

                else:
                    # Safe Error Handling
                    try:
                        error_detail = response.json().get("detail", response.text)
                    except Exception:
                        error_detail = response.text
                    st.error(f"Backend Audit Error ({response.status_code}): {error_detail}")

            except Exception as err:
                st.error(f"Failed to communicate with audit backend: {err}")