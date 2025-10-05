import os
import re
import json
import pdfplumber
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from difflib import SequenceMatcher

# Load .env
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    st.warning("Set GEMINI_API_KEY in .env or environment before running.")
else:
    genai.configure(api_key=GEMINI_KEY)

MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(page_title="PO vs Invoice Comparator", layout="wide")

st.title("📄 PO vs Invoice Comparator (Streamlit + Gemini)")

# ---- Helper: extract text from PDF pages ----
def extract_text_from_pdf(file_obj):
    """Return extracted text from all pages of uploaded PDF (file-like object)."""
    try:
        with pdfplumber.open(file_obj) as pdf:
            pages_text = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
        return "\n\n".join(pages_text).strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return ""

# ---- Helper: call Gemini to extract structured JSON ----
def call_gemini_for_structure(text, doc_type="Document"):
    """Ask Gemini to extract structured JSON from text."""
    if not GEMINI_KEY:
        return None, "No API key configured."

    prompt = f"""
You are a reliable document parser. Given the following {doc_type} text (may be multi-page, may contain tables),
extract the following fields and return only valid JSON (no explanation):

- document_type: "Purchase Order" or "Invoice"
- number: document number (PO number or Invoice number)
- vendor: vendor / supplier name
- date: date if present (any common format)
- grand_total: grand total amount if present (numeric)
- items: list of item objects, each with: description, qty (number), unit_price (number), total (number)

Return result exactly as JSON. Example:
{{ "document_type":"Purchase Order", "number":"PO-12345", "vendor":"ACME Ltd",
  "date":"2024-03-01", "grand_total":"12345.67",
  "items":[{{"description":"Bolt 10mm","qty":10,"unit_price":5.0,"total":50.0}}, ...] }}
  
Here is the {doc_type} text to analyze:

\"\"\"{text[:15000]}\"\"\"
    """

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Try to extract JSON substring
        json_text = raw
        # If there is extra text around JSON, try to find first { ... } block
        if not (json_text.startswith("{") and json_text.endswith("}")):
            m = re.search(r"(\{[\s\S]*\})", json_text)
            if m:
                json_text = m.group(1)

        parsed = json.loads(json_text)
        return parsed, None
    except json.JSONDecodeError:
        return None, f"AI returned invalid JSON. Raw response:\n{raw}"
    except Exception as e:
        return None, f"Error calling Gemini: {e}"

# ---- Helper: match two item descriptions (simple similarity) ----
def similar(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

# ---- Comparison logic ----
def compare_structures(po_struct, inv_struct, item_match_threshold=0.7):
    rows = []

    # Compare header-level fields
    header_checks = []
    for field_label, po_key, inv_key in [
        ("Document Number", "number", "number"),
        ("Vendor", "vendor", "vendor"),
        ("Date", "date", "date"),
        ("Grand Total", "grand_total", "grand_total"),
    ]:
        po_val = po_struct.get(po_key, "") if po_struct else ""
        inv_val = inv_struct.get(inv_key, "") if inv_struct else ""
        status = "Match" if str(po_val).strip() and str(inv_val).strip() and str(po_val).strip() == str(inv_val).strip() else "Mismatch"
        header_checks.append((field_label, po_val, inv_val, status))

    # For items: match by best similarity of description
    po_items = po_struct.get("items", []) if po_struct else []
    inv_items = inv_struct.get("items", []) if inv_struct else []
    inv_used = set()

    for po_it in po_items:
        best_match = None
        best_score = 0.0
        best_idx = None
        for idx, inv_it in enumerate(inv_items):
            if idx in inv_used:
                continue
            score = similar(po_it.get("description",""), inv_it.get("description",""))
            if score > best_score:
                best_score = score
                best_match = inv_it
                best_idx = idx

        if best_score >= item_match_threshold:
            inv_used.add(best_idx)
            inv_it = best_match
        else:
            inv_it = None

        # Compare quantities/prices
        po_qty = po_it.get("qty", "")
        inv_qty = inv_it.get("qty", "") if inv_it else ""
        po_price = po_it.get("unit_price", "")
        inv_price = inv_it.get("unit_price", "") if inv_it else ""
        po_total = po_it.get("total", "")
        inv_total = inv_it.get("total", "") if inv_it else ""

        item_status = "Match"
        if str(po_qty) != str(inv_qty) or str(po_price) != str(inv_price) or str(po_total) != str(inv_total):
            item_status = "Mismatch"

        rows.append({
            "field": "Item",
            "po_value": po_it.get("description",""),
            "inv_value": inv_it.get("description","") if inv_it else "",
            "po_qty": po_qty,
            "inv_qty": inv_qty,
            "po_price": po_price,
            "inv_price": inv_price,
            "po_total": po_total,
            "inv_total": inv_total,
            "status": item_status,
            "match_score": round(best_score, 2)
        })

    # Any invoice items not matched -> add as unmatched
    for idx, inv_it in enumerate(inv_items):
        if idx not in inv_used:
            rows.append({
                "field": "Item (Unmatched Invoice)",
                "po_value": "",
                "inv_value": inv_it.get("description",""),
                "po_qty": "",
                "inv_qty": inv_it.get("qty",""),
                "po_price": "",
                "inv_price": inv_it.get("unit_price",""),
                "po_total": "",
                "inv_total": inv_it.get("total",""),
                "status": "Mismatch",
                "match_score": 0.0
            })

    return header_checks, rows

# ---- UI ----
st.sidebar.header("Options")
threshold = st.sidebar.slider("Item match threshold (similarity)", 50, 95, 70) / 100.0

col1, col2 = st.columns(2)
with col1:
    po_file = st.file_uploader("Upload Purchase Order (PDF)", type=["pdf"])
with col2:
    inv_file = st.file_uploader("Upload Invoice (PDF)", type=["pdf"])

if st.button("Compare PO ↔ Invoice"):
    if not po_file or not inv_file:
        st.error("Upload both PO and Invoice PDFs.")
    else:
        with st.spinner("Extracting text from PDFs..."):
            po_text = extract_text_from_pdf(po_file)
            inv_text = extract_text_from_pdf(inv_file)

        st.subheader("Raw extracted text (first 4000 chars)")
        st.code(po_text[:4000], language="text")
        st.code(inv_text[:4000], language="text")

        with st.spinner("Calling Gemini to parse documents..."):
            po_struct, po_err = call_gemini_for_structure(po_text, "Purchase Order")
            inv_struct, inv_err = call_gemini_for_structure(inv_text, "Invoice")

        if po_err:
            st.error(f"PO parsing error: {po_err}")
        if inv_err:
            st.error(f"Invoice parsing error: {inv_err}")

        st.subheader("AI Parsed JSON (PO)")
        st.json(po_struct if po_struct else {})

        st.subheader("AI Parsed JSON (Invoice)")
        st.json(inv_struct if inv_struct else {})

        # Compare
        header_checks, item_rows = compare_structures(po_struct or {}, inv_struct or {}, item_match_threshold=threshold)

        st.subheader("Header Comparison")
        header_df = pd.DataFrame(header_checks, columns=["Field", "PO Value", "Invoice Value", "Status"])
        def color_status(val):
            if val == "Match":
                return "background-color: #d4f7dc"  # green
            else:
                return "background-color: #f7d4d4"  # red
        st.dataframe(header_df.style.applymap(lambda v: "background-color: #d4f7dc" if v=="Match" else "", subset=pd.IndexSlice[:, ["Status"]]), use_container_width=True)

        st.subheader("Line Item Comparison (side-by-side)")
        # Build display table rows
        display_rows = []
        for r in item_rows:
            display_rows.append({
                "PO Item": r["po_value"],
                "PO Qty": r["po_qty"],
                "PO Unit Price": r["po_price"],
                "PO Total": r["po_total"],
                "Invoice Item": r["inv_value"],
                "Invoice Qty": r["inv_qty"],
                "Invoice Unit Price": r["inv_price"],
                "Invoice Total": r["inv_total"],
                "Status": r["status"],
                "Match Score": r["match_score"]
            })
        df_items = pd.DataFrame(display_rows)
        # style: color Status column
        def status_color(val):
            return 'background-color: #d4f7dc' if val == "Match" else 'background-color: #f7d4d4'
        st.dataframe(df_items.style.applymap(lambda v: status_color(v) if v in ["Match","Mismatch"] else "", subset=["Status"]), use_container_width=True)

        st.success("Comparison complete.")

if st.button("Clear"):
    st.experimental_rerun()

st.markdown("---")
st.caption("Notes: Uses pdfplumber to pull text then Gemini to parse structure. If Gemini returns invalid JSON, check raw response and adjust prompt or increase text chunking.")
