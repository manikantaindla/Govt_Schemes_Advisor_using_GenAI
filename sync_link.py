import json
import os
from pathlib import Path
import requests

BASE = Path(__file__).resolve().parent
PDF_DIR = BASE / "data" / "sources" / "pdfs_raw"
OUT_JSON = BASE / "data" / "sources" / "scheme_links.json"

SCHEMES = [
    {
        "scheme_id": "ap_ntr_bharosa_ssp",
        "scheme_name": "AP Social Security Pensions (NTR Bharosa)",
        "state": "ap",
        "apply_link": "https://gsws-nbm.ap.gov.in/NBM/Home/Main",
        "source_links": [
            "https://sspensions.ap.gov.in/",
            "https://sspensions.ap.gov.in/SSP/Documents/GO%20MS%2043%2013.06.2024.pdf",
        ],
    },
    {
        "scheme_id": "tel_kalyana_lakshmi",
        "scheme_name": "Kalyana Lakshmi / Shaadi Mubarak",
        "state": "telangana",
        "apply_link": "https://telanganaepass.cgg.gov.in/KalyanaLakshmiLinks.jsp",
        "source_links": [
            "https://telanganaepass.cgg.gov.in/KalyanLakshmi.do",
            "https://wdsc.telangana.gov.in/PwD/GOs/GO.Ms.No.04%20PwD%20Kalyana%20Lashmi%20Pathakam.PDF",
        ],
    },
    {
        "scheme_id": "tel_aasara_pension",
        "scheme_name": "Aasara Pensions",
        "state": "telangana",
        "apply_link": "https://www.cheyutha.telangana.gov.in/SSPTG/UserInterface/Portal/GeneralSearch.aspx",
        "source_links": [
            "https://medak.telangana.gov.in/scheme/aasara-pensions/",
            "https://services.india.gov.in/service/detail/telangana-aasara-pensions-undersociety-for-elimination-of-rural-poverty-details-of-the-beneficiariess",
        ],
    },
]

def download_pdf(url: str, folder: Path) -> str | None:
    if not url.lower().endswith(".pdf"):
        return None
    folder.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0]
    out_path = folder / filename

    r = requests.get(url, timeout=45)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return str(out_path)

def main():
    # 1) Write mapping JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(SCHEMES, indent=2), encoding="utf-8")
    print(f"✅ Wrote mapping: {OUT_JSON}")

    # 2) Download PDFs (only those in source_links that are PDFs)
    downloaded = 0
    for s in SCHEMES:
        for link in s["source_links"]:
            try:
                p = download_pdf(link, PDF_DIR)
                if p:
                    downloaded += 1
                    print(f"⬇️  {p}")
            except Exception as e:
                print(f"⚠️  Failed: {link} ({e})")

    print(f"✅ PDFs downloaded: {downloaded}")
    print("Next: run `py build_data.py` to rebuild the index.")

if __name__ == "__main__":
    main()