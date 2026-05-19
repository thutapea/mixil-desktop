"""
Build json/inventory_full.json from formatted-inventory.xlsx
=============================================================
Converts every row of the spreadsheet into a JSON entry with:
  - page_content  : natural-language sentence ready for embedding
  - metadata      : all structured fields (name, location, count, etc.)

Usage:
    python scripts/build_json.py
    python scripts/build_json.py --input inventory_pipeline/formatted-inventory.xlsx
    python scripts/build_json.py --output json/my_inventory.json
"""

import json
import argparse
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Known manufacturer prefixes – used for best-effort make/model extraction
# ---------------------------------------------------------------------------
KNOWN_MAKES = [
    "Agilent Technologies", "Agilent", "Keysight Technologies", "Keysight",
    "Fluke", "Mini-Circuits", "MINI CIRCUITS", "Keithley",
    "National Instruments", "Thorlabs", "Tektronix",
    "Anritsu", "HP", "Hewlett-Packard",
    "Adam Equipment", "Adam", "Amput",
    "U.S. Solid", "AMRESCO SOLAR", "AMRESCO",
    "Shenzhen Xiangding Technology", "Shenzhen",
    "Texas Instruments", "Arduino", "Raspberry Pi", "Raspberry",
    "STMicroelectronics", "DJI", "Pixhawk", "Holybro",
    "3DR", "FrSky", "TBS", "ELRS", "Spektrum",
    "SainSmart", "Elegoo", "Adafruit", "SparkFun",
    "Cisco", "TP-Link", "Ubiquiti",
    "Bosch", "Dewalt", "Milwaukee", "Makita",
    "Hakko", "Weller", "Metcal",
    "Rigol", "Siglent",
]


def extract_make_model(name: str) -> tuple[str, str]:
    """Best-effort extraction of manufacturer and model number from a raw name."""
    name_stripped = name.strip()
    make = ""
    model = ""

    for m in KNOWN_MAKES:
        if name_stripped.lower().startswith(m.lower()):
            make = m
            rest = name_stripped[len(m):].strip()
            # First token that contains a digit is likely the model number
            for tok in rest.split():
                clean = tok.strip(".,()[]")
                if re.search(r"\d", clean):
                    model = clean
                    break
            break

    if not make and name_stripped:
        # Fall back: first word as make
        make = name_stripped.split()[0]

    return make, model


def clean_str(val) -> str:
    s = str(val).strip()
    return "" if s in ("nan", "NaT", "None", "") else s


def build_page_content(name: str, location: str, count: int,
                       inv_num: str, serial: str, desc: str) -> str:
    loc_str = f"located at {location}" if location else "location unspecified"
    parts = [f"The {name} is {loc_str}."]
    parts.append(f"Count: {count}.")
    if inv_num:
        parts.append(f"Inventory #: {inv_num}.")
    if serial:
        parts.append(f"S/N: {serial}.")
    if desc:
        parts.append(desc.rstrip(".") + ".")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Build inventory JSON from xlsx")
    parser.add_argument(
        "--input",
        default="inventory_pipeline/formatted-inventory.xlsx",
        help="Path to the xlsx file (relative to project root)",
    )
    parser.add_argument(
        "--output",
        default="json/inventory_full.json",
        help="Output JSON path (relative to project root)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    input_path = root / args.input
    output_path = root / args.output

    try:
        import pandas as pd
    except ImportError:
        print("pandas + openpyxl required:  pip install pandas openpyxl")
        return

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    df = pd.read_excel(input_path)
    print(f"Read {len(df)} rows from {input_path.name}")

    items = []
    skipped = 0

    for _, row in df.iterrows():
        name = clean_str(row.get("Name", ""))
        if not name:
            skipped += 1
            continue

        location = clean_str(row.get("Place", ""))
        inv_num = clean_str(row.get("Inventory Number", ""))
        serial = clean_str(row.get("Serial Number", ""))
        desc = clean_str(row.get("Description", ""))

        count_raw = row.get("Count", 1)
        try:
            count = int(count_raw) if pd.notna(count_raw) else 1
        except (ValueError, TypeError):
            count = 1

        make, model = extract_make_model(name)

        content = build_page_content(name, location, count, inv_num, serial, desc)

        items.append({
            "page_content": content,
            "metadata": {
                "inventory_number": inv_num,
                "name": name,
                "make": make,
                "model": model,
                "serial_number": serial,
                "location": location,
                "count": count,
                "description": desc,
                "source": "Lab Inventory",
            },
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Written {len(items)} items ({skipped} skipped blank rows) → {output_path}")


if __name__ == "__main__":
    main()
