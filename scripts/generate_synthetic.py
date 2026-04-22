"""Generate a small synthetic warranty dataset for end-to-end smoke testing.

Produces `tests/fixtures/synthetic_warranty.xlsx` with 20 rows containing:
- A mix of make/model/model_year/mileage
- English customer complaints and technician comments
- Mixed DTC patterns (P0171, P0300, C0035, B1000, U0100, plus typos like
  p0l7l and lowercased codes) so Step 5 regex + typo correction can be
  exercised
- Some missing values so Step 4 NaN handling can be exercised

Run directly: `python scripts/generate_synthetic.py`
"""

from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

import pandas as pd

random.seed(7)


def _dates(n: int) -> list[dt.date]:
    """Return n distinct repair dates spread across ~8 months."""
    base = dt.date(2024, 3, 1)
    return [base + dt.timedelta(days=random.randint(0, 240)) for _ in range(n)]


def _vins(n: int) -> list[str]:
    """Synthetic 17-char VINs; not valid VIN check-digit, only for tests."""
    alphabet = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    return ["".join(random.choices(alphabet, k=17)) for _ in range(n)]


ROWS = [
    # (complaint, tech_comment, model, model_year, mileage, dealer, region)
    (
        "Engine misfires at idle and check engine light on.",
        "Found P0300 random misfire. Replaced spark plugs and ignition coil cyl 3.",
        "Sedan A", 2023, 12500, "D001", "US-CA",
    ),
    (
        "AC blows warm air on driver side.",
        "Recharged refrigerant; suspected slow leak at condenser. No DTC.",
        "SUV B", 2022, 28750, "D002", "US-TX",
    ),
    (
        "ABS warning light illuminated, brake pedal feels normal.",
        "Retrieved C0035 LF wheel speed sensor. Replaced sensor; cleared codes.",
        "Sedan A", 2022, 41200, "D003", "US-NY",
    ),
    (
        "Battery dead after vehicle sat 3 days.",
        "Parasitic draw 450 mA from BCM. DTC b1000 present. Reprogrammed BCM per TSB.",
        "Sedan A", 2021, 55300, "D001", "US-CA",
    ),
    (
        "Infotainment head unit reboots randomly.",
        "U0100 lost communication ECM intermittent. Updated head unit firmware.",
        "SUV B", 2023, 8900, "D002", "US-TX",
    ),
    (
        "Vibration at highway speeds around 65 mph.",
        "Road-force balanced all four wheels. No DTC. Recommend tire rotation.",
        "Sedan A", 2022, 33450, "D004", "US-FL",
    ),
    (
        "Transmission shudders on 2-3 upshift.",
        "P0776 pressure control solenoid B. Performed transmission fluid exchange.",
        "SUV B", 2021, 62100, "D003", "US-NY",
    ),
    (
        "Water leak near windshield during rain.",
        "Resealed windshield urethane bead. No electrical faults.",
        "Sedan A", 2020, 71000, "D004", "US-FL",
    ),
    (
        "Check engine light, rough idle.",
        "P0171 system too lean bank 1. Cleaned MAF, replaced vacuum hose.",
        "Sedan A", 2022, 22800, "D001", "US-CA",
    ),
    (
        "Brake pedal pulsates when stopping.",
        "Front rotors warped; machined rotors, replaced pads.",
        "SUV B", 2022, 30500, "D002", "US-TX",
    ),
    (
        "",  # missing complaint — NaN handling
        "No DTC stored. Customer-reported noise not reproducible.",
        "Sedan A", 2023, 6500, "D001", "US-CA",
    ),
    (
        "Engine overheating in traffic.",
        "Found water pump seeping. Replaced water pump, refilled coolant.",
        "SUV B", 2021, 48900, "D003", "US-NY",
    ),
    (
        "Lane keep assist disengages on highway.",
        "Camera calibration performed per OEM bulletin. No DTC.",
        "Sedan A", 2023, 14200, "D004", "US-FL",
    ),
    (
        "Forward-collision warning false alert.",
        "Front radar aimed and calibrated.",
        "SUV B", 2023, 9800, "D002", "US-TX",
    ),
    (
        "Seat memory function not saving position.",
        "Reprogrammed driver seat module.",
        "Sedan A", 2022, 26500, "D001", "US-CA",
    ),
    (
        "Engine knock noise on cold start.",
        "Replaced timing chain tensioner per TSB. DTC p0l7l noted (customer earlier).",  # typo P0171
        "Sedan A", 2020, 82000, "D003", "US-NY",
    ),
    (
        "Fuel gauge reads empty with full tank.",
        "Fuel level sensor faulty. Replaced sending unit.",
        "SUV B", 2022, 37600, "D002", "US-TX",
    ),
    (
        "Catalytic converter warning.",
        "p0420 catalyst efficiency low bank 1. Replaced catalytic converter assembly.",
        "Sedan A", 2019, 96500, "D004", "US-FL",
    ),
    (
        "Bluetooth will not pair to phone.",
        "Cleared head unit user data; re-paired successfully.",
        "SUV B", 2023, 5400, "D002", "US-TX",
    ),
    (
        "Steering pull to the right.",
        "Four-wheel alignment performed. No DTC.",
        "Sedan A", 2021, 44300, "D001", "US-CA",
    ),
]


def build_dataframe() -> pd.DataFrame:
    """Assemble the dataframe with the expected English column names."""
    n = len(ROWS)
    dates = _dates(n)
    vins = _vins(n)

    records = []
    for i, (complaint, tech, model, my, miles, dealer, region) in enumerate(ROWS):
        in_service = dates[i] - dt.timedelta(days=random.randint(200, 1200))
        records.append({
            "vin": vins[i],
            "claim_id": f"RO{1000 + i:05d}",
            "repair_date": dates[i],
            "in_service_date": in_service,
            "make": "Acme",
            "model": model,
            "model_year": my,
            "trim": random.choice(["SE", "SEL", "Limited"]),
            "engine_type": random.choice(["ICE-2.0L", "ICE-2.5L", "HEV-2.0L"]),
            "transmission_type": random.choice(["AT-8", "CVT", "DCT-7"]),
            "drivetrain": random.choice(["FWD", "AWD"]),
            "plant_code": random.choice(["P1", "P2"]),
            "mileage_at_repair": miles,
            "region": region,
            "dealer_id": dealer,
            "customer_complaint": complaint,
            "tech_comment": tech,
            "repair_cost": round(random.uniform(80.0, 1500.0), 2),
            "labor_hours": round(random.uniform(0.3, 4.5), 1),
        })
    return pd.DataFrame.from_records(records)


def main() -> None:
    df = build_dataframe()
    here = Path(__file__).resolve().parent
    out = here.parent / "tests" / "fixtures" / "synthetic_warranty.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False, engine="openpyxl")
    print(f"Wrote {out} with {len(df)} rows and {len(df.columns)} columns.")


if __name__ == "__main__":
    main()
