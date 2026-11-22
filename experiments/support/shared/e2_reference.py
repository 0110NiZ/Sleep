"""Locate the E2 scores and verify them against the E4 reproduction record."""
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def verified_e2_table():
    record = ROOT / "docs/checks/reproduction/E2_reference.sha256"
    expected, filename = record.read_text(encoding="utf-8").split()
    table = ROOT / filename
    if sha256(table.read_bytes()).hexdigest() != expected:
        raise ValueError(f"E2 scores differ from the archived E4 reference: {table}")
    return table
