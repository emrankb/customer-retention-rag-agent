"""Tests for customer lookup and the supplied-data checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from customer_retention.customer_db import (
    CUSTOMER_COLUMNS, CustomerDB, load_customers,
)
from customer_retention.source_checks import REFERENCE_FILES, inspect_documents, inspect_sources

ROOT = Path(__file__).resolve().parents[1]


def row(customer_id="100001", **changes):
    customer = {
        "CustomerId": customer_id,
        "Surname": "Test",
        "CreditScore": "690",
        "Geography": "France",
        "Gender": "Female",
        "Age": "41",
        "Tenure": "3",
        "Balance": "0",
        "NumOfProducts": "2",
        "HasCrCard": "1",
        "IsActiveMember": "0",
        "EstimatedSalary": "12000.50",
        "Exited": "1",
    }
    customer.update(changes)
    return customer


def write_csv(path: Path, rows: list[dict], sentiment=False):
    fields = list(CUSTOMER_COLUMNS) + (["Sentiment_Label"] if sentiment else [])
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class CustomerDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "Churn_Modeling.csv"
        self.enriched = self.root / "Churn_With_Sentiment.csv"
        self.docs = self.root / "reference"
        self.docs.mkdir()
        write_csv(self.base, [row(), row("100002", Exited="0", Balance="510.25")])
        write_csv(self.enriched, [
            row(Balance="0.0", Sentiment_Label="Positive"),
            row("100002", Exited="0", Balance="510.25", Sentiment_Label="Negative"),
        ], sentiment=True)
        for name in REFERENCE_FILES:
            (self.docs / name).write_text("Fee information: ₹500.\n", encoding="utf-8")

    def test_customer_lookup_returns_typed_values(self):
        db = CustomerDB(self.enriched)
        self.assertEqual(len(db), 2)
        context = db.get_context("100001")
        self.assertEqual(context["customer_id"], "100001")
        self.assertEqual(context["credit_score"], 690)
        self.assertTrue(context["has_credit_card"])
        self.assertFalse(context["is_active_member"])
        self.assertTrue(context["historical_exited"])
        self.assertEqual(context["balance"], "0.0")
        self.assertEqual(context["sentiment_label"], "Positive")
        self.assertNotIn("churn_probability", context)

    def test_integer_id_and_missing_customer(self):
        db = CustomerDB(self.base)
        self.assertEqual(db.get_context(100002)["customer_id"], "100002")
        self.assertIsNone(db.get_context("999999"))
        self.assertIsNone(db.get_context("100001")["sentiment_label"])

    def test_invalid_ids_are_rejected(self):
        db = CustomerDB(self.enriched)
        for bad in (True, 1.5, "100 001", "-2", "", "x123"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                db.get_context(bad)

    def test_source_agreement_ignores_numeric_formatting(self):
        report = inspect_sources(self.root, self.docs)
        self.assertEqual(report["customer_records"], 2)
        self.assertEqual(report["historical_exited"], 1)
        self.assertEqual(report["sentiment_counts"], {"Negative": 1, "Positive": 1})
        self.assertEqual(len(report["documents"]), 6)

    def test_sentiment_file_is_optional(self):
        self.enriched.unlink()
        report = inspect_sources(self.root, self.docs)
        self.assertFalse(report["sentiment_available"])
        self.assertNotIn("sentiment_counts", report)

    def test_mismatched_customer_ids_rejected(self):
        write_csv(self.enriched, [row("999999", Sentiment_Label="Positive")], sentiment=True)
        with self.assertRaisesRegex(ValueError, "Customer IDs differ"):
            inspect_sources(self.root, self.docs)

    def test_mismatched_record_rejected(self):
        write_csv(self.enriched, [
            row(CreditScore="691", Sentiment_Label="Positive"),
            row("100002", Exited="0", Balance="510.25", Sentiment_Label="Negative"),
        ], sentiment=True)
        with self.assertRaisesRegex(ValueError, "Non-sentiment fields differ"):
            inspect_sources(self.root, self.docs)

    def test_duplicate_ids_rejected(self):
        write_csv(self.base, [row(), row()])
        with self.assertRaisesRegex(ValueError, "duplicate CustomerId"):
            load_customers(self.base)

    def test_missing_column_rejected(self):
        self.base.write_text("CustomerId,Surname\n100001,Test\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing"):
            load_customers(self.base)

    def test_duplicate_header_rejected(self):
        self.base.write_text("CustomerId,CustomerId\n100001,100002\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            load_customers(self.base)

    def test_csv_with_extra_fields_is_rejected(self):
        with self.base.open("a", encoding="utf-8") as handle:
            handle.write(",extra\n")
        with self.assertRaisesRegex(ValueError, "wrong number of CSV fields"):
            load_customers(self.base)

    def test_csv_with_missing_fields_is_rejected(self):
        with self.base.open("a", encoding="utf-8") as handle:
            handle.write("100003,OnlyName\n")
        with self.assertRaisesRegex(ValueError, "wrong number of CSV fields"):
            load_customers(self.base)

    def test_customer_csv_requires_utf8(self):
        self.base.write_bytes(b"CustomerId,Surname\n100001,\xff\n")
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            load_customers(self.base)

    def test_invalid_binary_column_rejected(self):
        write_csv(self.base, [row(Exited="maybe")])
        with self.assertRaisesRegex(ValueError, "Exited must be 0 or 1"):
            load_customers(self.base)

    def test_invalid_credit_score_rejected(self):
        write_csv(self.base, [row(CreditScore="670.1")])
        with self.assertRaisesRegex(ValueError, "CreditScore"):
            load_customers(self.base)

    def test_nonfinite_amount_rejected(self):
        for amount in ("NaN", "Infinity", "-2"):
            with self.subTest(amount=amount):
                write_csv(self.base, [row(Balance=amount)])
                with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                    load_customers(self.base)

    def test_invalid_sentiment_rejected(self):
        write_csv(self.enriched, [row(Sentiment_Label="Unsure")], sentiment=True)
        with self.assertRaisesRegex(ValueError, "unknown Sentiment_Label"):
            load_customers(self.enriched)

    def test_blank_customer_name_rejected(self):
        write_csv(self.base, [row(Surname=" ")])
        with self.assertRaisesRegex(ValueError, "missing Surname"):
            load_customers(self.base)

    def test_missing_reference_document_rejected(self):
        (self.docs / REFERENCE_FILES[0]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing reference document"):
            inspect_documents(self.docs)

    def test_bad_document_encoding_rejected(self):
        (self.docs / REFERENCE_FILES[0]).write_bytes(b"\xff\xfe\xfe")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            inspect_documents(self.docs)

    def test_empty_document_rejected(self):
        (self.docs / REFERENCE_FILES[0]).write_text(" \n")
        with self.assertRaisesRegex(ValueError, "blank"):
            inspect_documents(self.docs)

    def test_nul_document_rejected(self):
        (self.docs / REFERENCE_FILES[0]).write_text("Hello\x00world")
        with self.assertRaisesRegex(ValueError, "NUL"):
            inspect_documents(self.docs)

    def test_document_digest_has_real_length(self):
        report = inspect_documents(self.docs)
        first = report[0]
        self.assertEqual(first["characters"], len("Fee information: ₹500.\n"))
        self.assertEqual(len(first["sha256"]), 64)
        self.assertNotIn("Fee information", json.dumps(report))

    def test_cli_reports_counts_without_profiles(self):
        command = [
            sys.executable, str(ROOT / "scripts/check_sources.py"),
            "--data-dir", str(self.root), "--docs-dir", str(self.docs),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["customer_records"], 2)
        self.assertNotIn("Test", result.stdout)

    def test_cli_exits_on_invalid_sources(self):
        self.base.unlink()
        command = [
            sys.executable, str(ROOT / "scripts/check_sources.py"),
            "--data-dir", str(self.root), "--docs-dir", str(self.docs),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Source check failed", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
