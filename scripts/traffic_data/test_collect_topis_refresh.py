"""Offline tests for the public TOPIS raw-data collector."""
import io
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.traffic_data import collect_topis_refresh as collector
from scripts.traffic_data.refresh_readiness import readiness


def workbook_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
    return stream.getvalue()


class TopisCollectorTests(unittest.TestCase):
    def test_partial_year_is_recorded_and_raw_is_not_overwritten(self):
        content = workbook_bytes()
        entries = {1: {key: "x" for key in (
            "apndFileNm", "apndFilePathNm", "bdwrSeq", "blbdDivCd")}}
        with tempfile.TemporaryDirectory(dir=collector.ROOT / "artifacts") as temp:
            with patch.object(collector, "monthly_entries", return_value=entries), \
                 patch.object(collector, "post", return_value=content):
                root = Path(temp)
                first = collector.collect(2026, root)
                second = collector.collect(2026, root)
                self.assertEqual(first["months_collected"], 1)
                self.assertFalse(first["complete_year"])
                self.assertEqual(first, second)
                self.assertEqual(json.loads((root / "2026/manifest.json").read_text(encoding="utf-8")), first)
                self.assertEqual((root / "2026/01_topis_traffic_2026.xlsx").read_bytes(), content)
            with patch.object(collector, "monthly_entries", return_value=entries), \
                 patch.object(collector, "post", return_value=workbook_bytes() + b"revised"):
                with self.assertRaisesRegex(ValueError, "revised an existing raw month"):
                    collector.collect(2026, root)
                self.assertEqual((root / "2026/01_topis_traffic_2026.xlsx").read_bytes(), content)

    def test_empty_current_year_is_not_a_failed_download(self):
        with tempfile.TemporaryDirectory(dir=collector.ROOT / "artifacts") as temp:
            with patch.object(collector, "monthly_entries", return_value={}):
                result = collector.collect(2027, Path(temp))
        self.assertEqual(result["months_collected"], 0)
        self.assertFalse(result["complete_year"])

    def test_listing_rejects_duplicate_months(self):
        rows = [{"apndFileNm": f"{i}월 서울시 교통량.xlsx"} for i in (1, 1)]
        with patch.object(collector, "post", return_value=json.dumps({"rows": rows}).encode()):
            with self.assertRaisesRegex(ValueError, "more than one workbook"):
                collector.monthly_entries(2026)

    def test_readiness_never_approves_only_topis(self):
        with tempfile.TemporaryDirectory(dir=collector.ROOT / "artifacts") as temp:
            root = Path(temp)
            self.assertFalse(readiness(2026, root)["ready_for_preprocessing"])
            directory = root / "topis" / "2026"
            directory.mkdir(parents=True)
            records = []
            for month in range(1, 13):
                name = f"{month:02d}.xlsx"
                content = f"month={month}".encode()
                (directory / name).write_bytes(content)
                records.append({"month": month, "file": name,
                                "sha256": hashlib.sha256(content).hexdigest()})
            (directory / "manifest.json").write_text(json.dumps({
                "year": 2026, "complete_year": True, "records": records,
            }), encoding="utf-8")
            result = readiness(2026, root)
            self.assertTrue(result["sources"]["topis"]["ready"])
            self.assertFalse(result["sources"]["accidents"]["ready"])
            self.assertIn("accidents", result["collectors_not_implemented"])
            self.assertFalse(result["ready_for_preprocessing"])
            (directory / "01.xlsx").write_bytes(b"tampered")
            self.assertFalse(readiness(2026, root)["sources"]["topis"]["ready"])


if __name__ == "__main__":
    unittest.main()
