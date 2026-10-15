import io
import subprocess
import unittest
from datetime import date
from unittest.mock import patch

from PIL import Image

from identity_ocr import (
    IdentityOCRUnavailable,
    check_identity_document,
    extract_labeled_date_of_birth,
    name_appears_in_text,
    verify_document_text,
)


class IdentityOCRTests(unittest.TestCase):
    def test_extracts_date_only_when_labeled_as_date_of_birth(self):
        self.assertEqual(
            extract_labeled_date_of_birth("Date of Birth: 13/10/1990"),
            date(1990, 10, 13),
        )
        self.assertEqual(
            extract_labeled_date_of_birth("DOB 1990-10-13"),
            date(1990, 10, 13),
        )
        self.assertIsNone(extract_labeled_date_of_birth("Issue date: 13/10/1990"))

    def test_exact_normalized_name_match(self):
        self.assertTrue(
            name_appears_in_text(
                "Asha Devi",
                "GOVERNMENT OF INDIA\nAsha Devi\nDOB 01/01/1990",
            )
        )
        self.assertFalse(
            name_appears_in_text(
                "Asha Devi",
                "GOVERNMENT OF INDIA\nAsha Dey\nDOB 01/01/1990",
            )
        )

    def test_verification_requires_name_dob_and_minimum_age(self):
        dob = date(date.today().year - 30, 1, 1)
        matched = verify_document_text(
            f"Government of India\nAlex Example\nDate of Birth: {dob:%d/%m/%Y}",
            "Alex Example",
            dob,
        )
        self.assertEqual(
            matched,
            {
                "name_matches": True,
                "date_of_birth_matches": True,
                "meets_minimum_age": True,
            },
        )
        mismatch = verify_document_text(
            f"Alex Example\nDOB: {dob:%d/%m/%Y}",
            "Different Person",
            dob,
        )
        self.assertFalse(mismatch["name_matches"])

    def test_admin_verification_accepts_applicants_who_are_23_or_older(self):
        today = date.today()
        age_23_dob = date(today.year - 23, 1, 1)
        age_24_dob = date(today.year - 24, 1, 1)

        age_23 = verify_document_text(
            f"Alex Example\nDOB: {age_23_dob:%d/%m/%Y}",
            "Alex Example",
            age_23_dob,
        )
        age_24 = verify_document_text(
            f"Alex Example\nDOB: {age_24_dob:%d/%m/%Y}",
            "Alex Example",
            age_24_dob,
        )

        self.assertTrue(age_23["meets_minimum_age"])
        self.assertTrue(age_24["meets_minimum_age"])

    def test_ocr_uses_in_memory_stdin_and_does_not_create_image_file(self):
        image = io.BytesIO()
        Image.new("RGB", (12, 12), "white").save(image, format="PNG")
        dob = date(date.today().year - 30, 1, 1)
        ocr_output = f"Alex Example\nDOB: {dob:%d/%m/%Y}".encode()
        with patch("identity_ocr.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=["tesseract"],
                returncode=0,
                stdout=ocr_output,
            )
            result = check_identity_document(
                image.getvalue(),
                "Alex Example",
                dob,
            )
        self.assertTrue(result["name_matches"])
        call = run.call_args
        self.assertEqual(call.kwargs["input"][:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(call.kwargs["stderr"], subprocess.DEVNULL)

    def test_missing_ocr_executable_is_reported(self):
        image = io.BytesIO()
        Image.new("RGB", (4, 4), "white").save(image, format="PNG")
        with patch(
            "identity_ocr.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            with self.assertRaises(IdentityOCRUnavailable):
                check_identity_document(
                    image.getvalue(),
                    "Alex Example",
                    date(1990, 1, 1),
                )


if __name__ == "__main__":
    unittest.main()
