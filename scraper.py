import argparse
import html as _html
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import OCR module (should be in the same directory)
try:
    from ocr import ocr_one_image
except ImportError:
    print("Warning: OCR module not found. Captcha solving will fail.")
    ocr_one_image = None


# =============================================================================
# Configuration & Constants
# =============================================================================

@dataclass
class ClassConfig:
    """Configuration derived from class value."""
    class_value: int
    course: str      # SSC or HSSC
    exam_type: str   # 1 or 2
    
    @classmethod
    def from_class(cls, class_value: int) -> "ClassConfig":
        """
        Create configuration from class value.
        
        Class 9, 10 -> SSC
        Class 11, 12 -> HSSC
        Class 9, 11 -> exam_type 1 (Part-I)
        Class 10, 12 -> exam_type 2 (Part-II)
        """
        if class_value not in (9, 10, 11, 12):
            raise ValueError(f"Class must be 9, 10, 11, or 12. Got: {class_value}")
        
        course = "SSC" if class_value in (9, 10) else "HSSC"
        exam_type = "1" if class_value in (9, 11) else "2"
        
        return cls(class_value=class_value, course=course, exam_type=exam_type)


BASE_URL = os.environ.get("BASE_URL", "")

# =============================================================================
# Database Schema & Operations
# =============================================================================

def get_db_filename(year: int, class_value: int, start_roll: int, end_roll: int) -> str:
    """Generate database filename in the required format."""
    return f"{year}_class_{class_value}_{start_roll}_{end_roll}.db"


def create_database(db_path: str) -> sqlite3.Connection:
    """
    Create SQLite database with optimized, search-friendly schema.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            RollNo TEXT PRIMARY KEY,
            RegistrationNo TEXT,
            Name TEXT,
            FatherName TEXT,
            Cnic TEXT,
            FatherCnic TEXT,
            DOB TEXT,
            "Group" TEXT,
            Institute TEXT,
            TotalMarks TEXT,
            MarksObtainedLine TEXT,
            ResultSentence TEXT,
            PicPath TEXT,
            Photo BLOB
        )
    """)
    
    # Create indexes for fast searching
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_RegistrationNo ON students(RegistrationNo)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_Name ON students(Name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_FatherName ON students(FatherName)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_FatherCnic ON students(FatherCnic)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_Group ON students(\"Group\")")
    
    conn.commit()
    return conn


def insert_result(conn: sqlite3.Connection, roll_number: str, result_data: dict, 
                  image_blob: Optional[bytes] = None):
    """Insert or update a result record."""
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO students (
            RollNo, RegistrationNo, Name, FatherName, FatherCnic, DOB,
            "Group", Institute, Cnic,
            TotalMarks, MarksObtainedLine, ResultSentence,
            PicPath, Photo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(RollNo) DO UPDATE SET
            RegistrationNo = excluded.RegistrationNo,
            Name = excluded.Name,
            FatherName = excluded.FatherName,
            FatherCnic = excluded.FatherCnic,
            DOB = excluded.DOB,
            "Group" = excluded."Group",
            Institute = excluded.Institute,
            Cnic = excluded.Cnic,
            TotalMarks = excluded.TotalMarks,
            MarksObtainedLine = excluded.MarksObtainedLine,
            ResultSentence = excluded.ResultSentence,
            PicPath = excluded.PicPath,
            Photo = excluded.Photo
    """, (
        roll_number,
        result_data.get('registration_no', ''),
        result_data.get('student_name', ''),
        result_data.get('father_name', ''),
        result_data.get('father_nic', ''),
        result_data.get('dob', ''),
        result_data.get('group', ''),
        result_data.get('institution_district', ''),
        result_data.get('bfarm', ''),
        result_data.get('total_max_marks', ''),
        result_data.get('marks_obtained_line', ''),
        result_data.get('result_sentence', ''),
        result_data.get('image_path', ''),
        image_blob
    ))
    
    conn.commit()


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get statistics about the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM students")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM students WHERE Name IS NOT NULL AND Name != ''")
    success = cursor.fetchone()[0]
    return {
        'total': total,
        'success': success,
        'not_found': total - success
    }


# =============================================================================
# HTML Parsing Utilities
# =============================================================================

def _truncate_png_bytes(b: bytes) -> bytes:
    """Truncate PNG to valid payload (removes appended HTML)."""
    iend = b.rfind(b"IEND")
    if iend == -1:
        return b
    end = iend + 8
    return b[:end] if end <= len(b) else b


def _iter_input_tags(html: str) -> list[str]:
    """Extract all input tags from HTML."""
    return re.findall(r"<input\b[^>]*>", html, flags=re.IGNORECASE)


def _parse_attrs(tag_html: str) -> dict[str, str]:
    """Parse HTML attributes from a tag."""
    attrs = {}
    for k, v in re.findall(r'([a-zA-Z_:][\w:.-]*)\s*=\s*(".*?"|\'.*?\')', tag_html):
        attrs[k.lower()] = v[1:-1]
    return attrs


def _extract_all_hidden_fields(html: str) -> dict[str, str]:
    """Extract all hidden input fields from the page."""
    out: dict[str, str] = {}
    for tag in _iter_input_tags(html):
        attrs = _parse_attrs(tag)
        if attrs.get("type", "").lower() != "hidden":
            continue
        name = attrs.get("name")
        if name:
            out[name] = attrs.get("value", "")
    return out


def _extract_form_action(html: str) -> str:
    """Extract form action URL."""
    m = re.search(r'<form\b[^>]*\bmethod="post"[^>]*>', html, flags=re.IGNORECASE)
    if not m:
        return ""
    attrs = _parse_attrs(m.group(0))
    return attrs.get("action", "") or ""


def _extract_select_options(html: str, name: str) -> list[tuple[str, str]]:
    """Extract options from a select element."""
    m = re.search(
        rf'<select\b[^>]*\bname="{re.escape(name)}"[^>]*>(.*?)</select>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []
    body = m.group(1)
    opts: list[tuple[str, str]] = []
    for om in re.finditer(r"<option\b[^>]*>(.*?)</option>", body, flags=re.IGNORECASE | re.DOTALL):
        tag = om.group(0)
        attrs = _parse_attrs(tag)
        value = attrs.get("value", "")
        label = re.sub(r"<[^>]+>", "", om.group(1)).strip()
        opts.append((value, label))
    return opts


def _extract_submit_name(html: str) -> str:
    """Find the submit button name."""
    for tag in _iter_input_tags(html):
        attrs = _parse_attrs(tag)
        if attrs.get("type", "").lower() not in ("submit", "image", "button"):
            continue
        val = (attrs.get("value") or "").strip().lower()
        if "view result" in val or val == "view":
            return attrs.get("name", "") or ""
    return ""


def _extract_by_id_label(html: str, element_id: str) -> str:
    """Extract text from a label element by ID."""
    m = re.search(
        rf'<label\b[^>]*\bid="{re.escape(element_id)}"[^>]*>(.*?)</label>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ""
    txt = re.sub(r"<[^>]+>", "", m.group(1))
    return _html.unescape(txt).strip()


def _extract_total_row(html: str) -> dict[str, str]:
    """Extract TOTAL row values from the result table."""
    m = re.search(
        r"<tr\b[^>]*>\s*<td\b[^>]*>\s*TOTAL\s*</td>([\s\S]*?)</tr>",
        html,
        flags=re.IGNORECASE,
    )
    if not m:
        return {"total_max_marks": "", "marks_obtained_line": ""}
    row_html = m.group(0)
    tds = re.findall(r"<td\b[^>]*>([\s\S]*?)</td>", row_html, flags=re.IGNORECASE)
    texts = [_html.unescape(re.sub(r"<[^>]+>", "", td)).strip() for td in tds]
    total_max = texts[1] if len(texts) > 1 else ""
    marks_line = texts[2] if len(texts) > 2 else ""
    return {"total_max_marks": total_max, "marks_obtained_line": marks_line}


def _extract_image_path(html: str) -> str:
    """Extract image path from JavaScript."""
    m = re.search(r"Checkbrowser\('([^']+)'\)", html, flags=re.IGNORECASE)
    return (m.group(1) if m else "").strip()


def extract_result_fields(html: str) -> dict[str, str]:
    """Extract all result fields from the HTML response."""
    out: dict[str, str] = {
        "registration_no": _extract_by_id_label(html, "lblRegNum"),
        "group": _extract_by_id_label(html, "lblGroup"),
        "student_name": _extract_by_id_label(html, "Name"),
        "bfarm": _extract_by_id_label(html, "lblBFARM"),
        "father_name": _extract_by_id_label(html, "lblFatherName"),
        "father_nic": _extract_by_id_label(html, "lblFatherNIC"),
        "dob": _extract_by_id_label(html, "lblDOB"),
        "institution_district": _extract_by_id_label(html, "lblExamCenter"),
        "result_sentence": _extract_by_id_label(html, "lblResultinSentences"),
        "image_path": _extract_image_path(html),
    }
    out.update(_extract_total_row(html))
    return out


# =============================================================================
# Result Fetcher
# =============================================================================

class ResultFetcher:
    
    def __init__(self, config: ClassConfig, year: int):
        self.config = config
        self.year = str(year)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        self.captcha_path = Path("captcha_temp.png")
    
    def _get_page_and_captcha(self) -> tuple[str, dict, str, bytes]:
        """
        Get the main page, extract hidden fields, and download captcha.
        Returns: (page_html, hidden_fields, post_url, captcha_bytes)
        """
        # Get main page
        self.session.headers["Referer"] = BASE_URL
        resp = self.session.get(BASE_URL, timeout=30)
        resp.raise_for_status()
        html = resp.text
        
        # Extract hidden fields and form action
        hidden = _extract_all_hidden_fields(html)
        action = _extract_form_action(html)
        post_url = urljoin(BASE_URL, action) if action else BASE_URL
        
        # Find and download captcha
        m = re.search(r'src="(/Captcha\.aspx\?\d+)"', html)
        if not m:
            raise RuntimeError("Could not find captcha image URL on page")
        
        captcha_url = BASE_URL.rstrip("/") + m.group(1)
        r = self.session.get(
            captcha_url,
            timeout=30,
            headers={
                "Referer": BASE_URL,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Cache-Control": "no-cache",
            },
        )
        r.raise_for_status()
        
        captcha_bytes = r.content
        if not captcha_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Captcha download did not return a PNG")
        
        captcha_bytes = _truncate_png_bytes(captcha_bytes)
        
        return html, hidden, post_url, captcha_bytes
    
    def _solve_captcha(self, captcha_bytes: bytes) -> str:
        """Save captcha and solve using OCR."""
        if ocr_one_image is None:
            raise RuntimeError("OCR module not available")
        
        self.captcha_path.write_bytes(captcha_bytes)
        return ocr_one_image(str(self.captcha_path))
    
    def _submit_form(self, roll_number: str, hidden: dict, post_url: str, 
                     captcha_text: str, html: str) -> str:
        """Submit the form and return response HTML."""
        submit_name = _extract_submit_name(html) or "Button1"
        
        payload = {k: v for k, v in hidden.items() if v != ""}
        payload.update({
            "rdlistCourse": self.config.course,
            "txtFormNo": roll_number,
            "ddlExamType": self.config.exam_type,
            "ddlExamYear": self.year,
            "txtCaptcha": captcha_text,
            submit_name: "View Result",
        })
        
        post_headers = {
            "Origin": BASE_URL.rstrip("/"),
            "Referer": BASE_URL,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        
        resp = self.session.post(post_url, data=payload, headers=post_headers, timeout=60)
        resp.raise_for_status()
        
        # Return both text and final URL (after any redirects)
        return resp.text, resp.url
    
    def _download_image(self, image_path: str) -> Optional[bytes]:
        """Download student image if available."""
        if not image_path:
            return None
        
        try:
            # Use the ImageLoader.aspx endpoint with image_path as query parameter
            image_url = f"{BASE_URL}ImageLoader.aspx?Updateid={image_path}"
            resp = self.session.get(image_url, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception:
            return None
    
    def fetch_result(self, roll_number: str, max_retries: int = 5) -> tuple[dict, Optional[bytes], str, str]:
        """
        Fetch result for a single roll number.
        
        Returns: (result_data, image_blob, status, error_message)
        Status can be: 'success', 'not_found', 'error'
        """
        last_error = ""
        for attempt in range(max_retries):
            try:
                # Get page and captcha
                html, hidden, post_url, captcha_bytes = self._get_page_and_captcha()
                
                # Solve captcha
                captcha_text = self._solve_captcha(captcha_bytes)
                if not captcha_text:
                    last_error = "OCR failed to read captcha"
                    time.sleep(0.5)
                    continue  # Retry with new captcha
                
                # Submit form
                response_html, response_url = self._submit_form(roll_number, hidden, post_url, captcha_text, html)
                
                # Check if redirected to Error.aspx (student not found)
                if "Error.aspx" in response_url:
                    return {}, None, 'not_found', 'Student not found'
                
                # Check for common errors
                lower = response_html.lower()
                if "invalid captcha" in lower or ("captcha" in lower and "error" in lower):
                    last_error = "Invalid captcha"
                    time.sleep(0.5)
                    continue  # Retry with new captcha
                
                # Parse result
                result_data = extract_result_fields(response_html)
                
                # Check if we got valid data
                if not any(result_data.get(k) for k in ('student_name', 'registration_no', 'result_sentence')):
                    last_error = "Failed to parse result data"
                    time.sleep(0.5)
                    continue  # Retry
                
                # Download image
                image_blob = self._download_image(result_data.get('image_path', ''))
                
                return result_data, image_blob, 'success', None
                
            except Exception as e:
                last_error = str(e)
                time.sleep(1 + attempt * 0.5)  # Exponential backoff
                continue
        
        return {}, None, 'error', f'Max retries exceeded ({last_error})'
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.captcha_path.exists():
            self.captcha_path.unlink()


# =============================================================================
# Main Functions
# =============================================================================

def validate_inputs(start_roll: int, end_roll: int, class_value: int, year: int):
    """Validate all input parameters."""
    if not (100000 <= start_roll <= 999999):
        raise ValueError(f"Start roll must be a 6-digit number (100000-999999). Got: {start_roll}")
    if not (100000 <= end_roll <= 999999):
        raise ValueError(f"End roll must be a 6-digit number (100000-999999). Got: {end_roll}")
    if start_roll > end_roll:
        raise ValueError(f"Start roll ({start_roll}) must be <= end roll ({end_roll})")
    if class_value not in (9, 10, 11, 12):
        raise ValueError(f"Class must be 9, 10, 11, or 12. Got: {class_value}")
    if not (1900 <= year <= 2100):
        raise ValueError(f"Year must be a 4-digit year (1900-2100). Got: {year}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch results and store in SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py --start-roll 590580 --end-roll 590590 --class 11 --year 2021
        """
    )
    
    parser.add_argument("--start-roll", type=int, required=True, help="Starting 6-digit roll number")
    parser.add_argument("--end-roll", type=int, required=True, help="Ending 6-digit roll number")
    parser.add_argument("--class", dest="class_value", type=int, required=True, choices=[9, 10, 11, 12], help="Class: 9, 10, 11, or 12")
    parser.add_argument("--year", type=int, required=True, help="4-digit examination year")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the database")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests in seconds")
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    
    # Validate BASE_URL environment variable
    if not BASE_URL or "http" not in BASE_URL:
        print("Error: BASE_URL environment variable is not configured.", file=sys.stderr)
        print("Please set BASE_URL to the result portal URL.", file=sys.stderr)
        return 1
    
    args = parse_arguments()
    
    # Validate inputs
    try:
        validate_inputs(args.start_roll, args.end_roll, args.class_value, args.year)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Get configuration from class
    config = ClassConfig.from_class(args.class_value)
    
    # Generate database path
    db_filename = get_db_filename(args.year, args.class_value, args.start_roll, args.end_roll)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / db_filename
    
    print("=" * 60)
    print("Result Scraper")
    print("=" * 60)
    print(f"Roll Range:  {args.start_roll} - {args.end_roll}")
    print(f"Total:       {args.end_roll - args.start_roll + 1} students")
    print(f"Class:       {args.class_value}")
    print(f"Course:      {config.course}")
    print(f"Exam Type:   {config.exam_type} ({'Part-I' if config.exam_type == '1' else 'Part-II'})")
    print(f"Year:        {args.year}")
    print(f"Database:    {db_path}")
    print("=" * 60)
    
    # Create database
    conn = create_database(str(db_path))
    
    # Create fetcher
    fetcher = ResultFetcher(config, args.year)
    
    # Fetch results one by one
    total = args.end_roll - args.start_roll + 1
    success_count = 0
    not_found_count = 0
    error_count = 0
    failed_rolls = []  # Track failed rolls for retry
    
    try:
        for i, roll in enumerate(range(args.start_roll, args.end_roll + 1), 1):
            roll_str = str(roll).zfill(6)
            print(f"\n[{i}/{total}] Fetching roll {roll_str}...", end=" ", flush=True)
            
            result_data, image_blob, status, error_msg = fetcher.fetch_result(roll_str)
            
            if status == 'success':
                # Insert into database only on success
                insert_result(conn, roll_str, result_data, image_blob)
                success_count += 1
                name = result_data.get('student_name', 'Unknown')
                marks = result_data.get('marks_obtained_line', '')
                print(f"✓ {name} | {marks}")
            elif status == 'not_found':
                not_found_count += 1
                print("✗ Not Found")
            else:
                failed_rolls.append(roll_str)
                print(f"✗ Error: {error_msg}")
            
            # Delay between requests
            if i < total:
                time.sleep(args.delay)
        
        # Retry failed rolls
        if failed_rolls:
            print(f"\n\n{'='*60}")
            print(f"Retrying {len(failed_rolls)} failed roll(s)...")
            print("="*60)
            
            still_failed = []
            for i, roll_str in enumerate(failed_rolls, 1):
                print(f"\n[Retry {i}/{len(failed_rolls)}] Fetching roll {roll_str}...", end=" ", flush=True)
                
                # Wait a bit longer before retry
                time.sleep(1)
                
                result_data, image_blob, status, error_msg = fetcher.fetch_result(roll_str, max_retries=7)
                
                if status == 'success':
                    insert_result(conn, roll_str, result_data, image_blob)
                    success_count += 1
                    name = result_data.get('student_name', 'Unknown')
                    marks = result_data.get('marks_obtained_line', '')
                    print(f"✓ {name} | {marks}")
                elif status == 'not_found':
                    not_found_count += 1
                    print("✗ Not Found")
                else:
                    still_failed.append(roll_str)
                    print(f"✗ Error: {error_msg}")
            
            error_count = len(still_failed)
            if still_failed:
                print(f"\n⚠ Still failed after retry: {', '.join(still_failed)}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    
    finally:
        fetcher.cleanup()
        
        # Print final stats
        stats = get_stats(conn)
        print("\n" + "=" * 60)
        print("Final Statistics:")
        print(f"  Total:     {stats['total']}")
        print(f"  Success:   {stats['success']}")
        print(f"  Not Found: {not_found_count}")
        print(f"  Errors:    {error_count}")
        print("=" * 60)
        print(f"Database saved: {db_path}")
        
        conn.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
