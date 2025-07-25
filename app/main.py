import json
from pathlib import Path
from extractor import extract_text_with_fonts, extract_bookmarks
from heuristics import identify_title_and_headings
from utils import deduplicate_outline, sort_outline

INPUT_DIR = Path("app/pdfs")
OUTPUT_DIR = Path("app/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_outline_extractor(pdf_path: Path, output_path: Path):
    lines = extract_text_with_fonts(str(pdf_path))
    title, outline = identify_title_and_headings(lines)

    if not outline:
        outline = extract_bookmarks(str(pdf_path))

    outline = deduplicate_outline(outline)
    outline = sort_outline(outline)

    result = {
        "title": title,
        "outline": outline
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted: {pdf_path.name} -> {output_path.name}")

if __name__ == "__main__":
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in:", INPUT_DIR)
        exit(1)

    for pdf_file in pdf_files:
        output_file = OUTPUT_DIR / (pdf_file.stem + ".json")
        run_outline_extractor(pdf_file, output_file)

    print("🎉 Done extracting all PDFs.")
