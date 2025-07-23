# import fitz  

# def extract_text_with_fonts(pdf_path, y_threshold=5):
#     doc = fitz.open(pdf_path)
#     merged_content = []

#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         blocks = page.get_text("dict")["blocks"]

#         spans = []
#         for block in blocks:
#             if "lines" not in block:
#                 continue
#             for line in block["lines"]:
#                 for span in line["spans"]:
#                     text = span["text"].strip()
#                     if not text:
#                         continue
#                     spans.append({
#                         "text": text,
#                         "size": round(span["size"]),
#                         "font": span["font"],
#                         "bold": "Bold" in span["font"],
#                         "page": page_num + 1,
#                         "x": span["bbox"][0],
#                         "y": span["bbox"][1]
#                     })

#         spans.sort(key=lambda s: (s["y"], s["x"]))

#         blocks = []
#         current_block = []

#         for span in spans:
#             if not current_block:
#                 current_block.append(span)
#                 continue

#             prev = current_block[-1]
#             same_line = (
#                 abs(span["y"] - prev["y"]) <= y_threshold and
#                 span["page"] == prev["page"]
#             )
#             x_ordered = span["x"] >= prev["x"]

#             if same_line and x_ordered:
#                 current_block.append(span)
#             else:
#                 blocks.append(current_block)
#                 current_block = [span]

#         if current_block:
#             blocks.append(current_block)

#         for block in blocks:
#             block.sort(key=lambda s: s["x"])
#             merged_text = " ".join([s["text"] for s in block]).strip()
#             if merged_text:
#                 merged_content.append({
#                     "text": merged_text,
#                     "size": block[0]["size"],  
#                     "font": block[0]["font"],
#                     "bold": block[0]["bold"],
#                     "page": block[0]["page"],
#                     "y": block[0]["y"]
#                 })

#     return merged_content


# def is_heading_candidate(text, size, max_words=10):
#     return len(text.split()) <= max_words and len(text.strip()) >= 3 and size >= 10


# def generate_outline_from_blocks(blocks):
#     candidates = [b for b in blocks if is_heading_candidate(b["text"], b["size"])]

#     size_freq = Counter([b["size"] for b in candidates])
#     sorted_sizes = sorted(size_freq.items(), key=lambda x: -x[0])  

#     size_to_level = {}
#     if len(sorted_sizes) > 0:
#         size_to_level[sorted_sizes[0][0]] = "H1"
#     if len(sorted_sizes) > 1:
#         size_to_level[sorted_sizes[1][0]] = "H2"
#     if len(sorted_sizes) > 2:
#         size_to_level[sorted_sizes[2][0]] = "H3"

#     outline = []
#     for b in candidates:
#         level = size_to_level.get(b["size"])
#         if level:
#             outline.append({
#                 "level": level,
#                 "text": b["text"],
#                 "page": b["page"]
#             })

#     return {
#         "title": "Untitled Document",
#         "outline": outline
#     }


# def extract_outline(pdf_path):
#     blocks = extract_text_with_fonts(pdf_path)
#     return generate_outline_from_blocks(blocks)


# def extract_bookmarks(pdf_path):
#     doc = fitz.open(pdf_path)
#     toc = doc.get_toc(simple=True)  
#     outline = []
#     for level, title, page in toc:
#         if level > 3:
#             continue
#         outline.append({
#             "level": f"H{level}",
#             "text": title.strip(),
#             "page": page
#         })
#     return outline



import fitz 

def extract_text_with_fonts(pdf_path, y_threshold=5):
    doc = fitz.open(pdf_path)
    merged_content = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        spans = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    spans.append({
                        "text": text,
                        "size": round(span["size"]),
                        "font": span["font"],
                        "bold": "Bold" in span["font"],
                        "page": page_num + 1,
                        "x": span["bbox"][0],
                        "y": span["bbox"][1]
                    })

        spans.sort(key=lambda s: (s["y"], s["x"]))

        blocks = []
        current_block = []

        for span in spans:
            if not current_block:
                current_block.append(span)
                continue

            prev = current_block[-1]
            same_line = abs(span["y"] - prev["y"]) <= y_threshold and span["page"] == prev["page"]
            x_ordered = span["x"] >= prev["x"]

            if same_line and x_ordered:
                current_block.append(span)
            else:
                blocks.append(current_block)
                current_block = [span]

        if current_block:
            blocks.append(current_block)

        for block in blocks:
            block.sort(key=lambda s: s["x"])
            merged_text = " ".join([s["text"] for s in block]).strip()
            if merged_text:
                merged_content.append({
                    "text": merged_text,
                    "size": block[0]["size"],
                    "font": block[0]["font"],
                    "bold": block[0]["bold"],
                    "page": block[0]["page"],
                    "y": block[0]["y"]
                })

    return merged_content


def extract_bookmarks(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)
    outline = []
    for level, title, page in toc:
        if level > 3:
            continue
        outline.append({
            "level": f"H{level}",
            "text": title.strip(),
            "page": page
        })
    return outline
