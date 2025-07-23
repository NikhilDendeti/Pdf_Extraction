from collections import Counter

def clean_text(text):
    return text.strip("0123456789.-– ").strip()

def identify_title_and_headings(lines):
    sizes = [line["size"] for line in lines if len(line["text"].split()) > 2]
    size_freq = Counter(sizes)
    sorted_sizes = sorted(size_freq.items(), key=lambda x: -x[0])
    top_sizes = [size for size, _ in sorted_sizes[:4]]

    title = None
    outline = []

    for line in lines:
        text = line["text"]
        size = line["size"]
        page = line["page"]

        if page == 1 and size == top_sizes[0] and not title:
            title = clean_text(text)
            continue

        level = None
        if size == top_sizes[0]:
            level = "H1"
        elif size == top_sizes[1]:
            level = "H2"
        elif size == top_sizes[2]:
            level = "H3"

        if level:
            clean = clean_text(text)
            if len(clean.split()) >= 2 and not clean.lower().startswith("table of"):
                outline.append({
                    "level": level,
                    "text": clean,
                    "page": page
                })

    return title or "Untitled Document", outline
