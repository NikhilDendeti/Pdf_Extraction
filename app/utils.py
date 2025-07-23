def deduplicate_outline(outline):
    seen = set()
    unique = []
    for entry in outline:
        key = (entry['level'], entry['text'], entry['page'])
        if key not in seen:
            unique.append(entry)
            seen.add(key)
    return unique

def sort_outline(outline):
    return sorted(outline, key=lambda x: (x["page"], x["level"]))
