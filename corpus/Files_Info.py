import json
from collections import Counter
from pathlib import Path


def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, file_path):
    """Save data to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def get_clause_records(document):
    """Return the list of clause records from a processed document payload."""
    if isinstance(document, dict) and "clauses" in document:
        return document["clauses"]
    if isinstance(document, list):
        return document
    raise TypeError(f"Unsupported document payload type: {type(document).__name__}")


def count_records(data):
    """Count total number of records in a clause payload."""
    return len(get_clause_records(data))


def filter_by_value(data, key, value):
    """Filter clause records where record[key] == value."""
    return [record for record in get_clause_records(data) if record.get(key) == value]


def search_contains(data, key, text):
    """Search for text inside a field, case-insensitive."""
    needle = text.lower()
    return [
        record
        for record in get_clause_records(data)
        if needle in str(record.get(key, "")).lower()
    ]


def count_by_field(data, key):
    """Count how many times each value appears for a field."""
    values = [record.get(key) for record in get_clause_records(data) if key in record]
    return Counter(values)


def filter_numeric(data, key, operator, value):
    """Numeric filtering for clause records."""
    operations = {
        ">": lambda x: x > value,
        ">=": lambda x: x >= value,
        "<": lambda x: x < value,
        "<=": lambda x: x <= value,
        "==": lambda x: x == value,
        "!=": lambda x: x != value,
    }

    if operator not in operations:
        raise ValueError(f"Unsupported operator: {operator}")

    result = []
    for record in get_clause_records(data):
        field_value = record.get(key)
        if isinstance(field_value, (int, float)) and operations[operator](field_value):
            result.append(record)

    return result


# =========================
# MAIN
# =========================

root = Path(__file__).resolve().parents[1]
file_path = root / "corpus" / "processed" / "clauses" / "draft-ietf-tls-tls13-20.json"
data = load_json(file_path)
clauses = get_clause_records(data)

print("Total records:", count_records(data))

normative = filter_by_value(clauses, key="classification", value="NORMATIVE")
print("\nNormative count:", len(normative))
print("Sample normative clause:")
if normative:
    print(normative[0])

classification_counts = count_by_field(data, "classification")
print("\nClassification counts:")
for classification, count in classification_counts.items():
    print(f"{classification}: {count}")

# Example search
matches = search_contains(data, "section_heading", "requirements")
print(f"\nMatches for 'requirements': {len(matches)}")
if matches:
    print(matches[0])

save_json(normative[:10], root / "corpus" / "processed" / "filtered.json")