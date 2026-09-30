import json
from pathlib import Path

def test_golden_set_has_40_unique_questions_and_page_metadata():
    path = Path(__file__).parent / "golden_qa.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == 40
    assert len({row["id"] for row in rows}) == 40
    assert all(row["question"] and row["reference_answer"] for row in rows)
    assert all(row["expected_pages"] for row in rows)
