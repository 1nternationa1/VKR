import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "valuation_prompt.md"


@lru_cache(maxsize=1)
def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found at {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(property_data: Dict[str, Any]) -> str:
    template = load_prompt_template()
    formatted = json.dumps(property_data, ensure_ascii=False, indent=2)
    return template.replace("{{PROPERTY_JSON}}", formatted)
