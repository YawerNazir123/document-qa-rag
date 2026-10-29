"""Document loaders.

Two sources:

1. The Kashmir Birds field guide — one text document per species, built
   from ``bird_database/*/info.json`` (common name, scientific name,
   Kashmiri name, family, size, migratory status, region, habitat,
   description). This is the built-in corpus for this project.
2. Upload-your-own documents — ``.pdf`` (via pypdf), ``.txt`` and
   ``.md`` files from any folder.
"""

from __future__ import annotations

import json
from pathlib import Path

Document = dict  # {"text": str, "metadata": dict}


def _bird_text(info: dict) -> str:
    """Render one species' info.json as a plain-text document."""
    lines = [f"{info.get('common_name', '')} ({info.get('scientific_name', '')})"]
    kashmiri = info.get("local_name_kashmiri") or info.get("local_name") or ""
    if kashmiri:
        lines.append(f"Kashmiri name: {kashmiri}.")
    for label, key in [
        ("Family", "family"),
        ("Genus", "genus"),
        ("Category", "category"),
        ("Region", "region"),
        ("Migratory status", "migratory_status"),
        ("Habitat", "habitat"),
        ("Length", "length"),
        ("Wingspan", "wingspan"),
        ("Weight", "weight"),
        ("Size", "size"),
        ("Conservation status", "conservation_status"),
        ("IUCN status", "iucn_status"),
    ]:
        value = (info.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}.")
    description = (info.get("description") or "").strip()
    if description:
        lines.append(description)
    return "\n".join(lines)


def load_bird_corpus(bird_database_dir: str | Path) -> list[Document]:
    """Load every species in ``bird_database/`` as one document each.

    Skips folders with a missing or unreadable ``info.json`` and records
    how many were skipped in the returned documents' metadata is not
    possible, so callers should compare counts against the folder count.
    """
    docs: list[Document] = []
    root = Path(bird_database_dir)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        info_path = folder / "info.json"
        if not info_path.exists():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        common = (info.get("common_name") or folder.name).strip()
        if not common:
            continue
        docs.append(
            {
                "text": _bird_text(info),
                "metadata": {
                    "source": f"bird_database/{folder.name}/info.json",
                    "common_name": common,
                    "scientific_name": (info.get("scientific_name") or "").strip(),
                    "kashmiri_name": (
                        info.get("local_name_kashmiri") or info.get("local_name") or ""
                    ).strip(),
                    "family": (info.get("family") or "").strip(),
                    "page": 1,
                },
            }
        )
    return docs


def load_text_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "text": text,
        "metadata": {"source": path.name, "common_name": "", "scientific_name": "", "page": 1},
    }


def load_pdf_file(path: Path) -> list[Document]:
    """One document per PDF page, so page numbers survive into citations."""
    from pypdf import PdfReader

    docs: list[Document] = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            {
                "text": text,
                "metadata": {
                    "source": path.name,
                    "common_name": "",
                    "scientific_name": "",
                    "page": i,
                },
            }
        )
    return docs


def load_documents(folder: str | Path) -> list[Document]:
    """Load ``.pdf`` / ``.txt`` / ``.md`` files from a folder (non-recursive)."""
    docs: list[Document] = []
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(load_pdf_file(path))
        elif suffix in {".txt", ".md"}:
            docs.append(load_text_file(path))
    return docs
