"""Keyword search over the policy documents in data/policies/.

Each Markdown file is split into sections at its "## " headings. A question is
matched against the sections with BM25, the ranking formula classic search
engines use. The documents are small, so this is enough and needs no
embeddings API, and it is fully deterministic, which makes it easy to test.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")

# Very common words that say nothing about the topic. "where" and "when" are kept
# on purpose: they match the FAQ headings "Where are you?" and "When are you open?".
_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how
    i if in into is it its me my of on or our please so than that the their them then
    there these they this to us was we were what which who will with would you your""".split()
)
# "e-bike", "e bike", "ebikes" -> "electric bike", so they match "Electric bikes" in the FAQ.
_EBIKE = re.compile(r"\be[\s-]?bikes?\b")
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    text = _EBIKE.sub("electric bike", text.lower())
    return [_STEMMER.stemWord(word) for word in _WORD.findall(text) if word not in _STOPWORDS]


@dataclass(frozen=True)
class PolicySection:
    source: str  # file name, e.g. "faq.md"
    document: str  # the file's "# " title
    heading: str  # the section's "## " heading
    text: str  # the section body, as Markdown


def split_sections(markdown: str, source: str) -> list[PolicySection]:
    document = source
    heading = ""
    body: list[str] = []
    sections: list[PolicySection] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append(PolicySection(source, document, heading or document, text))

    for line in markdown.splitlines():
        if line.startswith("# "):
            flush()
            document, heading, body = line[2:].strip(), "", []
        elif line.startswith("## "):
            flush()
            heading, body = line[3:].strip(), []
        else:
            body.append(line)
    flush()
    return sections


class PolicyIndex:
    K1 = 1.5  # how fast repeated words stop adding to the score
    B = 0.75  # how much long sections are penalised

    def __init__(self, sections: list[PolicySection]):
        self.sections = sections
        self._term_counts = [Counter(self._index_terms(s)) for s in sections]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._avg_length = (sum(self._lengths) / len(sections)) if sections else 1.0
        document_frequency: Counter = Counter()
        for counts in self._term_counts:
            document_frequency.update(counts.keys())
        n = len(sections)
        self._idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()}

    @classmethod
    def from_dir(cls, policy_dir: Path) -> PolicyIndex:
        sections: list[PolicySection] = []
        for path in sorted(policy_dir.glob("*.md")):
            sections.extend(split_sections(path.read_text(encoding="utf-8"), path.name))
        return cls(sections)

    @staticmethod
    def _index_terms(section: PolicySection) -> list[str]:
        # The heading counts twice: FAQ headings are the questions customers ask.
        return tokenize(section.heading) * 2 + tokenize(section.text) + tokenize(section.document)

    def search(self, query: str, limit: int = 3) -> list[tuple[PolicySection, float]]:
        terms = set(tokenize(query))
        scored = []
        for i, counts in enumerate(self._term_counts):
            length_factor = 1 - self.B + self.B * self._lengths[i] / self._avg_length
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self._idf[term] * tf * (self.K1 + 1) / (tf + self.K1 * length_factor)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(self.sections[i], score) for score, i in scored[:limit]]

    def find_section(self, heading: str) -> PolicySection | None:
        for section in self.sections:
            if section.heading.lower() == heading.lower():
                return section
        return None
