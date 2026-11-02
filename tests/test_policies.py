import pytest

from helpdesk.policies import split_sections, tokenize


@pytest.mark.parametrize(
    "query, source, section",
    [
        ("opening hours Saturday", "faq.md", "When are you open?"),
        ("What time do you open on Saturday?", "faq.md", "When are you open?"),
        ("e-bike price per day", "prices.md", "Bike rental"),
        ("How much is an e-bike for a whole day?", "prices.md", "Bike rental"),
        ("Can my 14-year-old rent an e-bike?", "faq.md", "Are there age limits?"),
        ("cancel booking refund rules", "cancellations.md", "Cancelling a booking"),
        ("storm warning", "cancellations.md", "Bad weather"),
        ("late return fee", "prices.md", "Late return"),
        ("can I pay the deposit in cash", "faq.md", "How can I pay?"),
        ("where is the shop", "faq.md", "Where are you?"),
        ("change the dates of my booking", "cancellations.md", "Changing a booking"),
    ],
)
def test_search_ranks_the_right_section_first(tools, query, source, section):
    results = tools.search_policies(query)["results"]
    assert results, f"nothing found for {query!r}"
    assert results[0]["source"] == f"data/policies/{source}"
    assert results[0]["section"] == section


def test_search_returns_the_section_text(tools):
    first = tools.search_policies("opening hours")["results"][0]
    assert "April to October: every day, 08:00–20:00." in first["text"]


def test_search_returns_at_most_three_sections(tools):
    assert len(tools.search_policies("bike price booking deposit")["results"]) == 3


@pytest.mark.parametrize("query", ["What is the capital of Australia?", "quantum physics", ""])
def test_search_finds_nothing_for_unrelated_queries(tools, query):
    result = tools.search_policies(query)
    assert result["results"] == []
    assert result["message"]


def test_tokenize_stems_words_and_normalizes_e_bike():
    assert tokenize("Cancelling cancelled cancellation") == ["cancel", "cancel", "cancel"]
    assert tokenize("e-bike") == tokenize("E bike") == tokenize("ebikes") == tokenize("electric bike")


def test_split_sections_handles_windows_line_endings():
    markdown = "# Title\r\n\r\nIntro text.\r\n\r\n## First\r\nBody one.\r\n## Second\r\nBody two.\r\n"
    sections = split_sections(markdown, "x.md")
    assert [(s.document, s.heading, s.text) for s in sections] == [
        ("Title", "Title", "Intro text."),
        ("Title", "First", "Body one."),
        ("Title", "Second", "Body two."),
    ]
