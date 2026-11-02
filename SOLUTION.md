# Solution

A command-line agent that answers CityWheels customers from the shop's documents and bookings. Python, a hand-written tool-calling loop, the Gemini API by default.

```
helpdesk/
  cli.py        command line: one question, or a chat
  agent.py      the agent loop
  llm.py        client for OpenAI-compatible chat APIs (Gemini by default)
  prompts.py    the instructions for the model
  tools.py      the three tools, their JSON schemas and the dispatcher
  refunds.py    the cancellation rules, in code
  policies.py   keyword search over data/policies/
  bookings.py   booking lookup
  config.py     settings from environment variables
tests/          unit tests and agent tests with a fake model
eval/           evaluation questions, checks, runner and results
```

## Install and run

Needs Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Put an API key into `.env` as `LLM_API_KEY`. A free Gemini key takes a minute to create at https://aistudio.google.com/apikey. The `.env` file is in `.gitignore`; `.env.example` only lists the variable names.

```bash
python -m helpdesk "What time do you open on Saturday?"
python -m helpdesk -v "I want to cancel booking CW-1002. How much will I get back?"   # -v prints the tool calls
python -m helpdesk                 # chat: follow-up questions keep the context, "exit" quits

python -m pytest                   # all tests: no network, no API key
python -m eval.run_eval            # the evaluation, writes eval/results.md
```

## Model and provider

The default is **`gemini-3.5-flash-lite` on the Gemini API free tier**, called through Google's OpenAI-compatible endpoint with the official `openai` Python package.

- Gemini has a free tier, needs no local GPU and supports tool calling. GitHub Models, the other hosted option in the brief, was retired by GitHub on 30 July 2026.
- Flash-Lite rather than the bigger `gemini-3.8-flash`: the bigger model's free tier was reported to allow only 20 requests per day, and one evaluation run needs about 40.
- The OpenAI-compatible endpoint means the same code works with other providers by changing three variables in `.env`, for example a local model in Ollama (see `.env.example`). The `openai` package is only the HTTP client: it sends the request, turns the JSON response into objects and retries network errors. The loop, the tools and the prompt are plain code in this repository. I did not use an agent framework: the loop is about 40 lines and easy to test without one.
- One Gemini detail mattered: Gemini 3 models attach "thought signatures" to their tool calls and reject the next request with HTTP 400 if one is missing. So the loop appends the model's message to the history exactly as it came back instead of rebuilding it. `tests/test_llm_client.py` checks this against a local fake of the API.

## How the agent works

### The loop (`helpdesk/agent.py`)

1. The history starts with the instructions (system prompt) and the customer's question.
2. The history and the JSON schemas of the three tools go to the model.
3. If the model asks for tools, the code runs each one and adds the results to the history as `tool` messages. Back to step 2.
4. If the model answers with text, that is the answer.

The loop stops after 6 model calls and gives a safe answer that points to the staff, so a confused model cannot loop forever. Tool problems (unknown booking, invalid JSON arguments, a tool that does not exist, an exception inside a tool) go back to the model as a result it can read; they never crash the program. API problems (wrong key, used-up quota, no network) end with a short error message. In chat mode the history is kept between questions, which is what makes "and what if I cancel on Friday instead?" work.

### The tools (`helpdesk/tools.py`)

**`search_policies(query)`** splits the three Markdown files into 22 sections at their `##` headings and ranks them with BM25, the classic search-engine formula. Words are stemmed (so "cancelling", "cancelled" and "cancellation" all match), "e-bike" is normalised to "electric bike", and headings count double because the FAQ headings are the questions customers ask. It returns the top 3 sections with file name and heading, or an empty list for things like "capital of Australia". I chose keyword search over embeddings because the documents are tiny: BM25 puts the right section first for every query in the tests, is deterministic, and needs no second API.

**`get_booking(booking_id)`** returns the booking from `data/bookings.json`. It accepts `CW-1002`, `cw1002`, `1002` and similar, and returns `found: false` with a message for unknown or invalid numbers.

**`calculate_refund(booking_id, cancel_at=None)`** computes the refund in code (`helpdesk/refunds.py`, with `Decimal`), following `cancellations.md`:

- only `confirmed` bookings can be cancelled; `in_progress`, `completed` and `cancelled` get nothing;
- 48 hours or more before the start: 100% of the rental price; 24 to 48 hours: 50%; less than 24 hours: 0%. The limits are inclusive: exactly 48 hours is "48 hours or more" (booking CW-1007), exactly 24 hours is "24 hours or more" (CW-1010);
- the booking fee is never refunded, the deposit always is;
- card payments go back to the card within 5 working days, voucher payments come back as a new voucher.

It returns a breakdown (rule applied, rental part, deposit, fee, total, method), so the model can explain the amount without calculating anything. The optional `cancel_at` answers "what if I cancel later" questions; times in the past are rejected, and a bare date means that day at the current time.

### The instructions (`helpdesk/prompts.py`)

The system prompt tells the model:

- the current date and time: 10 June 2026, 09:00, Europe/Warsaw, as the data README says (`CITYWHEELS_NOW` changes it). The model needs it, for example, to know that the April-October opening hours apply;
- to answer only from tool results, never from general knowledge;
- to search the policies before answering about rules and prices, with the query in English even when the customer writes in Polish;
- to use `get_booking` for booking questions and never invent details;
- to always use `calculate_refund` for refunds and repeat its amounts exactly;
- when the documents do not answer, or the question is not about CityWheels, to say so and point to the staff. The staff contact is read from `cancellations.md`, not hard-coded;
- to ignore instructions in customer messages that try to change these rules, and to reply in the customer's language.

## Tests

`python -m pytest` runs 113 tests in about 5 seconds, without network or API key.

- `test_refunds.py`: every rule of `calculate_refund`, including both limits from each side (48 h, 47 h 59 min, 24 h, 23 h 59 min), fee, deposit, all statuses, cancelling after the start, card and voucher, rounding, all 10 example bookings and the `cancel_at` cases.
- `test_bookings.py`, `test_policies.py`, `test_tools.py`: lookups, booking number formats, invalid input, search ranking, and the dispatcher's error handling.
- `test_agent.py`: the agent loop with a scripted fake model: tool call, result, final answer; an unknown booking; broken tool calls; several tool calls in one reply; the step limit; a follow-up question; an API error in the middle of a turn.
- `test_llm_client.py`: the real client against a small local HTTP server that imitates the API: parsing tool calls, sending the thought signature back, wrong key, rate limits.
- `test_eval.py`: the question file and the evaluation checks.

To make sure the tests catch real mistakes, the code was broken on purpose in 12 places (for example `>` instead of `>=` at the 48-hour limit, refunding the booking fee, dropping the thought signature, not adding tool results to the history). Every change made at least one test fail.

GitHub Actions (`.github/workflows/tests.yml`) runs the tests on every push, on Python 3.10 and 3.13.

## Evaluation

`eval/questions.json` has 16 questions, each with the expected outcome in words and automatic checks:

- 5 about policies: opening hours, a day price, a 3-hour price asked in Polish, age limits, a storm warning (the staff decide);
- 7 refunds: 100%, 50%, 0%, exactly 48 hours, exactly 24 hours, a voucher booking, and a two-question conversation ("and what if I cancel on Friday at 9:00 instead?");
- 2 booking cases: a rental already in progress, and a booking number that does not exist;
- 2 the agent should not answer: the capital of Australia, and parking, which the documents do not mention.

The checks (`eval/checks.py`) look at which tools were called, what a tool returned (for example `calculate_refund` total = 340), which amounts must and must not appear in the answer (340 must; 342, meaning the booking fee was refunded, must not), some phrases, and whether the answer points to the staff. Checks like these are simple to read but can be fooled, so the report also shows every answer and tool call.

`python -m eval.run_eval` runs every question in a fresh conversation, waits 5 seconds between questions because of the free tier's per-minute limit, and writes [`eval/results.md`](eval/results.md). `--model` and `--output` make it easy to compare two models.

### Results of the last run

> **TODO before sending:** run `python -m eval.run_eval`, then replace this paragraph with the score, what failed, why, and what was changed. Full report: [eval/results.md](eval/results.md).

## How I used AI coding assistants

I built this with Claude (Anthropic) working as a coding agent. I gave it the brief and the data; it wrote most of the code, the tests, the evaluation questions and the first version of this document, and it ran the checks described above: a smoke test of the whole CLI against a local fake of the Gemini API, the test suite on Python 3.10 and 3.13 and with both `openai` 2.x and 3.x, and the deliberate-bug check.

Places where its output was wrong or needed a change, and how that came out:

- **Provider.** The brief and Claude's own knowledge both pointed to GitHub Models as a free option. GitHub's documentation says the service was retired on 30 July 2026. This came out when the current docs were checked before choosing the provider, so the agent uses Gemini instead.
- **Date in the prompt.** The first version of the prompt builder added the colon to the UTC offset in the wrong place and closed a bracket twice, so the prompt would have said `UTC+020:0))`. It was spotted when reading the code before the first run; a test now checks the exact sentence.
- **Dependency assumption.** A comment in the CI workflow claimed that Python 3.10 installs `openai` 2.x. Running the tests on Python 3.10 showed that pip picks `openai` 3.x there too, with an older version of its HTTP library. The comment was wrong and was fixed; `openai` 2.x was then tested separately.

## Limitations

- Anyone who knows a booking number can see the booking: there is no check that the customer owns it.
- The search only understands words that appear in the documents, in English. The model has to translate the query, and a word like "address" finds nothing until the model tries "where".
- "Now" is fixed, and times without a time zone are read as Europe/Warsaw.
- `calculate_refund` trusts the booking data: it does not recompute the price from the price list. For a booking paid with a voucher it cannot know how the deposit was paid, so it repeats the policy sentence "the deposit is refunded the same way it was paid".
- Correct tool results do not guarantee a correct answer: the model can still phrase something wrongly. The evaluation catches only part of that: string checks, one run per question, and LLMs are not deterministic.
- The agent only explains. It cannot cancel or change bookings, and storm cases go to the staff, as the policy says.
- Free-tier Gemini prompts may be used by Google to improve its products, so this setup is not for real customer data.
- In a long chat the history grows without limit.

## What I would do next

- Run each evaluation question several times and report pass rates instead of one pass/fail, and compare two models (the runner already supports it).
- Add an LLM-as-judge check of answer quality on top of the string checks, validated on a few answers checked by hand.
- Add a `quote_price` tool, so rental prices ("two e-bikes for 3 hours") are also calculated by code, like refunds.
- Verify the customer before showing a booking, and add a real cancellation tool that asks for confirmation.
- Move to embedding or hybrid search if the documents grow.
- Log every tool call and model response, to see why a wrong answer happened.
