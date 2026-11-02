# CityWheels help desk agent

A command-line AI agent that answers customer questions for CityWheels, a made-up bike rental shop. It uses an LLM with tool calling and answers only from the shop's documents and bookings in [`data/`](data/README.md).

```bash
pip install -r requirements.txt
cp .env.example .env      # then put your Gemini API key into .env
python -m helpdesk "I want to cancel booking CW-1002. How much will I get back?"
```

- How to install and run everything, how it works, and what the evaluation found: [SOLUTION.md](SOLUTION.md)
- Last evaluation run: [eval/results.md](eval/results.md)
- What the data files mean: [data/README.md](data/README.md)
