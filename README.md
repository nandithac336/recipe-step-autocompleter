[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/wkfhxdbd8p/badge.svg)](https://msrit.ask.thili.ai/b/wkfhxdbd8p)
# Recipe Step Autocompleter

A local command-line application that completes partial cooking instructions using a trigram (n-gram) language model. It trains on the included recipe-instruction corpus and does not need an API key, internet connection, hosted service, or GPU.

## Project goal

A user enters an incomplete recipe instruction and receives likely next-word suggestions or a generated continuation.

## Milestones completed

- [x] **1. Compile a corpus of cooking recipe instructions.** `data/recipe_instructions.txt` contains cleaned, one-instruction-per-line cooking examples. `load_corpus()` tokenizes and normalizes them.
- [x] **2. Implement n-gram counting and maximum-likelihood probabilities with smoothing.** `NGramModel.fit()` counts contexts and next tokens. Add-alpha (Laplace) smoothing assigns non-zero probability to unseen continuations, with shorter-context backoff.
- [x] **3. Create a function to generate recipe step continuations.** `suggest()` ranks next-token candidates and `generate()` produces a short greedy continuation.
- [x] **4. Build a CLI for users to input partial steps and get suggestions.** The CLI supports suggestions, generated completions, and interactive input.

## Requirements

- Python 3.10 or newer
- Git (only needed to upload to GitHub)
- No API key or model download required

## Setup

### Windows PowerShell
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the application

```bash
python -m app.cli hello
python -m app.cli suggest "heat oil in a"
python -m app.cli suggest "add the chopped" --count 5
python -m app.cli complete "wash and chop" --words 6
python -m app.cli interactive
```

Type `quit` or `exit` to leave interactive mode.

## Run tests

```bash
pytest -q
```

GitHub Actions automatically runs the test suite on each push and pull request.

## How it works

1. **Preprocessing:** lowercases recipe instructions and tokenizes words and punctuation.
2. **Training:** builds counts for one-word through (n-1)-word contexts and their following tokens.
3. **Probability estimation:** estimates next-word probabilities using add-alpha smoothing:  
   `P(w | h) = (count(h,w) + alpha) / (count(h) + alpha * |V|)`.
4. **Backoff:** if a longer context was not observed, the predictor tries a shorter context.
5. **Prediction:** ranks candidate next tokens by probability. The completion command repeatedly chooses the highest-ranked token until its word limit or an end-of-instruction token is reached.

## Example

```text
$ python -m app.cli suggest "heat oil in a"
Input: heat oil in a
Suggestions:
  large        ...
  pan          ...
```

Exact probabilities depend on the included corpus and model vocabulary.

## Repository contents

```text
recipe-step-autocompleter/
├── app/
│   ├── cli.py
│   ├── core.py
│   └── storage.py
├── data/
│   └── recipe_instructions.txt
├── tests/
│   ├── test_core.py
│   └── test_smoke.py
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

## Limitations and future improvements

This is a small educational n-gram model, not a large language model. Suggestions are limited by the training corpus and may not always be factually correct cooking advice. Future work could expand the corpus, compare bigram/trigram models, add beam search, or allow users to choose among several continuations.
