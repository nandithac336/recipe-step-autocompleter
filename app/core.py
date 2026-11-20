"""N-gram language model for completing short cooking instructions."""
from __future__ import annotations
from collections import Counter, defaultdict
import re
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "recipe_instructions.txt"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,!?;:]")

def tokenize(text: str) -> list[str]:
    """Normalize text and split it into word/punctuation tokens."""
    return TOKEN_RE.findall(text.lower())

def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[list[str]]:
    """Milestone 1: read, clean and tokenize one recipe instruction per line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recipe corpus not found: {path}")
    sentences = [tokenize(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()]
    return [sentence for sentence in sentences if sentence]

class NGramModel:
    """Interpolated n-gram predictor with add-alpha smoothing."""
    def __init__(self, n: int = 3, alpha: float = 0.5):
        if n < 2:
            raise ValueError("n must be at least 2")
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        self.n = n
        self.alpha = alpha
        self.vocabulary: set[str] = set()
        self.context_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.unigram_counts: Counter = Counter()
        self.trained = False

    def fit(self, corpus: list[list[str]]) -> "NGramModel":
        """Milestone 2: count n-grams and estimate smoothed next-token probabilities."""
        self.vocabulary = set()
        self.context_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        for sentence in corpus:
            tokens = ["<s>"] * (self.n - 1) + list(sentence) + ["</s>"]
            self.vocabulary.update(sentence)
            self.vocabulary.add("</s>")
            for i in range(self.n - 1, len(tokens)):
                target = tokens[i]
                self.unigram_counts[target] += 1
                for size in range(1, self.n):
                    context = tuple(tokens[i-size:i])
                    self.context_counts[context][target] += 1
        self.vocabulary.add("<unk>")
        self.trained = bool(self.unigram_counts)
        if not self.trained:
            raise ValueError("Cannot train on an empty corpus")
        return self

    def _probability(self, context: tuple[str, ...], word: str) -> float:
        counts = self.context_counts.get(context, Counter())
        total = sum(counts.values())
        vocab_size = max(1, len(self.vocabulary))
        return (counts[word] + self.alpha) / (total + self.alpha * vocab_size)

    def next_word_probabilities(self, prefix: str) -> dict[str, float]:
        if not self.trained:
            raise RuntimeError("Train the model before requesting predictions")
        words = tokenize(prefix)
        candidates = [w for w in self.vocabulary if w not in {"<s>", "<unk>"}]
        context = tuple(words[-(self.n - 1):])
        # Back off to shorter contexts if the exact context has never appeared.
        while context and not self.context_counts.get(context):
            context = context[1:]
        probs = {w: self._probability(context, w) for w in candidates}
        total = sum(probs.values())
        if total == 0:
            return {}
        return {w: p / total for w, p in probs.items()}

    def suggest(self, prefix: str, count: int = 3) -> list[tuple[str, float]]:
        """Return the most likely next words for a partial instruction."""
        if count < 1:
            raise ValueError("count must be positive")
        probs = self.next_word_probabilities(prefix)
        return sorted(probs.items(), key=lambda item: (-item[1], item[0]))[:count]

    def generate(self, prefix: str, max_words: int = 8) -> str:
        """Milestone 3: greedily continue a partial instruction."""
        if max_words < 1:
            return prefix.strip()
        result = prefix.strip()
        for _ in range(max_words):
            suggestions = self.suggest(result, 1)
            if not suggestions:
                break
            word = suggestions[0][0]
            if word == "</s>":
                break
            if word in ".,!?;:":
                result = result.rstrip() + word
            else:
                result = (result + " " + word).strip()
        return result

def train_default_model() -> NGramModel:
    return NGramModel(n=3, alpha=0.5).fit(load_corpus())
