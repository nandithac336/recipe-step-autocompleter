import pytest
from app.core import NGramModel, load_corpus, tokenize

def test_tokenize_normalizes_text():
    assert tokenize("Add TOMATOES, then stir!") == ["add", "tomatoes", ",", "then", "stir", "!"]

def test_milestone_1_loads_recipe_corpus():
    corpus = load_corpus()
    assert len(corpus) >= 20
    assert ["heat", "oil", "in", "a", "large", "pan"] in corpus

def test_milestone_2_trains_smoothed_ngram_model():
    model = NGramModel(n=3, alpha=0.5).fit([["add", "salt"], ["add", "pepper"]])
    probs = model.next_word_probabilities("add")
    assert "salt" in probs and "pepper" in probs
    assert all(p > 0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0)

def test_milestone_3_generates_continuation():
    model = NGramModel().fit([["add", "salt", "and", "stir"], ["add", "pepper", "and", "mix"]])
    result = model.generate("add", max_words=2)
    assert result.startswith("add ")
    assert len(result.split()) >= 2

def test_milestone_4_cli_suggest_and_complete(capsys):
    from app.cli import main
    main(["suggest", "heat oil"])
    output = capsys.readouterr().out
    assert "Suggestions:" in output
    main(["complete", "heat oil", "--words", "2"])
    assert "Completed instruction:" in capsys.readouterr().out

def test_empty_training_data_is_rejected():
    with pytest.raises(ValueError):
        NGramModel().fit([])
