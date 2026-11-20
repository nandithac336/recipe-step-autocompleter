from app.cli import main
from app.storage import Store

def test_store(tmp_path):
    store = Store(str(tmp_path / "app.json"))
    store.put("recipe", {"step": 1})
    assert store.get("recipe") == {"step": 1}

def test_hello(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out
