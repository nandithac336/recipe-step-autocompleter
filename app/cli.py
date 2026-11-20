"""Command-line interface for Recipe Step Autocompleter."""
import argparse
from .core import train_default_model

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="recipe-step-autocompleter",
        description="Suggest likely next words for incomplete recipe instructions."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the application is ready")
    suggest = sub.add_parser("suggest", help="suggest the next words")
    suggest.add_argument("text", nargs="+", help="partial recipe instruction in quotes")
    suggest.add_argument("-k", "--count", type=int, default=3, help="number of suggestions (default: 3)")
    complete = sub.add_parser("complete", help="generate a short continuation")
    complete.add_argument("text", nargs="+", help="partial recipe instruction in quotes")
    complete.add_argument("-w", "--words", type=int, default=8, help="maximum words to generate")
    sub.add_parser("interactive", help="enter multiple instructions interactively")
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Recipe Step Autocompleter is ready.")
        return
    model = train_default_model()
    if args.command == "suggest":
        prompt = " ".join(args.text)
        if not prompt.strip():
            parser.error("Please enter a non-empty recipe instruction.")
        print(f"\nInput: {prompt}\nSuggestions:")
        for word, probability in model.suggest(prompt, args.count):
            print(f"  {word:<12} {probability:.3f}")
    elif args.command == "complete":
        prompt = " ".join(args.text)
        print("\nCompleted instruction:")
        print("  " + model.generate(prompt, args.words))
    elif args.command == "interactive":
        print("Recipe Step Autocompleter (type 'quit' to exit)")
        while True:
            try:
                prompt = input("\nPartial instruction: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if prompt.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break
            if not prompt:
                print("Please enter some words.")
                continue
            print("Suggestions:")
            for word, probability in model.suggest(prompt, 3):
                print(f"  {word:<12} {probability:.3f}")

if __name__ == "__main__":
    main()
