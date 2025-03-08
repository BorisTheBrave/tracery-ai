See https://www.boristhebrave.com/2025/03/08/tracery-ai/ for details.

# Usage

Download a model in GGUF format from hugging face (e.g. https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF/tree/main)

Then run:

```
uv run .\llama_grammar_gen.py --grammar grammars/test.json --samples 10 -t 1 --model models/my_model.gguf
```

(or setup your python environment and run the script from there)