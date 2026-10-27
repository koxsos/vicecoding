# Python skript pro LLM API

Ukázka volá OpenAI API, které může podle potřeby spustit lokální nástroj
`calculator`. Výsledek nástroje se vrátí modelu, který následně vytvoří odpověď.

## Spuštění

Vyžaduje Python 3.9+ a API klíč OpenAI:

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY="váš-api-klíč"
python pythonLLMscript.py
```

Vlastní dotaz zadejte jako argument:

```bash
python pythonLLMscript.py "Kolik je 144 děleno 12?"
```

Případně lze změnit model přepínačem `--model`. Skript klíč načítá pouze
z proměnné prostředí; nevkládejte jej do zdrojového kódu ani do Git repozitáře.

Kalkulačka přijímá číselné výrazy s operátory `+`, `-`, `*`, `/`, `//`, `%`
a `**`. Výrazy vyhodnocuje omezeným AST parserem bez použití `eval`.