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

## Testy bez API klíče

Po instalaci závislostí spusťte offline testy:

```bash
python -m unittest discover -s tests -v
```

Testy ověřují kalkulačku i simulovaný cyklus volání nástroje; nepotřebují API
klíč ani neposílají síťové požadavky.

## Test se skutečným API

Nastavte `OPENAI_API_KEY` podle výše uvedeného návodu a spusťte například:

```bash
python pythonLLMscript.py "What is 25 * (10 + 2)?"
```

Tento příkaz odešle požadavek do OpenAI API a může spotřebovat kredity.