# Catalogue assistant

This lab uses LangChain to extract search parameters from a shopping request, Pydantic
to validate them, and SQLite to retrieve matching products. The report covers all five
parts of the assignment, including tool calls, retries, reasoning and follow-up
requests.

## Review the results

Start with [the report](artifacts/report.md). The project includes the recorded
experiment results and a populated database with **90 synthetic products**: 30 laptops,
30 smartphones and 30 tablets.

| File | Contents |
| --- | --- |
| [artifacts/report.md](artifacts/report.md) | Results and comparisons for every part, error analysis, retry trace, conclusions and answers to the ten review questions |
| [artifacts/results.json](artifacts/results.json) | Original model outputs, validated queries, expected and retrieved products, tool calls, timings, token usage, query schema, run configuration and catalogue snapshot |
| [artifacts/events.jsonl](artifacts/events.jsonl) | Pipeline events grouped by `request_id`, showing attempts, validation and search |
| [catalogue.sqlite3](catalogue.sqlite3) | Ready-to-query catalogue; it can also be recreated from the generator |

The assignment is included as [SPECIFICATION.md](SPECIFICATION.md).

## Check the saved results without making model requests

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Make. Use
Python 3.12 or newer. Open a terminal in the project directory and run:

```sh
make setup
make check test verify
```

`make setup` installs the dependencies pinned in [uv.lock](uv.lock) and creates `.env`
from [`.env.example`](.env.example) if it is missing. Existing configuration is
preserved. The first install requires internet access unless dependencies are already
cached. Python's standard-library sqlite3 module handles the database.

These commands check formatting, lint the code, run strict type checks and offline
tests, and validate the saved results and execution log. They do not require a model
token.

## Reproduce all experiments

1. Run `make setup` if it has not been run already.
2. Edit `.env` and set `ITMO_LLM_TOKEN` to your own university token.
3. Connect to the university network. The example uses
   `http://10.32.2.11:35000/v1` and model `qwen3.8-27b`.
4. Run:

   ```sh
   make ping
   make review
   ```

The connectivity check should return `OK`. `make review` populates the catalogue and
runs code checks and tests. It then checks model access, runs every experiment, builds
the report and validates the results. It refreshes the three files in `artifacts/`.

The results contain eight sections: basic chain, structured extraction, database search,
tool calling, thinking comparison, dialogues, answers without database access, and error
handling with scripted responses. The scenarios include ten searches, seven tool
requests, four reasoning-mode comparisons and twelve dialogue turns. Reasoning is
disabled except in the comparison required by task 2.4.

Model responses and timings can vary between runs. Expectations are fixed in the test
data. Failed checks remain visible in `results.json`, and report generation and
verification stop with a nonzero exit code. An interrupted run has `completed_at: null`
and cannot produce a completed report.

## Recreate or expand the database

A populated database is provided. To recreate it when missing or add missing products to
an existing catalogue, run:

```sh
make populate
```

This command also creates `.env` when needed and installs the locked dependencies. It
does not call the model. To test reconstruction while retaining the supplied database,
set `DATABASE_PATH=./regenerated.sqlite3` in `.env`, run `make populate`, then restore
the original path when finished.

| `.env` setting | Purpose | Default example |
| --- | --- | --- |
| `DATABASE_PATH` | Database destination; missing parent directories are created | `./catalogue.sqlite3` |
| `PRODUCTS_PATH` | Product examples supplying names and price ranges | `./fixtures/products.json` |
| `PRODUCTS_PER_CATEGORY` | Number of products per category, including examples | `30` |
| `POPULATION_SEED` | Reproducible generation of additional prices | `42` |

[The generator](catalogue/population.py) retains the fifteen example products and adds
variants with equal prices, prices just below and above the example values, and
repeatable random prices within each category's range. Laptop price limits of 49999,
50000 and 50001 yield different subsets; tablets with a zero price limit yield none. The
catalogue uses synthetic prices for the experiments.

Repeated population does not duplicate rows. Increasing the size preserves existing IDs
and values. Changing the seed or examples, reducing the catalogue size, or editing its
rows can create conflicts with existing products. In that case, choose a new
`DATABASE_PATH`; existing rows are never overwritten or removed. After changing
population settings, rerun the experiments before rendering or verifying their results.

## Run individual scenarios

Runtime configuration comes exclusively from `.env`; shell variables and command-line
arguments do not override it. Paths resolve from the project working directory.

| Scenario | `.env` settings | Command and expected result |
| --- | --- | --- |
| JSON search | `EXTRACTION_METHOD=json`, `QUERY_TEXT=Покажи ноутбуки до 50 тысяч` | `make run`: validated query and matching products |
| Tool search | `EXTRACTION_METHOD=tools`, same text | `make run`: validated `search_products` call and direct search results |
| Empty result | `QUERY_TEXT=Покажи планшеты до 0 рублей` | `make run`: `empty` |
| Unsupported category | `QUERY_TEXT=Найди холодильники` | `make run`: `unsupported_category` |
| Price confirmation | `REQUIRE_PRICE_CONFIRMATION=true`, `UNRESTRICTED_SEARCH_CONFIRMED=false`, `QUERY_TEXT=Хочу посмотреть планшеты` | `make run`: `confirmation_required`; set confirmation to `true` and rerun to search |
| Search without a model | Blank `QUERY_TEXT`; `PRODUCT_QUERY='{"category":"laptop","max_price":50000}'` | `make run`: deterministic search without a token |
| Saved measurements | Keep settings matching the supplied run | `make report verify`: rebuild the report and check the saved results without calling the model |

`make run` ensures the catalogue exists and appends request events to the journal. It
prints the result without creating another artifact. Restore the example confirmation
settings before running the complete review.

## Reproduce failure scenarios without the university network

To retain the submitted live evidence, first change `ARTIFACTS_PATH` in `.env` to
`./offline-results`, then run:

```sh
make offline
```

This executes the real parser, LCEL retries and SQLite adapter with scripted model
responses. The scenarios cover malformed JSON, invalid categories and prices, recovery
after a failed attempt, exhausted retries, execution without retries, price confirmation
and a response with no tool call. Its report is labelled `offline`. Restore
`ARTIFACTS_PATH=./artifacts` afterward. Running it with the default path replaces the
submitted live results with offline results.

## Source layout and troubleshooting

- [catalogue/](catalogue/) contains domain models, the search use case, SQLite and
  model adapters, dialogue contexts, the experiment runner and the report renderer.
- [fixtures/](fixtures/) contains product examples, queries, dialogues and error scenarios.
- [prompts/](prompts/) contains model instructions; [tests/](tests/) checks the behavior.
- [pyproject.toml](pyproject.toml) declares dependencies and named entry points.
- [docs/report_template.md](docs/report_template.md) supplies the report's explanatory text.

Configuration errors identify the invalid field. Transport or authentication errors
require network access and a valid token; parsing retries cannot fix them. If generated
products conflict with existing rows, select a new database destination. For an
incomplete run or failed expectation, inspect the JSON and journal, fix the cause, and
rerun `make review`.
