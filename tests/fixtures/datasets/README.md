# Dataset fixtures

Tiny, synthetic datasets for `tests/unit/inspection_dataset/`. No row comes from a real dataset;
all text was written for these tests. Binary files are regenerated with
`uv run --no-sync python tests/fixtures/datasets/generate_binary.py`.

| File | Format | Purpose |
|---|---|---|
| `preference.jsonl` | JSON Lines | `system`/`question`/`chosen`/`rejected` + metadata `lang`/`vulnerability` (shape of the example dataset); Korean, emoji, combining marks, code fences |
| `messages.jsonl` | JSON Lines | conversational `messages` (one row with a `tool_calls` turn → datasets `Json` field) + metadata `id` |
| `prompt_completion.parquet` | Parquet, 2 row groups | `prompt`/`completion` + metadata `source` |
| `text.csv` | CSV | `text` column with a quoted comma, a quoted newline and an `NA` cell (pandas NA rule) |
| `array.json` | JSON array | `instruction`/`output` objects (one-load JSON array path) |
| `malformed.jsonl` | JSON Lines | line 3 is not valid JSON; the other four records are |
| `multi_split/{train,test}.jsonl` | JSON Lines dir | split isolation (`train` and `test` by file name) |
| `ambiguous.jsonl` | JSON Lines | both `prompt` and `question` (and `answer`) present → ambiguous mapping |
| `json_lines.json` | JSON Lines named `.json` | extension says JSON, content is JSON Lines (like the example dataset) |
