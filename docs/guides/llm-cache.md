# LLM response cache

Inside `lasagna benchmark experiment`, a request asked at temperature 0 that an
earlier run made with the same seed is answered from
`~/.lasagna/cache/llm.sqlite`. A second run of the same sweep, after a crash or
with one arm added, then spends no tokens on what it already asked.

Token counts read as in the cold run. `llm_cache_hits` and `llm_cache_misses` on
each task, and a hit rate in `results.json` and the bundle, say what was
replayed.

## What is never replayed

- **Sampled calls.** A response at temperature 0.7 is one draw. Replaying it
  would make repeated runs identical and hide the spread `n_runs` is meant to
  measure.
- **The RAGAS judge**, which asks for temperature 0.01, unless you set
  `llm_cache: {deterministic_only: false}`.
- **Models that take no temperature.** The cache replays a call only when
  temperature 0 was actually sent, so a model that gets no temperature (Claude
  4.7 and later, some OpenAI reasoning models) is always called.

## Your own provider

A provider you set with `set_llm_provider` gets the cache too. The chat model its
`get_chat_model` returns is replayed when temperature 0 was asked for and the
model's `temperature` is 0. A returned object that is not a LangChain chat model
cannot carry the cache, and `results.json` then says `"applied": false` with the
reason under `llm_cache`.

## Turn it off or move it

- `llm_cache: false` in the sweep file
- `--no-cache` on the command
- `LASAGNA_LLM_CACHE=0` in the environment
- `llm_cache: {path: ...}` puts the file elsewhere

Nothing is cached outside an experiment, so `lasagna serve` and the MCP server
write no prompt to disk.
