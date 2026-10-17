---
name: lasagna-support
description: Use when Lasagna errors, hangs, returns no results, or behaves unexpectedly, and the problem should be reported. Captures redacted diagnostics and files a GitHub issue with enough detail to be actionable, using the right template.
---

# Report a Lasagna problem

The goal is a report a maintainer can act on without a round trip. That means
diagnostics, a reproduction, and the right template, all of which are
mechanical, so do them mechanically.

## 1. Rule out the common causes first

Do not file until these are checked. Roughly all reports are one of them.

```bash
lasagna doctor
```

| `doctor` says | It is not a bug |
|---|---|
| `model_cache: no Hugging Face cache found` | The first run on the default configuration downloads 3.2 GB (BAAI/bge-m3 and mixedbread-ai/mxbai-rerank-large-v1). It is not hung. |
| `llm_credentials: no LLM API key detected` | On the default profile, a search the router sends to agentic retrieval needs a key. `lasagna configure --profile speed` never calls an LLM. |
| `import:X not installed` | Missing extra. Run the `pip install` it prints. |
| `plugin:X failed to load` | Their plugin is broken. Results silently used the default. |

If `doctor` reports `FAIL` on `vector_backend` or `storage`, fix that first:
everything downstream will misbehave.

The first lines of `doctor` name the session, profile, settings file and data
directory in force. A profile set with `lasagna configure -s NAME` applies to
that session only (`docs/configuration.md`); check that the session in the
report is the one the user searches.

## 2. Reduce it to something runnable

A report without a reproduction usually gets one question back and then stalls.
Get to the smallest command that shows the problem:

```bash
lasagna upload ./one-file.md -s repro
lasagna search "the query" -s repro -k 5
```

If it only reproduces with a specific document, say so, and say what kind
(scanned PDF, huge table, non-English), not the document itself.

## 3. Capture diagnostics

```bash
lasagna doctor --issue
```

Emits a collapsible block: version, install source, platform, GPU, and every
check. **Already redacted**: no absolute paths, no credential values.

Do not hand-write this section. The redaction is what makes it safe to paste.

## 4. File it

Check for a duplicate first:

```bash
gh issue list --repo dolev31/lasagna --search "your error message" --state all
```

Pick the template by what kind of problem it is:

| Template | Use when |
|---|---|
| `bug_report.yml` | it crashes, hangs, or returns wrong results |
| `component_contribution.yml` | you built a component and want it listed |
| `result_dispute.yml` | a **published number** does not reproduce |

```bash
gh issue create --repo dolev31/lasagna \
  --title "search returns 0 results after successful upload" \
  --body "$(cat <<'EOF'
## What happened
`lasagna search` returns no results for a corpus that ingested cleanly.

## Reproduce
lasagna upload ./notes.md -s repro     # reports 12 chunks
lasagna search "gallium" -s repro      # returns nothing

## Expected
The chunk containing "gallium" to be returned.

EOF
)

$(lasagna doctor --issue)"
```

For a **result dispute**, run the comparison first and paste its output; it
usually resolves the question without anyone having to rule:

```bash
lasagna results compare theirs.json yours.json
```

Most disputes turn out to be two runs that held different things fixed, which
the field-level diff shows directly.

## Rules

- Never paste raw logs containing paths or keys. `--issue` output is redacted;
  arbitrary terminal output is not.
- Do not file without `lasagna doctor` output. It is the difference between a
  report someone can act on and one that needs three replies first.
- One problem per issue.
- If `gh` is not installed, print the assembled body and tell the user to open
  the issue in a browser. Do not silently skip filing.
