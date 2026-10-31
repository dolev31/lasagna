# Stage isolation

A comparison tells you something only when the two arms differ in the one thing
you changed. Lasagna splits a pipeline into eight stages and checks, at run time,
that a swap stayed inside its stage.

## The eight stages

```text
document_processor -> chunker -> embedder -> retriever -> query_enhancer
                                             -> reranker -> agentic_pipeline -> generator
```

Each stage has a fixed interface. A reranker takes a query and candidates and
returns them ranked. A chunker takes text and returns chunks. Because the
interface is fixed, you can swap one component for another and leave the other
seven stages alone. The [catalog](../catalog/stages.md) lists the stages and their
contracts, and the [write a component](../guides/write-a-component.md) guide shows
how to fill one.

## What the check does

Each task records a stage manifest. The manifest is a digest of every setting and
every component that reached each stage. After the sweep, Lasagna compares the
manifests of all arms against what each arm said it would change.

Two things count as a failure:

- **Undeclared change.** A setting differs between two arms, but neither arm
  named it, and it is not one the harness sets because of what an arm named
  (naming a reranker turns reranking on, for example). The stage it sits in may
  be one an arm changes for another reason: a `rerank_top_k` that leaked into a
  reranker sweep is caught, although every arm there names a reranker.
- **Inert change.** An arm names a change that its manifest does not show. The
  setting or component never reached the pipeline, so the arm's number is the
  baseline's under another name.

Both are judged setting by setting, and per track. The retrieval track searches
every arm the same way in a few settings (it turns the agentic loop off and asks
for a fixed depth), and the generation track searches with the arm's own; the
manifest records those settings for each track, and which tracks ran. An arm
that names one of them is inert on the retrieval track, where its retrieval
numbers are the control's, and live on the generation track, and the run says
so.

The run prints both. `lasagna results validate` refuses a bundle with an
undeclared change, and `lasagna results view` shows the check on its page.

## What the check does not see

The manifest digests configuration, not behavior. A component that changes state
outside its stage while it runs, such as patching another module, leaves every
digest unchanged and passes the check. Two other guards cover that case:

- The stage contracts, which say what a component may touch.
- The conformance tests. `lasagna component validate <module:Class>` runs them
  against your component. See
  [Contributing components](../contributing-components.md).

## Strict stages

A stage that cannot run fails the task. If you ask for query expansion and no LLM
is configured, or an agentic search errors, the task fails. It does not fall back
to the baseline and report the baseline's score under the arm's name.

## Settings that do not count

Some settings change how a run is operated, not what it computes. A cache backend,
a connection pool size and a log level are examples. They stay out of the
manifest, so they never read as a change to the system under test.
