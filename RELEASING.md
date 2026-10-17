# Releasing

Publishing a GitHub Release runs `.github/workflows/release.yml`, which checks
the tree, builds the sdist and wheel, smoke-tests the wheel in a clean venv and
uploads both to PyPI by trusted publishing. Nothing else uploads, and no token
is stored anywhere.

## One-time setup

Do these once, in this order, after the repository is public.

1. **GitHub Actions runs.** <https://github.com/settings/billing> must show no
   failed payment or spending-limit block, or no job starts. No workflow has
   run on GitHub since billing stopped them, so the first run after billing is
   restored is the first test of the workflows as they stand: start both by
   hand and watch them to the end before anything else.
   `gh workflow run tests.yml --repo dolev31/lasagna --ref main` runs every
   job, including the E2E gate and the live compose stack. The Release dry run
   builds, checks and smoke-tests but never publishes:
   `gh workflow run release.yml --repo dolev31/lasagna --ref main`. Follow each
   with `gh run watch --repo dolev31/lasagna`. Every job runs on
   `ubuntu-24.04`, named in the workflows rather than `ubuntu-latest`, so a new
   runner image cannot change a run without a commit.
2. **PyPI pending publisher.** On pypi.org (account with 2FA), go to Your
   account > Publishing > Add a new pending publisher > GitHub, and enter:
   PyPI project name `lasagna-rag`, owner `dolev31`, repository name `lasagna`,
   workflow name `release.yml`, environment name `pypi`. A pending publisher
   does not reserve the name: until the first upload, anyone can register
   `lasagna-rag` and the pending publisher is discarded.
3. **GitHub environment.** Settings > Environments > New environment, name
   `pypi`. Under Deployment branches and tags choose Selected branches and
   tags and add a tag rule `v*`. A required reviewer is optional; with one, the
   upload waits for your approval in the Actions run.
4. **Zenodo.** Sign in at zenodo.org with GitHub, open
   <https://zenodo.org/account/settings/github/>, press Sync now and switch
   `dolev31/lasagna` on. It must be on before the release is created: Zenodo
   only archives releases published after it is switched on.

## Release day

From an up-to-date `main` with CI green. The release date is the UTC date.

```bash
# 1. Date the release in CHANGELOG.md and CITATION.cff, check, commit, push.
DATE=$(date -u +%F)
perl -pi -e "s/^## \[0\.1\.0\] - YYYY-MM-DD/## [0.1.0] - $DATE/" CHANGELOG.md
perl -pi -e "s/^date-released: .*/date-released: \"$DATE\"/" CITATION.cff
python3.12 scripts/check_release.py v0.1.0
git commit -am "Release 0.1.0" && git push origin HEAD:main

# 2. The one action that publishes: create the release on that exact commit.
{ awk '/^## \[0\.1\.0\] /{f=1; next} /^## \[/{f=0} f' CHANGELOG.md
  echo "Earlier changes: https://github.com/dolev31/lasagna/blob/v0.1.0/CHANGELOG.md"
} > /tmp/notes-0.1.0.md
gh release create v0.1.0 --repo dolev31/lasagna --target "$(git rev-parse HEAD)" \
  --title "Lasagna 0.1.0" --notes-file /tmp/notes-0.1.0.md
gh run watch --repo dolev31/lasagna
```

The workflow refuses, before building, a tag that is not `v` + the
`pyproject.toml` version, a CHANGELOG still dated `YYYY-MM-DD`, or a
`CITATION.cff` whose version or `date-released` disagrees with it. If it fails
before the publish job, nothing was uploaded: fix `main`, then
`gh release delete v0.1.0 --repo dolev31/lasagna --cleanup-tag --yes` and repeat
step 2. Once a file is on PyPI it can never be replaced; a broken release is
yanked and followed by 0.1.1.

## Verify

```bash
python3.12 -m venv /tmp/lasagna-check
/tmp/lasagna-check/bin/pip install --no-cache-dir lasagna-rag==0.1.0
/tmp/lasagna-check/bin/lasagna --version        # lasagna 0.1.0
```

The PyPI page is <https://pypi.org/project/lasagna-rag/0.1.0/>.

## After the release

Zenodo mints a DOI within minutes (zenodo.org > GitHub shows it). In
`CITATION.cff`, uncomment `doi:` and set it to the DOI of version 0.1.0, then
commit. `date-released` was already set in step 1, because it has to be in the
tagged tree.

## Constraints file

`constraints/<version>.txt` pins the exact `[eval]` closure the release was
tested with, on Python 3.12. To make it for the next version, in the venv the
suite passes in (`uv venv --python 3.12`, then
`uv pip install -e ".[eval]" --group dev`):

```bash
uv pip freeze --exclude-editable > /tmp/tested.txt
uv pip compile pyproject.toml --extra eval --python-version 3.12 \
  --no-annotate --no-header -c /tmp/tested.txt -o /tmp/closure.txt
```

Write the three header lines of the current file, with the new version, above
`/tmp/closure.txt`, reinstall the venv with `-c constraints/<version>.txt`, and
run the suite again. The file pins the `[eval]` closure, but users combine it
with other extras, so check that every extra still resolves against it:
`uv pip compile pyproject.toml --all-extras --python-version 3.12 -c constraints/<version>.txt`.
0.1.0 pins typer 0.26.8, not the newest, because docling-core requires typer<0.27. Update the version in `pyproject.toml`, `CITATION.cff`,
the README install line and the two `ARG CONSTRAINTS=` lines of
`src/lasagna/deploy/Dockerfile` (the image installs the same versions), add a
`## [<version>] - YYYY-MM-DD` section to CHANGELOG.md, and remove the old
constraints file. `tests/test_release_check.py` and
`tests/test_the_image_is_built_from_the_tested_versions.py` fail until they all
agree.
