# Contribution guidelines

Contributing to this project should be as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## GitHub is used for everything

GitHub is used to host code, to track issues and feature requests, as well as accept pull requests.

Pull requests are the best way to propose changes to the codebase.

1. Fork the repo and create your branch from `main`.
2. If you've changed something, update the documentation.
3. Make sure your code lints (`mise run lint`) and the tests pass (`pytest`).
4. Issue that pull request!

## Mutation testing

Mutation testing with [mutmut](https://mutmut.readthedocs.io/) shows tests that would not notice a changed line. CI runs it in the Mutations job on pushes to main and manual runs (not on pull requests: about 8 minutes on 8 local cores, longer on CI runners); run it locally with:

```bash
poetry run mutmut run
# check results
poetry run mutmut results
```

## Configuration and prompts

The packaged prompts in `recording_transcriber/prompts/` are generic. The production configuration, glossary and specialized prompts live in `examples/medical-advisory-board/`; the README (Releases and deployments) explains how to build an app folder from an example. `config.example.yaml` is the commented reference of every key.

## Never commit recordings or meeting content

Recordings, transcripts, protocols and attendee lists are confidential. Never add them to an issue, a pull request or a test: use short, invented text and fictional names in tests. The `.gitignore` excludes audio files, Word files and the local working folders (`samples/`, `runs/`, `report/`, `scratch/`, `notes/`).

## Any contributions you make will be under the GNU GPL v3

In short, when you submit code changes, your submissions are understood to be under the
same [GNU General Public License v3](https://choosealicense.com/licenses/gpl-3.0/) (or later) that covers the project. Feel free to contact the
maintainers if that's a concern.

## Report bugs using GitHub's [issues](../../issues)

GitHub issues are used to track public bugs.
Report a bug by [opening a new issue](../../issues/new/choose); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.
