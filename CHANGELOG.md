# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Action checks: PHYSICAL, COVERT and WORK are settled by a 3d6 roll against an LLM-assessed difficulty and edge, with five graded results from critical success to critical failure. The executor's narration writes the rolled result instead of deciding success, and a covert act's exposure follows the roll when someone is present to notice.
- The narrative feed shows each rolled result's tier, the dice and the assessor's reason.
- The stage narrator: after each step a background writer turns the step's record into a play script for the audience, in acts and scenes, with stage directions and narration around the characters' own lines quoted verbatim. Characters never read it. `engine.narration_enabled` turns it off.
- A theater view beside the record in the world screen: the script up to the step the map shows; a scene heading pans the map to its place, and chase follows each new script's busiest scene.
- `python -m tuning stats <world-id>`: action types, success rates and roll tiers counted from snapshots, without calling a model.

### Fixed

- The tuning action harness read a field `ActionResult` doesn't have when reporting observations.

## [0.1.0] - 2026-10-08

First public release.

### Added

- Theme-driven world building: a single theme produces locations, characters, personas and initial relationships.
- Character cognition: perception, emotions, needs, goals, memory (an objective event stream and a subjective experience stream), relationships, reflection and decision-making, orchestrated in stages.
- Runtime engine: clock, scheduling, environment, messaging, action execution and arbitration, world event injection, NPCs.
- Pluggable infrastructure: any OpenAI-compatible LLM endpoint, with per-scene model routing; swappable embeddings, vector store and snapshots.
- Web UI: world management, live observation, replay, 2D map, relationship graph, director console.
- Developer tools: LLM call tracing, narrative quality audits, prompt replay.
- One-command Docker Compose deployment with a guided first-run setup.
- Two completed example worlds, imported on first start and available for browsing and replay without API keys.

[Unreleased]: https://github.com/dreamscroll-lab/asamana/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dreamscroll-lab/asamana/releases/tag/v0.1.0
