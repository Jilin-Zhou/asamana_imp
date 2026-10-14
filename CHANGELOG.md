# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Action checks: PHYSICAL, COVERT and WORK are settled by a 3d6 roll against an LLM-assessed difficulty and edge, with five graded results from critical success to critical failure. The executor's narration writes the rolled result instead of deciding success, and a covert act's exposure follows the roll when someone is present to notice.
- The narrative feed shows each rolled result's tier, the dice and the assessor's reason.
- Character abilities: five general attributes (体魄 / 身手 / 心智 / 口才 / 胆识, 1–5) and up to four named skills (1–3), rated for the whole cast in one world-building call (`ability_design` scene) so they compare across characters. A check's modifier now comes from the attribute and skill it draws on, less the opponent's resisting attribute, plus a situational edge narrowed to −2..+2. Characters read their own strengths and weaknesses as words in their persona; the character card shows them, and the feed itemizes what went into each roll. Worlds built before this load everyone as ordinary (all 3, no skills).
- Money: theme analysis names the world's currency, unit and up to six reference prices, and the ability-design call hands each character starting cash on one scale. Money is a divisible thing in the world (one pile per holder and per place) that splits and merges as it changes hands; only its holder sees how much is in a purse. PHYSICAL acts can take or hand over part of a pile (the judge rules how much, within what was at stake), and errands can carry a sum.
- TRADE: a new action type for an agreed exchange with someone present. The counterpart's turn is spent as in TALK; a neutral narrator writes the haggling and rules whether the deal was struck and for how much, and the exchange lands all at once or not at all. The feed shows the haggling and how the deal ended; the map plays it as speech bubbles.
- Weather: one sky over the world, chosen climate at build and moved on by a seeded rule every three world hours. Every situation header (and so every cognition and judge prompt) carries it in words, a turn of the sky is noticed everywhere, heavy rain and snow slow travel, and each step reports the sky to observers.
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
