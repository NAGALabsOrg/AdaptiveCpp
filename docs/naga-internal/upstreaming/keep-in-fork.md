# Upstreaming triage: keep-in-fork decisions

Pins: U `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc`, N `0e2bac974c40554f62286cbffe1f06c87cc9caa9`, B `9ec394efa584ced004d644af2194d2133e111d26`; merge tree M `350667e47d62407ad9bef5977bdab5a632ae8e87` (see baseline.md). Status: first run, before Review 1. Every decision below is a proposal for Jack; the boundaries are open items U10 in decisions.md.

Block format as in candidates.md: `paths` are whole paths (`dir/**` = every difference-set path under `dir`, never the records directory `docs/naga-internal/upstreaming/`, which is outside the difference set), `hunks` are `path @start,count` hunks of a split path (new-file coordinates of `git diff --no-renames -U0 U M -- <path>`).

Reasons are the plan's three: fork identity, transitional, scope. Counts: 3 decisions; fork identity 2 (KF1, KF2), transitional 1 (KF3), scope 0 (none).

## KF1 Fork project tooling

- reason: fork identity
- sentence: The project bank and its commit hook (`.project/`) and the `.gitignore` line for them are tooling of this fork's own operation and mean nothing to AdaptiveCpp.
- size: files 3, lines +55 / -0
- changes:

```assignment KF1
paths
  .project/**
hunks
  .gitignore @57,5
```

## KF2 NAGA internal documents

- reason: fork identity
- sentence: The kernel-language, graph-API, vocabulary and research documents under `docs/naga-internal/` are NAGA Labs' own design documents, not documentation of AdaptiveCpp.
- size: files 7, lines +1301 / -0
- changes:

```assignment KF2
paths
  docs/naga-internal/**
```

## KF3 Patches and pins for the fork's LLVM snapshot

- reason: transitional
- sentence: The patches to the pinned AMD LLVM 23 snapshot and the file that pins its sources are a workaround for the fork's own build environment and apply to no tree AdaptiveCpp ships.
- size: files 10, lines +651 / -0
- seam: candidate C20 reads these files (`M:packages/core/fetch-sources.nu:70` opens `pins.toml`, `:71` names the `patches` directory); see C20 in candidates.md and unresolved U06.
- changes:

```assignment KF3
paths
  packages/core/patches/**
  packages/core/pins.toml
```
