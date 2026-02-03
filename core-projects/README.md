# Core Projects Directory

**STATUS: READ-ONLY. IMMUTABLE.**

## Purpose

This directory contains third-party vendor libraries and Open Source systems that are integrated into the OpenEvolve Federation.

## The Law of the "Air Gap"

**CRITICAL:** Do NOT import, include, or require files from this directory in your glue code.

- These projects are treated as external vendor libraries
- Direct dependencies create coupling that breaks when upstream projects update
- All integration must happen through the Adapter Layer

## Directory Structure

Each core project should have its own subdirectory:
```
core-projects/
├── z3/              # Z3 Theorem Prover
├── lean4/           # Lean 4 Proof Assistant
├── leanaide/        # LeanAide AI Assistant
└── ...
```

## Protocol

1. **READ ONLY**: Never modify source code in this directory
2. **VERSION PINNING**: Track exact versions in documentation
3. **UPSTREAM FIRST**: All changes must go to the original project
4. **ADAPTER PATTERN**: All integration happens via `/glue/adapters/{project}-adapter/`

## Violation Consequences

Importing from this directory violates **Law #1: The Air Gap** and creates technical debt that will break when upstream projects update.

Always rewrite utilities in the Glue Layer rather than importing from Core Projects.
