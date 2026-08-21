"""LeanAIDE integrations package.

Collection of standalone modules for Lean 4 proof generation / verification,
MCTS-MDAP planning, evolutionary strategies, and assorted adapters.

NOTE: This package is used as a flat set of importable scripts (modules import
each other and `lean4_integration*` by bare module name). Several modules are
stubs or require external services (Lean 4 toolchain, OpenAI API, CrewAI,
FastAPI) that are NOT satisfied in this repo. See ACTUAL_STATUS.md.
"""
