"""OpenEvolve test suite.

Intentionally empty: importing test classes here breaks pytest collection when
any single test module fails to import, and the auto-generated re-exports that
used to live here referenced classes that do not exist.
"""
