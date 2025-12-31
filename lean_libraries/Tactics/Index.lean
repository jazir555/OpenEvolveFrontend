/-!
# Physics Tactics Library Index

This file imports and exports all physics-specific tactics for Lean 4.
Import this file to get access to all custom tactics.

## Usage

```lean
import LeanLraries.Tactics

-- All tactics are now available:
-- quantum_normalize, apply_unitary, compute_expectation, spectral_decompose
-- tensor_simplify, covariant_derivative, raise_lower_indices, curvature_identities
-- ensemble_average, thermodynamic_limit, maxwell_boltzmann, canonical_transform
-- asymptotic_expand, interval_arithmetic, perturbation_theory
```

## Library Structure

* `Quantum.lean` - Quantum mechanics tactics
* `Relativity.lean` - General relativity tactics
* `StatMech.lean` - Statistical mechanics tactics
* `Analysis.lean` - Mathematical analysis tactics

-/

/-! Import all tactic libraries -/

import LeanLraries.Tactics.Quantum
import LeanLraries.Tactics.Relativity
import LeanLraries.Tactics.StatMech
import LeanLraries.Tactics.Analysis

/-! Export all namespaces -/

open Quantum
open Relativity
open StatMech
open Analysis

/-! Quick Reference -/

#align quantum_normalize Quantum.quantumNormalize
#align apply_unitary Quantum.applyUnitary
#align compute_expectation Quantum.computeExpectation
#align spectral_decompose Quantum.spectralDecompose
#align quantum_simp Quantum.quantumSimp

#align tensor_simplify Relativity.tensorSimplify
#align covariant_derivative Relativity.covariantDerivative
#align raise_lower_indices Relativity.raiseLowerIndices
#align curvature_identities Relativity.curvatureIdentities
#align relativity_simp Relativity.relativitySimp
#align einstein_simplify Relativity.einsteinSimplify

#align ensemble_average StatMech.ensembleAverage
#align thermodynamic_limit StatMech.thermodynamicLimit
#align maxwell_boltzmann StatMech.maxwellBoltzmann
#align canonical_transform StatMech.canonicalTransform
#align statmech_simp StatMech.statmechSimp
#align canonical_simplify StatMech.canonicalSimplify

#align asymptotic_expand Analysis.asymptoticExpand
#align interval_arithmetic Analysis.intervalArithmetic
#align perturbation_theory Analysis.perturbationTheory
#align analysis_simp Analysis.analysisSimp
#align series_expand Analysis.seriesExpand
#align rigorous_bound Analysis.rigorousBound
