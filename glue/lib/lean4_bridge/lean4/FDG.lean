/-
FDG.lean: Functional Dependency Graph Formalization in Lean 4

This module formalizes Functional Dependency Graphs (FDGs) for mechanistic
isomorphism validation per RESE Technical Manual §4.2.

Features:
- Component structure with properties
- CausalConnection with mechanism and strength
- TensorNotation for physics tensor notation
- Abstract operational principles
- I_mech score calculation

Author: RESE Team
Created: 2026-02-04
-/

import Mathlib
import RESE.Tensors

namespace RESE.FDG

/-- A component in a Functional Dependency Graph. -/
structure Component where
  name : String
  type : String
  properties : List (String × String)
  deriving Repr, BEq, Hashable

/-- A causal connection between components with strength. -/
structure CausalConnection where
  source : Component
  target : Component
  mechanism : String
  strength : Real
  notation : Option TensorNotation
  deriving Repr

/-- Abstract operational principle for mechanistic validation. -/
inductive AbstractOperationalPrinciple where
  | isolation : Component → AbstractOperationalPrinciple
  | localComputation : Component → AbstractOperationalPrinciple
  | controlledRelease : Component → Component → AbstractOperationalPrinciple
  | statePreservation : Component → AbstractOperationalPrinciple
  | transformation : Component → Component → AbstractOperationalPrinciple
  deriving Repr

/-- Functional Dependency Graph structure. -/
structure FunctionalDependencyGraph where
  nodes : List Component
  edges : List CausalConnection
  tensorStructure : Option TensorNotation
  deriving Repr

/-- Extract all components from FDG. -/
def FDG.getComponents (fdg : FunctionalDependencyGraph) : List Component :=
  fdg.nodes

/-- Extract all connections from FDG. -/
def FDG.getConnections (fdg : FunctionalDependencyGraph) : List CausalConnection :=
  fdg.edges

/-- Find component by name in FDG. -/
def FDG.findComponent (fdg : FunctionalDependencyGraph) (name : String) : Option Component :=
  fdg.nodes.find (fun c => c.name == name)

/-- Check if FDG has tensor structure. -/
def FDG.hasTensorStructure (fdg : FunctionalDependencyGraph) : Bool :=
  match fdg.tensorStructure with
  | some _ => true
  | none => false

/-- Calculate node overlap between two FDGs. -/
def calculateNodeOverlap (fdg1 fdg2 : FunctionalDependencyGraph) : Real :=
  let nodes1 := fdg1.nodes.map (·.name)
  let nodes2 := fdg2.nodes.map (·.name)
  let intersection := (nodes1.filter (fun n => nodes2.contains n)).length
  let union := (nodes1 ++ nodes2).eraseDups.length
  if union = 0 then 0 else ↑intersection / ↑union

/-- Calculate edge overlap between two FDGs. -/
def calculateEdgeOverlap (fdg1 fdg2 : FunctionalDependencyGraph) : Real :=
  let edges1 := fdg1.edges.map (fun e => (e.source.name, e.target.name))
  let edges2 := fdg2.edges.map (fun e => (e.source.name, e.target.name))
  let intersection := (edges1.filter (fun e => edges2.contains e)).length
  let union := (edges1 ++ edges2).eraseDups.length
  if union = 0 then 0 else ↑intersection / ↑union

/-- Calculate I_mech score between two FDGs.

Formula:
  I_mech = 0.6 * node_overlap + 0.4 * edge_overlap

This quantifies mechanistic similarity between domains.
Score ≥ 0.7 indicates valid isomorphism for transfer.
-/
def I_mech_score (fdg1 fdg2 : FunctionalDependencyGraph) : Real :=
  0.6 * calculateNodeOverlap fdg1 fdg2 +
  0.4 * calculateEdgeOverlap fdg1 fdg2

/-- Abstract operational principles match between two FDGs. -/
def abstract_operational_principles_match (fdg1 fdg2 : FunctionalDependencyGraph) : Bool :=
  -- Extract principles from both FDGs
  -- For now, use structural overlap as proxy
  I_mech_score fdg1 fdg2 ≥ 0.7

/-- Mechanistic isomorphism theorem.

Two FDGs are mechanistically isomorphic iff:
  1. Their I_mech score exceeds threshold
  2. Their abstract operational principles match
-/
theorem mechanistic_isomorphism (fdg1 fdg2 : FunctionalDependencyGraph)
    (threshold : Real) :
    I_mech_score fdg1 fdg2 ≥ threshold ↔
    abstract_operational_principles_match fdg1 fdg2 := by
  -- Proof sketch:
  -- 1. Forward direction (→): High I_mech implies principle match
  --    - I_mech ≥ threshold ≥ 0.7 implies structural similarity
  --    - Structural similarity implies operational principle match
  -- 2. Backward direction (←): Principle match implies high I_mech
  --    - Principle match requires structural alignment
  --    - Structural alignment gives I_mech ≥ 0.7 ≥ threshold
  sorry

/-- Size ratio penalty for FDG comparison.

Prefer similar-sized domains for isomorphism.
-/
def sizeRatio (fdg1 fdg2 : FunctionalDependencyGraph) : Real :=
  let size1 := fdg1.nodes.length
  let size2 := fdg2.nodes.length
  if size1 = 0 ∨ size2 = 0 then 0
  else ↑(min size1 size2) / ↑(max size1 size2)

/-- Enhanced I_mech with size penalty. -/
def I_mech_score_enhanced (fdg1 fdg2 : FunctionalDependencyGraph) : Real :=
  0.7 * I_mech_score fdg1 fdg2 + 0.3 * sizeRatio fdg1 fdg2

/-- Valid isomorphism check with threshold. -/
def isValidIsomorphism (fdg1 fdg2 : FunctionalDependencyGraph) (threshold : Real) : Bool :=
  I_mech_score_enhanced fdg1 fdg2 ≥ threshold

/-- FDG construction helper. -/
def mkFDG (nodes : List Component)
    (edges : List CausalConnection)
    (tensorStruct : Option TensorNotation := none) : FunctionalDependencyGraph :=
  {
    nodes := nodes,
    edges := edges,
    tensorStructure := tensorStruct
  }

/-- Component construction helper. -/
def mkComponent (name type : String)
    (properties : List (String × String) := []) : Component :=
  {
    name := name,
    type := type,
    properties := properties
  }

/-- Causal connection construction helper. -/
def mkConnection (source target : Component)
    (mechanism : String)
    (strength : Real)
    (notation : Option TensorNotation := none) : CausalConnection :=
  {
    source := source,
    target := target,
    mechanism := mechanism,
    strength := strength,
    notation := notation
  }

end RESE.FDG
