/**
 * LeanAide Parameter Schema
 */

import { ParameterSchema } from '@/types/plugin';

export const leanaideParameters: ParameterSchema[] = [
  {
    name: 'theorem',
    type: 'textarea',
    label: 'Theorem Statement',
    description: 'The theorem to prove or verify',
    required: true,
    multiline: true,
    placeholder: 'Enter theorem statement...',
  },
  {
    name: 'mode',
    type: 'select',
    label: 'Mode',
    description: 'LeanAide operation mode',
    required: true,
    defaultValue: 'prove',
    options: [
      { value: 'prove', label: 'Prove Theorem' },
      { value: 'verify', label: 'Verify Proof' },
      { value: 'suggest', label: 'Suggest Proof Strategy' },
      { value: 'generate', label: 'Generate Proof' },
      { value: 'tactics', label: 'Apply Tactics' },
      { value: 'decompose', label: 'Decompose Goal' },
    ],
  },
  {
    name: 'leanVersion',
    type: 'select',
    label: 'Lean Version',
    description: 'Lean formal prover version',
    defaultValue: 'lean4',
    options: [
      { value: 'lean4', label: 'Lean 4' },
      { value: 'lean3', label: 'Lean 3' },
    ],
  },
  {
    name: 'proofDepth',
    type: 'select',
    label: 'Proof Depth',
    description: 'Level of detail in proof',
    options: [
      { value: 'sketch', label: 'Proof Sketch' },
      { value: 'intermediate', label: 'Intermediate Steps' },
      { value: 'detailed', label: 'Detailed Proof' },
      { value: 'formal', label: 'Fully Formal' },
    ],
  },
  {
    name: 'tactics',
    type: 'multiselect',
    label: 'Allowed Tactics',
    description: 'Restrict to specific tactics',
    options: [
      { value: 'intro', label: 'intro' },
      { value: 'apply', label: 'apply' },
      { value: 'exact', label: 'exact' },
      { value: 'rewrite', label: 'rewrite' },
      { value: 'induction', label: 'induction' },
      { value: 'cases', label: 'cases' },
      { value: 'simp', label: 'simp' },
    ],
  },
  {
    name: 'autoSimplify',
    type: 'boolean',
    label: 'Auto Simplify',
    description: 'Automatically simplify goals',
    defaultValue: true,
  },
  {
    name: 'generateLeanCode',
    type: 'boolean',
    label: 'Generate Lean Code',
    description: 'Output actual Lean 4 code',
    defaultValue: true,
  },
];
