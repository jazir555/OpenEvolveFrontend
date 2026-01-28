// Lean compiler output
// Module: Mathlib.LinearAlgebra.AffineSpace.Combination
// Imports: public import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Indicator public import Mathlib.Algebra.Module.BigOperators public import Mathlib.LinearAlgebra.AffineSpace.AffineSubspace.Basic public import Mathlib.LinearAlgebra.Finsupp.LinearCombination public import Mathlib.Tactic.FinCases
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lp_mathlib_Pi_single___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubOfPoint___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationSingleWeights(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instAddTorsorForall___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_smulRight___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instAddTorsorForall___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_addCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_addGroupIsAddTorsor___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationSingleWeights___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubVSubWeights(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instAddTorsorForall___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instAddTorsorForall(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubOfPoint___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_eval(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationSingleWeights___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationLineMapWeights(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubVSubWeights___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationSingleWeights___redArg___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__1___closed__0;
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubOfPoint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instAddTorsor___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_snd___lam__0(lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_fst___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubOfPoint___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationLineMapWeights___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubOfPoint___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_5);
x_8 = lean_apply_1(x_1, x_5);
x_9 = lean_apply_2(x_2, x_8, x_3);
x_10 = lp_mathlib_LinearMap_smulRight___redArg(x_4, x_7, x_9);
x_11 = lean_apply_1(x_10, x_6);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubOfPoint___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_3, 1);
lean_inc(x_8);
lean_dec_ref(x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Finset_weightedVSubOfPoint___redArg___lam__0), 6, 4);
lean_closure_set(x_9, 0, x_5);
lean_closure_set(x_9, 1, x_8);
lean_closure_set(x_9, 2, x_6);
lean_closure_set(x_9, 3, x_2);
x_10 = lp_mathlib_LinearMap_addCommMonoid___redArg(x_7);
x_11 = lp_mathlib_Finset_sum___redArg(x_10, x_4, x_9);
lean_dec_ref(x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubOfPoint(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Finset_weightedVSubOfPoint___redArg(x_5, x_6, x_7, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubOfPoint___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Finset_weightedVSubOfPoint(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instAddTorsorForall___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instAddTorsorForall___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_instAddTorsorForall___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instAddTorsorForall___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_3 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_2);
lean_dec_ref(x_2);
x_4 = lp_mathlib_addGroupIsAddTorsor___redArg(x_3);
lean_dec_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Finset_instAddTorsorForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lp_mathlib_Pi_instAddTorsor___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instAddTorsorForall(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_instAddTorsorForall___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationSingleWeights___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationSingleWeights___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_affineCombinationSingleWeights___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationSingleWeights___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_6);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_11 = lean_ctor_get(x_10, 1);
lean_inc_ref(x_11);
lean_dec_ref(x_10);
x_12 = lean_ctor_get(x_11, 2);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Finset_affineCombinationSingleWeights___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_13, 0, x_9);
x_14 = lp_mathlib_Pi_single___redArg(x_13, x_2, x_3, x_12, x_4);
lean_dec(x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationSingleWeights(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Finset_affineCombinationSingleWeights___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubVSubWeights___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_1);
x_6 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_7 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_6);
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_7, 2);
lean_inc(x_8);
lean_dec_ref(x_7);
lean_inc(x_5);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_9 = lp_mathlib_Finset_affineCombinationSingleWeights___redArg(x_1, x_2, x_3, x_5);
x_10 = lp_mathlib_Finset_affineCombinationSingleWeights___redArg(x_1, x_2, x_4, x_5);
x_11 = lean_apply_2(x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_weightedVSubVSubWeights(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Finset_weightedVSubVSubWeights___redArg(x_2, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationLineMapWeights___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_inc_ref(x_1);
x_7 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_8);
lean_inc_ref(x_9);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_9);
x_11 = lean_ctor_get(x_10, 1);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_9);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc(x_6);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_14 = lp_mathlib_Finset_weightedVSubVSubWeights___redArg(x_1, x_2, x_4, x_3, x_6);
x_15 = lean_apply_2(x_13, x_5, x_14);
x_16 = lp_mathlib_Finset_affineCombinationSingleWeights___redArg(x_1, x_2, x_3, x_6);
x_17 = lean_apply_2(x_11, x_15, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_affineCombinationLineMapWeights(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Finset_affineCombinationLineMapWeights___redArg(x_2, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lp_mathlib_Finset_weightedVSubOfPoint___redArg(x_1, x_2, x_3, x_4, x_7, x_8);
x_10 = lean_apply_1(x_9, x_5);
return x_10;
}
}
static lean_object* _init_lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_fst___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc(x_4);
x_6 = lean_apply_1(x_1, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_4);
x_8 = lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__1___closed__0;
x_9 = lp_mathlib_LinearMap_comp___redArg(x_7, x_8);
lean_inc_ref(x_5);
x_10 = lean_apply_1(x_9, x_5);
x_11 = lp_mathlib_LinearMap_snd___lam__0(x_5);
lean_dec_ref(x_5);
x_12 = lean_apply_2(x_2, x_10, x_11);
x_13 = lean_apply_2(x_3, x_6, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_1, 2);
lean_inc(x_7);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__0), 6, 5);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_3);
lean_closure_set(x_8, 3, x_4);
lean_closure_set(x_8, 4, x_5);
x_9 = lean_alloc_closure((void*)(lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__1), 5, 3);
lean_closure_set(x_9, 0, x_5);
lean_closure_set(x_9, 1, x_7);
lean_closure_set(x_9, 2, x_2);
x_10 = lp_mathlib_LinearMap_addCommMonoid___redArg(x_6);
x_11 = lp_mathlib_Finset_sum___redArg(x_10, x_4, x_9);
lean_dec_ref(x_10);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_8);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_AffineMap_weightedVSubOfPoint___redArg(x_5, x_6, x_7, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AffineMap_weightedVSubOfPoint___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_AffineMap_weightedVSubOfPoint(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_4);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Indicator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_AffineSpace_AffineSubspace_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_AffineSpace_Combination(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Indicator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_AffineSpace_AffineSubspace_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__1___closed__0 = _init_lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__1___closed__0();
lean_mark_persistent(lp_mathlib_AffineMap_weightedVSubOfPoint___redArg___lam__1___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
