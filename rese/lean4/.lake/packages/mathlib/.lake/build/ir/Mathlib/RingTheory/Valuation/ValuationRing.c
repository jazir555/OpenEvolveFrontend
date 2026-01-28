// Lean compiler output
// Module: Mathlib.RingTheory.Valuation.ValuationRing
// Imports: public import Init public import Mathlib.RingTheory.Bezout public import Mathlib.RingTheory.LocalRing.Basic public import Mathlib.RingTheory.Localization.FractionRing public import Mathlib.RingTheory.Localization.Integer public import Mathlib.RingTheory.Valuation.Integers public import Mathlib.Tactic.LinearCombination public import Mathlib.Tactic.FieldSimp
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
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_commGroupWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_commGroupWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instOneValueGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeSup_toMax___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_completeLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_instOmegaCompletePartialOrder___redArg(lean_object*);
lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeInf_toMin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLEValueGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instMulValueGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInhabitedValueGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instOneValueGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInhabitedValueGroup___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instMulValueGroup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instMulValueGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInhabitedValueGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instZeroValueGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_commGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instZeroValueGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLEValueGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instOneValueGroup___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instMulValueGroup___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instZeroValueGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInhabitedValueGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_2);
x_4 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_3);
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_4);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInhabitedValueGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instInhabitedValueGroup___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInhabitedValueGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instInhabitedValueGroup(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLEValueGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLEValueGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instLEValueGroup(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instZeroValueGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_2);
x_4 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_3);
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_4);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instZeroValueGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instZeroValueGroup___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instZeroValueGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instZeroValueGroup(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instOneValueGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_3);
x_5 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_5, 2);
lean_inc(x_6);
lean_dec_ref(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instOneValueGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instOneValueGroup___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instOneValueGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instOneValueGroup(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instMulValueGroup___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instMulValueGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_2);
x_4 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_3);
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ValuationRing_instMulValueGroup___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instMulValueGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instMulValueGroup___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instMulValueGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instMulValueGroup(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Field_toSemifield___redArg(x_1);
x_3 = lp_mathlib_Semifield_toCommGroupWithZero___redArg(x_2);
x_4 = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(x_3);
x_5 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_4);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ValuationRing_instInvValueGroup___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instInvValueGroup___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_instInvValueGroup(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instInvValueGroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ValuationRing_instInvValueGroup___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_commGroupWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_ValuationRing_instMulValueGroup___redArg(x_1);
lean_inc_ref(x_1);
x_3 = lp_mathlib_ValuationRing_instOneValueGroup___redArg(x_1);
lean_inc(x_3);
lean_inc(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_2);
lean_closure_set(x_4, 2, x_3);
lean_inc(x_3);
lean_inc(x_2);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
lean_inc_ref(x_1);
x_6 = lp_mathlib_ValuationRing_instZeroValueGroup___redArg(x_1);
lean_inc_ref(x_5);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
x_8 = lp_mathlib_ValuationRing_instInvValueGroup___redArg(x_1);
lean_dec_ref(x_1);
lean_inc(x_8);
x_9 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_5);
lean_closure_set(x_9, 2, x_8);
lean_inc(x_2);
lean_inc(x_3);
x_10 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_3);
lean_closure_set(x_10, 2, x_2);
lean_inc(x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_3);
lean_closure_set(x_11, 2, x_2);
lean_closure_set(x_11, 3, x_8);
lean_closure_set(x_11, 4, x_10);
x_12 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_12, 0, x_7);
lean_ctor_set(x_12, 1, x_8);
lean_ctor_set(x_12, 2, x_9);
lean_ctor_set(x_12, 3, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_commGroupWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_commGroupWithZero___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_commGroupWithZero___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ValuationRing_commGroupWithZero(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_decidableLTOfDecidableLE___redArg(x_1, x_2, x_3);
if (x_4 == 0)
{
uint8_t x_5; 
x_5 = lp_mathlib_decidableEqOfDecidableLE___redArg(x_1, x_2, x_3);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = 2;
return x_6;
}
else
{
uint8_t x_7; 
x_7 = 1;
return x_7;
}
}
else
{
uint8_t x_8; 
lean_dec_ref(x_1);
x_8 = 0;
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
lean_inc_ref(x_3);
x_7 = lp_mathlib_Semiring_toModule___redArg(x_3);
x_8 = lp_mathlib_Submodule_completeLattice(lean_box(0), lean_box(0), x_3, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_3);
lean_inc_ref(x_8);
x_9 = lp_mathlib_CompleteLattice_instOmegaCompletePartialOrder___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_8);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_12);
x_13 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc_ref(x_14);
x_15 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_15);
lean_dec_ref(x_12);
lean_inc_ref(x_2);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_16, 0, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_10);
x_17 = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_10);
lean_closure_set(x_17, 2, x_2);
lean_inc_ref(x_2);
x_18 = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_10);
lean_closure_set(x_18, 2, x_2);
x_19 = lp_mathlib_SemilatticeInf_toMin___redArg(x_13);
x_20 = lp_mathlib_SemilatticeSup_toMax___redArg(x_15);
x_21 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_21, 0, x_14);
lean_ctor_set(x_21, 1, x_19);
lean_ctor_set(x_21, 2, x_20);
lean_ctor_set(x_21, 3, x_16);
lean_ctor_set(x_21, 4, x_2);
lean_ctor_set(x_21, 5, x_17);
lean_ctor_set(x_21, 6, x_18);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ValuationRing_instLinearOrderIdealOfDecidableLE___redArg(x_2, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Bezout(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_LocalRing_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Integer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Valuation_Integers(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_LinearCombination(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FieldSimp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Valuation_ValuationRing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Bezout(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_LocalRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_Integer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Valuation_Integers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FieldSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
