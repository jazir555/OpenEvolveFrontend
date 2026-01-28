// Lean compiler output
// Module: Mathlib.Algebra.Field.Opposite
// Imports: public import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.Ring.Opposite public import Mathlib.Data.Int.Cast.Lemmas
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
lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionRing(lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instField(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instField(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNNRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemifield(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRatCast(lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionRing___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionRing(lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNNRatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_instGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemifield(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNNRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instField___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionSemiring(lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_instSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNNRatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemifield___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionRing___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNNRatCast___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNNRatCast(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instNNRatCast___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNNRatCast___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNNRatCast(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instNNRatCast___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRatCast___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRatCast(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instRatCast___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRatCast___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRatCast(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instRatCast___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_2, x_3);
x_9 = lean_apply_2(x_7, x_4, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 4);
lean_inc(x_4);
lean_inc_ref(x_2);
x_5 = lp_mathlib_MulOpposite_instSemiring___redArg(x_2);
x_6 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_1);
x_7 = lp_mathlib_MulOpposite_instGroupWithZero___redArg(x_6);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 2);
lean_inc(x_9);
lean_dec_ref(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_3);
lean_inc(x_4);
x_11 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_12, 0, x_4);
x_13 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_13, 0, x_5);
lean_ctor_set(x_13, 1, x_8);
lean_ctor_set(x_13, 2, x_9);
lean_ctor_set(x_13, 3, x_10);
lean_ctor_set(x_13, 4, x_12);
lean_ctor_set(x_13, 5, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instDivisionSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionRing___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_2, x_3);
x_10 = lean_apply_2(x_8, x_4, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 5);
lean_inc(x_4);
lean_inc_ref(x_2);
x_5 = lp_mathlib_MulOpposite_instRing___redArg(x_2);
x_6 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_1);
x_7 = !lean_is_exclusive(x_1);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_8 = lean_ctor_get(x_1, 7);
lean_dec(x_8);
x_9 = lean_ctor_get(x_1, 6);
lean_dec(x_9);
x_10 = lean_ctor_get(x_1, 5);
lean_dec(x_10);
x_11 = lean_ctor_get(x_1, 4);
lean_dec(x_11);
x_12 = lean_ctor_get(x_1, 3);
lean_dec(x_12);
x_13 = lean_ctor_get(x_1, 2);
lean_dec(x_13);
x_14 = lean_ctor_get(x_1, 1);
lean_dec(x_14);
x_15 = lean_ctor_get(x_1, 0);
lean_dec(x_15);
lean_inc_ref(x_6);
x_16 = lp_mathlib_MulOpposite_instDivisionSemiring___redArg(x_6);
x_17 = lean_ctor_get(x_16, 1);
lean_inc(x_17);
x_18 = lean_ctor_get(x_16, 2);
lean_inc(x_18);
x_19 = lean_ctor_get(x_16, 4);
lean_inc(x_19);
lean_dec_ref(x_16);
x_20 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_6, 4);
lean_inc(x_21);
lean_dec_ref(x_6);
x_22 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_22, 0, x_3);
lean_inc(x_4);
x_23 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionRing___redArg___lam__1), 4, 2);
lean_closure_set(x_23, 0, x_2);
lean_closure_set(x_23, 1, x_4);
x_24 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_24, 0, x_4);
x_25 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_25, 0, x_20);
lean_closure_set(x_25, 1, x_21);
lean_ctor_set(x_1, 7, x_23);
lean_ctor_set(x_1, 6, x_25);
lean_ctor_set(x_1, 5, x_24);
lean_ctor_set(x_1, 4, x_19);
lean_ctor_set(x_1, 3, x_22);
lean_ctor_set(x_1, 2, x_18);
lean_ctor_set(x_1, 1, x_17);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
lean_dec(x_1);
lean_inc_ref(x_6);
x_26 = lp_mathlib_MulOpposite_instDivisionSemiring___redArg(x_6);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 2);
lean_inc(x_28);
x_29 = lean_ctor_get(x_26, 4);
lean_inc(x_29);
lean_dec_ref(x_26);
x_30 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_30);
x_31 = lean_ctor_get(x_6, 4);
lean_inc(x_31);
lean_dec_ref(x_6);
x_32 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_32, 0, x_3);
lean_inc(x_4);
x_33 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionRing___redArg___lam__1), 4, 2);
lean_closure_set(x_33, 0, x_2);
lean_closure_set(x_33, 1, x_4);
x_34 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_34, 0, x_4);
x_35 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_35, 0, x_30);
lean_closure_set(x_35, 1, x_31);
x_36 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_36, 0, x_5);
lean_ctor_set(x_36, 1, x_27);
lean_ctor_set(x_36, 2, x_28);
lean_ctor_set(x_36, 3, x_32);
lean_ctor_set(x_36, 4, x_29);
lean_ctor_set(x_36, 5, x_34);
lean_ctor_set(x_36, 6, x_35);
lean_ctor_set(x_36, 7, x_33);
return x_36;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instDivisionRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemifield___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_MulOpposite_instSemiring___redArg(x_2);
x_5 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_1);
lean_inc_ref(x_5);
x_6 = lp_mathlib_MulOpposite_instDivisionSemiring___redArg(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 2);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 4);
lean_inc(x_9);
lean_dec_ref(x_6);
x_10 = !lean_is_exclusive(x_5);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_11 = lean_ctor_get(x_5, 0);
x_12 = lean_ctor_get(x_5, 4);
x_13 = lean_ctor_get(x_5, 5);
lean_dec(x_13);
x_14 = lean_ctor_get(x_5, 3);
lean_dec(x_14);
x_15 = lean_ctor_get(x_5, 2);
lean_dec(x_15);
x_16 = lean_ctor_get(x_5, 1);
lean_dec(x_16);
x_17 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_17, 0, x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_18, 0, x_11);
lean_closure_set(x_18, 1, x_12);
lean_ctor_set(x_5, 5, x_18);
lean_ctor_set(x_5, 4, x_9);
lean_ctor_set(x_5, 3, x_17);
lean_ctor_set(x_5, 2, x_8);
lean_ctor_set(x_5, 1, x_7);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_19 = lean_ctor_get(x_5, 0);
x_20 = lean_ctor_get(x_5, 4);
lean_inc(x_20);
lean_inc(x_19);
lean_dec(x_5);
x_21 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_21, 0, x_3);
x_22 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_22, 0, x_19);
lean_closure_set(x_22, 1, x_20);
x_23 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_23, 0, x_4);
lean_ctor_set(x_23, 1, x_7);
lean_ctor_set(x_23, 2, x_8);
lean_ctor_set(x_23, 3, x_21);
lean_ctor_set(x_23, 4, x_9);
lean_ctor_set(x_23, 5, x_22);
return x_23;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemifield(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instSemifield___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instField___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_MulOpposite_instRing___redArg(x_2);
x_5 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
lean_inc_ref(x_5);
x_6 = lp_mathlib_MulOpposite_instDivisionRing___redArg(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 2);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 4);
lean_inc(x_9);
x_10 = lean_ctor_get(x_6, 5);
lean_inc(x_10);
lean_dec_ref(x_6);
x_11 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_5);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_11, 4);
lean_inc(x_13);
lean_dec_ref(x_11);
x_14 = !lean_is_exclusive(x_5);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_15 = lean_ctor_get(x_5, 0);
x_16 = lean_ctor_get(x_5, 5);
x_17 = lean_ctor_get(x_5, 7);
lean_dec(x_17);
x_18 = lean_ctor_get(x_5, 6);
lean_dec(x_18);
x_19 = lean_ctor_get(x_5, 4);
lean_dec(x_19);
x_20 = lean_ctor_get(x_5, 3);
lean_dec(x_20);
x_21 = lean_ctor_get(x_5, 2);
lean_dec(x_21);
x_22 = lean_ctor_get(x_5, 1);
lean_dec(x_22);
x_23 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_23, 0, x_3);
x_24 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_24, 0, x_12);
lean_closure_set(x_24, 1, x_13);
x_25 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionRing___redArg___lam__1), 4, 2);
lean_closure_set(x_25, 0, x_15);
lean_closure_set(x_25, 1, x_16);
lean_ctor_set(x_5, 7, x_25);
lean_ctor_set(x_5, 6, x_24);
lean_ctor_set(x_5, 5, x_10);
lean_ctor_set(x_5, 4, x_9);
lean_ctor_set(x_5, 3, x_23);
lean_ctor_set(x_5, 2, x_8);
lean_ctor_set(x_5, 1, x_7);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_26 = lean_ctor_get(x_5, 0);
x_27 = lean_ctor_get(x_5, 5);
lean_inc(x_27);
lean_inc(x_26);
lean_dec(x_5);
x_28 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_29, 0, x_12);
lean_closure_set(x_29, 1, x_13);
x_30 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivisionRing___redArg___lam__1), 4, 2);
lean_closure_set(x_30, 0, x_26);
lean_closure_set(x_30, 1, x_27);
x_31 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_31, 0, x_4);
lean_ctor_set(x_31, 1, x_7);
lean_ctor_set(x_31, 2, x_8);
lean_ctor_set(x_31, 3, x_28);
lean_ctor_set(x_31, 4, x_9);
lean_ctor_set(x_31, 5, x_10);
lean_ctor_set(x_31, 6, x_29);
lean_ctor_set(x_31, 7, x_30);
return x_31;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instField(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instField___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_2, x_3);
x_9 = lean_apply_2(x_7, x_8, x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 4);
lean_inc(x_4);
lean_inc_ref(x_2);
x_5 = lp_mathlib_AddOpposite_instSemiring___redArg(x_2);
x_6 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_1);
x_7 = lp_mathlib_AddOpposite_instGroupWithZero___redArg(x_6);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 2);
lean_inc(x_9);
lean_dec_ref(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_3);
lean_inc(x_4);
x_11 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_12, 0, x_4);
x_13 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_13, 0, x_5);
lean_ctor_set(x_13, 1, x_8);
lean_ctor_set(x_13, 2, x_9);
lean_ctor_set(x_13, 3, x_10);
lean_ctor_set(x_13, 4, x_12);
lean_ctor_set(x_13, 5, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instDivisionSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionRing___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_2, x_3);
x_10 = lean_apply_2(x_8, x_9, x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 5);
lean_inc(x_4);
lean_inc_ref(x_2);
x_5 = lp_mathlib_AddOpposite_instRing___redArg(x_2);
x_6 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_1);
x_7 = !lean_is_exclusive(x_1);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_8 = lean_ctor_get(x_1, 7);
lean_dec(x_8);
x_9 = lean_ctor_get(x_1, 6);
lean_dec(x_9);
x_10 = lean_ctor_get(x_1, 5);
lean_dec(x_10);
x_11 = lean_ctor_get(x_1, 4);
lean_dec(x_11);
x_12 = lean_ctor_get(x_1, 3);
lean_dec(x_12);
x_13 = lean_ctor_get(x_1, 2);
lean_dec(x_13);
x_14 = lean_ctor_get(x_1, 1);
lean_dec(x_14);
x_15 = lean_ctor_get(x_1, 0);
lean_dec(x_15);
lean_inc_ref(x_6);
x_16 = lp_mathlib_AddOpposite_instDivisionSemiring___redArg(x_6);
x_17 = lean_ctor_get(x_16, 1);
lean_inc(x_17);
x_18 = lean_ctor_get(x_16, 2);
lean_inc(x_18);
x_19 = lean_ctor_get(x_16, 4);
lean_inc(x_19);
lean_dec_ref(x_16);
x_20 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_6, 4);
lean_inc(x_21);
lean_dec_ref(x_6);
x_22 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_22, 0, x_3);
lean_inc(x_4);
x_23 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionRing___redArg___lam__1), 4, 2);
lean_closure_set(x_23, 0, x_2);
lean_closure_set(x_23, 1, x_4);
x_24 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_24, 0, x_4);
x_25 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_25, 0, x_20);
lean_closure_set(x_25, 1, x_21);
lean_ctor_set(x_1, 7, x_23);
lean_ctor_set(x_1, 6, x_25);
lean_ctor_set(x_1, 5, x_24);
lean_ctor_set(x_1, 4, x_19);
lean_ctor_set(x_1, 3, x_22);
lean_ctor_set(x_1, 2, x_18);
lean_ctor_set(x_1, 1, x_17);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
lean_dec(x_1);
lean_inc_ref(x_6);
x_26 = lp_mathlib_AddOpposite_instDivisionSemiring___redArg(x_6);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 2);
lean_inc(x_28);
x_29 = lean_ctor_get(x_26, 4);
lean_inc(x_29);
lean_dec_ref(x_26);
x_30 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_30);
x_31 = lean_ctor_get(x_6, 4);
lean_inc(x_31);
lean_dec_ref(x_6);
x_32 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_32, 0, x_3);
lean_inc(x_4);
x_33 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionRing___redArg___lam__1), 4, 2);
lean_closure_set(x_33, 0, x_2);
lean_closure_set(x_33, 1, x_4);
x_34 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_34, 0, x_4);
x_35 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_35, 0, x_30);
lean_closure_set(x_35, 1, x_31);
x_36 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_36, 0, x_5);
lean_ctor_set(x_36, 1, x_27);
lean_ctor_set(x_36, 2, x_28);
lean_ctor_set(x_36, 3, x_32);
lean_ctor_set(x_36, 4, x_29);
lean_ctor_set(x_36, 5, x_34);
lean_ctor_set(x_36, 6, x_35);
lean_ctor_set(x_36, 7, x_33);
return x_36;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivisionRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instDivisionRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemifield___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_AddOpposite_instSemiring___redArg(x_2);
x_5 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_1);
lean_inc_ref(x_5);
x_6 = lp_mathlib_AddOpposite_instDivisionSemiring___redArg(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 2);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 4);
lean_inc(x_9);
lean_dec_ref(x_6);
x_10 = !lean_is_exclusive(x_5);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_11 = lean_ctor_get(x_5, 0);
x_12 = lean_ctor_get(x_5, 4);
x_13 = lean_ctor_get(x_5, 5);
lean_dec(x_13);
x_14 = lean_ctor_get(x_5, 3);
lean_dec(x_14);
x_15 = lean_ctor_get(x_5, 2);
lean_dec(x_15);
x_16 = lean_ctor_get(x_5, 1);
lean_dec(x_16);
x_17 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_17, 0, x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_18, 0, x_11);
lean_closure_set(x_18, 1, x_12);
lean_ctor_set(x_5, 5, x_18);
lean_ctor_set(x_5, 4, x_9);
lean_ctor_set(x_5, 3, x_17);
lean_ctor_set(x_5, 2, x_8);
lean_ctor_set(x_5, 1, x_7);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_19 = lean_ctor_get(x_5, 0);
x_20 = lean_ctor_get(x_5, 4);
lean_inc(x_20);
lean_inc(x_19);
lean_dec(x_5);
x_21 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_21, 0, x_3);
x_22 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_22, 0, x_19);
lean_closure_set(x_22, 1, x_20);
x_23 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_23, 0, x_4);
lean_ctor_set(x_23, 1, x_7);
lean_ctor_set(x_23, 2, x_8);
lean_ctor_set(x_23, 3, x_21);
lean_ctor_set(x_23, 4, x_9);
lean_ctor_set(x_23, 5, x_22);
return x_23;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemifield(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instSemifield___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instField___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_AddOpposite_instRing___redArg(x_2);
x_5 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
lean_inc_ref(x_5);
x_6 = lp_mathlib_AddOpposite_instDivisionRing___redArg(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 2);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 4);
lean_inc(x_9);
x_10 = lean_ctor_get(x_6, 5);
lean_inc(x_10);
lean_dec_ref(x_6);
x_11 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_5);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_11, 4);
lean_inc(x_13);
lean_dec_ref(x_11);
x_14 = !lean_is_exclusive(x_5);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_15 = lean_ctor_get(x_5, 0);
x_16 = lean_ctor_get(x_5, 5);
x_17 = lean_ctor_get(x_5, 7);
lean_dec(x_17);
x_18 = lean_ctor_get(x_5, 6);
lean_dec(x_18);
x_19 = lean_ctor_get(x_5, 4);
lean_dec(x_19);
x_20 = lean_ctor_get(x_5, 3);
lean_dec(x_20);
x_21 = lean_ctor_get(x_5, 2);
lean_dec(x_21);
x_22 = lean_ctor_get(x_5, 1);
lean_dec(x_22);
x_23 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_23, 0, x_3);
x_24 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_24, 0, x_12);
lean_closure_set(x_24, 1, x_13);
x_25 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionRing___redArg___lam__1), 4, 2);
lean_closure_set(x_25, 0, x_15);
lean_closure_set(x_25, 1, x_16);
lean_ctor_set(x_5, 7, x_25);
lean_ctor_set(x_5, 6, x_24);
lean_ctor_set(x_5, 5, x_10);
lean_ctor_set(x_5, 4, x_9);
lean_ctor_set(x_5, 3, x_23);
lean_ctor_set(x_5, 2, x_8);
lean_ctor_set(x_5, 1, x_7);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_26 = lean_ctor_get(x_5, 0);
x_27 = lean_ctor_get(x_5, 5);
lean_inc(x_27);
lean_inc(x_26);
lean_dec(x_5);
x_28 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionSemiring___redArg___lam__1), 4, 2);
lean_closure_set(x_29, 0, x_12);
lean_closure_set(x_29, 1, x_13);
x_30 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivisionRing___redArg___lam__1), 4, 2);
lean_closure_set(x_30, 0, x_26);
lean_closure_set(x_30, 1, x_27);
x_31 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_31, 0, x_4);
lean_ctor_set(x_31, 1, x_7);
lean_ctor_set(x_31, 2, x_8);
lean_ctor_set(x_31, 3, x_28);
lean_ctor_set(x_31, 4, x_9);
lean_ctor_set(x_31, 5, x_10);
lean_ctor_set(x_31, 6, x_29);
lean_ctor_set(x_31, 7, x_30);
return x_31;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instField(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instField___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
