// Lean compiler output
// Module: Mathlib.Data.Matrix.Diagonal
// Imports: public import Init public import Mathlib.Data.Int.Cast.Basic public import Mathlib.Data.Int.Cast.Pi public import Mathlib.Data.Nat.Cast.Basic public import Mathlib.LinearAlgebra.Matrix.Defs public import Mathlib.Logic.Embedding.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diag(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddMonoidWithOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommGroupWithOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diag___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Matrix_diagonal___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMonoidWithOne___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Matrix_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Matrix_diagonal___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc(x_4);
x_6 = lean_apply_2(x_1, x_4, x_5);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_dec(x_4);
lean_dec(x_3);
lean_inc(x_2);
return x_2;
}
else
{
lean_object* x_8; 
x_8 = lean_apply_1(x_3, x_4);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_diagonal___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lp_mathlib_Matrix_diagonal___redArg___closed__0;
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_3);
x_9 = lean_apply_3(x_7, x_8, x_4, x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_diagonal___redArg(x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_4);
x_8 = lp_mathlib_Matrix_diagonal___redArg(x_2, x_3, x_7, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(x_4, 0, x_3);
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_instNatCastOfZero___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_4);
x_8 = lp_mathlib_Matrix_diagonal___redArg(x_2, x_3, x_7, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(x_4, 0, x_3);
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_instIntCastOfZero___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Matrix_one___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Matrix_one___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal), 7, 5);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_1);
lean_closure_set(x_5, 3, x_2);
lean_closure_set(x_5, 4, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_one___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddMonoidWithOne___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
x_7 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_5);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
lean_inc(x_8);
lean_inc_ref(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(x_9, 0, x_4);
lean_closure_set(x_9, 1, x_1);
lean_closure_set(x_9, 2, x_8);
x_10 = lp_mathlib_Matrix_addMonoid___redArg(x_5);
x_11 = lp_mathlib_Matrix_one___redArg(x_1, x_8, x_6);
lean_ctor_set(x_2, 2, x_11);
lean_ctor_set(x_2, 1, x_10);
lean_ctor_set(x_2, 0, x_9);
return x_2;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
x_14 = lean_ctor_get(x_2, 2);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
x_15 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_13);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc(x_16);
lean_inc_ref(x_1);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(x_17, 0, x_12);
lean_closure_set(x_17, 1, x_1);
lean_closure_set(x_17, 2, x_16);
x_18 = lp_mathlib_Matrix_addMonoid___redArg(x_13);
x_19 = lp_mathlib_Matrix_one___redArg(x_1, x_16, x_14);
x_20 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_20, 0, x_17);
lean_ctor_set(x_20, 1, x_18);
lean_ctor_set(x_20, 2, x_19);
return x_20;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddMonoidWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Matrix_instAddMonoidWithOne___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_apply_2(x_3, x_4, x_5);
x_7 = lean_apply_2(x_1, x_2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_2);
lean_inc_ref(x_3);
x_4 = lp_mathlib_Matrix_addGroup___redArg(x_3);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_2, 4);
x_9 = lean_ctor_get(x_2, 3);
lean_dec(x_9);
x_10 = lean_ctor_get(x_2, 2);
lean_dec(x_10);
lean_inc_ref(x_1);
x_11 = lp_mathlib_Matrix_instAddMonoidWithOne___redArg(x_1, x_7);
x_12 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_3);
lean_dec_ref(x_3);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = !lean_is_exclusive(x_11);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_15 = lean_ctor_get(x_11, 1);
lean_dec(x_15);
x_16 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_16);
x_17 = lean_ctor_get(x_4, 1);
lean_inc(x_17);
x_18 = lean_ctor_get(x_4, 2);
lean_inc(x_18);
lean_dec_ref(x_4);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instAddGroupWithOne___redArg___lam__0), 5, 1);
lean_closure_set(x_19, 0, x_8);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(x_20, 0, x_6);
lean_closure_set(x_20, 1, x_1);
lean_closure_set(x_20, 2, x_13);
lean_ctor_set(x_11, 1, x_16);
lean_ctor_set(x_2, 4, x_19);
lean_ctor_set(x_2, 3, x_18);
lean_ctor_set(x_2, 2, x_17);
lean_ctor_set(x_2, 1, x_11);
lean_ctor_set(x_2, 0, x_20);
return x_2;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_21 = lean_ctor_get(x_11, 0);
x_22 = lean_ctor_get(x_11, 2);
lean_inc(x_22);
lean_inc(x_21);
lean_dec(x_11);
x_23 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_4, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_4, 2);
lean_inc(x_25);
lean_dec_ref(x_4);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instAddGroupWithOne___redArg___lam__0), 5, 1);
lean_closure_set(x_26, 0, x_8);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(x_27, 0, x_6);
lean_closure_set(x_27, 1, x_1);
lean_closure_set(x_27, 2, x_13);
x_28 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_28, 0, x_21);
lean_ctor_set(x_28, 1, x_23);
lean_ctor_set(x_28, 2, x_22);
lean_ctor_set(x_2, 4, x_26);
lean_ctor_set(x_2, 3, x_25);
lean_ctor_set(x_2, 2, x_24);
lean_ctor_set(x_2, 1, x_28);
lean_ctor_set(x_2, 0, x_27);
return x_2;
}
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_29 = lean_ctor_get(x_2, 0);
x_30 = lean_ctor_get(x_2, 1);
x_31 = lean_ctor_get(x_2, 4);
lean_inc(x_31);
lean_inc(x_30);
lean_inc(x_29);
lean_dec(x_2);
lean_inc_ref(x_1);
x_32 = lp_mathlib_Matrix_instAddMonoidWithOne___redArg(x_1, x_30);
x_33 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_3);
lean_dec_ref(x_3);
x_34 = lean_ctor_get(x_33, 0);
lean_inc(x_34);
lean_dec_ref(x_33);
x_35 = lean_ctor_get(x_32, 0);
lean_inc(x_35);
x_36 = lean_ctor_get(x_32, 2);
lean_inc(x_36);
if (lean_is_exclusive(x_32)) {
 lean_ctor_release(x_32, 0);
 lean_ctor_release(x_32, 1);
 lean_ctor_release(x_32, 2);
 x_37 = x_32;
} else {
 lean_dec_ref(x_32);
 x_37 = lean_box(0);
}
x_38 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_38);
x_39 = lean_ctor_get(x_4, 1);
lean_inc(x_39);
x_40 = lean_ctor_get(x_4, 2);
lean_inc(x_40);
lean_dec_ref(x_4);
x_41 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instAddGroupWithOne___redArg___lam__0), 5, 1);
lean_closure_set(x_41, 0, x_31);
x_42 = lean_alloc_closure((void*)(lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(x_42, 0, x_29);
lean_closure_set(x_42, 1, x_1);
lean_closure_set(x_42, 2, x_34);
if (lean_is_scalar(x_37)) {
 x_43 = lean_alloc_ctor(0, 3, 0);
} else {
 x_43 = x_37;
}
lean_ctor_set(x_43, 0, x_35);
lean_ctor_set(x_43, 1, x_38);
lean_ctor_set(x_43, 2, x_36);
x_44 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_44, 0, x_42);
lean_ctor_set(x_44, 1, x_43);
lean_ctor_set(x_44, 2, x_39);
lean_ctor_set(x_44, 3, x_40);
lean_ctor_set(x_44, 4, x_41);
return x_44;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Matrix_instAddGroupWithOne___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMonoidWithOne___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lp_mathlib_Matrix_addCommMonoid___redArg(x_3);
x_5 = lp_mathlib_Matrix_instAddMonoidWithOne___redArg(x_1, x_2);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 1);
lean_dec(x_7);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_5, 0);
x_9 = lean_ctor_get(x_5, 2);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_5);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_4);
lean_ctor_set(x_10, 2, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMonoidWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Matrix_instAddCommMonoidWithOne___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommGroupWithOne___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_Matrix_addCommGroup___redArg(x_3);
x_5 = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(x_2);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_7 = lean_ctor_get(x_2, 3);
lean_dec(x_7);
x_8 = lean_ctor_get(x_2, 2);
lean_dec(x_8);
x_9 = lean_ctor_get(x_2, 1);
lean_dec(x_9);
x_10 = lean_ctor_get(x_2, 0);
lean_dec(x_10);
x_11 = lp_mathlib_Matrix_instAddGroupWithOne___redArg(x_1, x_5);
x_12 = lean_ctor_get(x_11, 1);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_11, 0);
lean_inc(x_13);
lean_dec_ref(x_11);
x_14 = lean_ctor_get(x_12, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_12, 2);
lean_inc(x_15);
lean_dec_ref(x_12);
lean_ctor_set(x_2, 3, x_15);
lean_ctor_set(x_2, 2, x_14);
lean_ctor_set(x_2, 1, x_13);
lean_ctor_set(x_2, 0, x_4);
return x_2;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
lean_dec(x_2);
x_16 = lp_mathlib_Matrix_instAddGroupWithOne___redArg(x_1, x_5);
x_17 = lean_ctor_get(x_16, 1);
lean_inc_ref(x_17);
x_18 = lean_ctor_get(x_16, 0);
lean_inc(x_18);
lean_dec_ref(x_16);
x_19 = lean_ctor_get(x_17, 0);
lean_inc(x_19);
x_20 = lean_ctor_get(x_17, 2);
lean_inc(x_20);
lean_dec_ref(x_17);
x_21 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_21, 0, x_4);
lean_ctor_set(x_21, 1, x_18);
lean_ctor_set(x_21, 2, x_19);
lean_ctor_set(x_21, 3, x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommGroupWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Matrix_instAddCommGroupWithOne___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diag(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
lean_inc(x_4);
x_5 = lean_apply_2(x_3, x_4, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diag___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
lean_inc(x_2);
x_3 = lean_apply_2(x_1, x_2, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Matrix_Diagonal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Matrix_diagonal___redArg___closed__0 = _init_lp_mathlib_Matrix_diagonal___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Matrix_diagonal___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
