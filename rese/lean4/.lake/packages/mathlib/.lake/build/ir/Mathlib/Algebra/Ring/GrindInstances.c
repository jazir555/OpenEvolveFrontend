// Lean compiler output
// Module: Mathlib.Algebra.Ring.GrindInstances
// Imports: public import Init public import Mathlib.Algebra.Ring.Defs public import Mathlib.Data.Int.Cast.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toGrindCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toGrindCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toGrindRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toGrindCommRing(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toGrindCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toGrindRing___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_eq(x_4, x_5);
if (x_6 == 1)
{
lean_dec(x_3);
lean_inc(x_1);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_4, x_7);
x_9 = lean_nat_dec_eq(x_8, x_5);
if (x_9 == 1)
{
lean_dec(x_8);
lean_dec(x_3);
lean_inc(x_2);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_nat_sub(x_8, x_7);
lean_dec(x_8);
x_11 = lean_unsigned_to_nat(2u);
x_12 = lean_nat_add(x_10, x_11);
lean_dec(x_10);
x_13 = lean_apply_1(x_3, x_12);
return x_13;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_2);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_1, 3);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 2);
lean_inc(x_7);
x_8 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
lean_dec_ref(x_1);
lean_inc_ref(x_8);
x_9 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_9, 2);
lean_inc(x_11);
lean_dec_ref(x_9);
x_12 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_8);
x_13 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_12);
x_14 = lean_ctor_get(x_13, 1);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_15, 0, x_4);
lean_inc(x_10);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1___boxed), 4, 3);
lean_closure_set(x_16, 0, x_14);
lean_closure_set(x_16, 1, x_11);
lean_closure_set(x_16, 2, x_10);
x_17 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_17, 0, x_6);
lean_ctor_set(x_17, 1, x_5);
lean_ctor_set(x_17, 2, x_10);
lean_ctor_set(x_17, 3, x_16);
lean_ctor_set(x_17, 4, x_7);
lean_ctor_set(x_17, 5, x_15);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Semiring_toGrindSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toGrindCommSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Semiring_toGrindSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toGrindCommSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Semiring_toGrindSemiring___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toGrindRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_3, 0);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 2);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 3);
lean_inc(x_7);
x_8 = lean_ctor_get(x_2, 3);
lean_inc(x_8);
x_9 = lean_ctor_get(x_3, 1);
lean_inc(x_9);
x_10 = lean_ctor_get(x_4, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_4, 2);
lean_inc(x_11);
x_12 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_14 = lean_ctor_get(x_12, 1);
x_15 = lean_ctor_get(x_12, 0);
x_16 = lean_ctor_get(x_12, 4);
lean_dec(x_16);
x_17 = lean_ctor_get(x_12, 3);
lean_dec(x_17);
x_18 = lean_ctor_get(x_12, 2);
lean_dec(x_18);
x_19 = lean_ctor_get(x_14, 0);
lean_inc(x_19);
lean_dec_ref(x_14);
x_20 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
lean_dec_ref(x_2);
lean_inc_ref(x_20);
x_21 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
x_23 = lean_ctor_get(x_21, 2);
lean_inc(x_23);
lean_dec_ref(x_21);
x_24 = lean_ctor_get(x_20, 0);
lean_inc_ref(x_24);
lean_dec_ref(x_20);
x_25 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_24);
x_26 = lean_ctor_get(x_25, 1);
lean_inc(x_26);
lean_dec_ref(x_25);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_27, 0, x_8);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1___boxed), 4, 3);
lean_closure_set(x_28, 0, x_26);
lean_closure_set(x_28, 1, x_23);
lean_closure_set(x_28, 2, x_22);
x_29 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_29, 0, x_10);
lean_ctor_set(x_29, 1, x_9);
lean_ctor_set(x_29, 2, x_19);
lean_ctor_set(x_29, 3, x_28);
lean_ctor_set(x_29, 4, x_11);
lean_ctor_set(x_29, 5, x_27);
lean_ctor_set(x_12, 4, x_7);
lean_ctor_set(x_12, 3, x_15);
lean_ctor_set(x_12, 2, x_6);
lean_ctor_set(x_12, 1, x_5);
lean_ctor_set(x_12, 0, x_29);
return x_12;
}
else
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; 
x_30 = lean_ctor_get(x_12, 1);
x_31 = lean_ctor_get(x_12, 0);
lean_inc(x_30);
lean_inc(x_31);
lean_dec(x_12);
x_32 = lean_ctor_get(x_30, 0);
lean_inc(x_32);
lean_dec_ref(x_30);
x_33 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
lean_dec_ref(x_2);
lean_inc_ref(x_33);
x_34 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_33);
x_35 = lean_ctor_get(x_34, 0);
lean_inc(x_35);
x_36 = lean_ctor_get(x_34, 2);
lean_inc(x_36);
lean_dec_ref(x_34);
x_37 = lean_ctor_get(x_33, 0);
lean_inc_ref(x_37);
lean_dec_ref(x_33);
x_38 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_37);
x_39 = lean_ctor_get(x_38, 1);
lean_inc(x_39);
lean_dec_ref(x_38);
x_40 = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_40, 0, x_8);
x_41 = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1___boxed), 4, 3);
lean_closure_set(x_41, 0, x_39);
lean_closure_set(x_41, 1, x_36);
lean_closure_set(x_41, 2, x_35);
x_42 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_42, 0, x_10);
lean_ctor_set(x_42, 1, x_9);
lean_ctor_set(x_42, 2, x_32);
lean_ctor_set(x_42, 3, x_41);
lean_ctor_set(x_42, 4, x_11);
lean_ctor_set(x_42, 5, x_40);
x_43 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_43, 0, x_42);
lean_ctor_set(x_43, 1, x_5);
lean_ctor_set(x_43, 2, x_6);
lean_ctor_set(x_43, 3, x_31);
lean_ctor_set(x_43, 4, x_7);
return x_43;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toGrindRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Ring_toGrindRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toGrindCommRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Ring_toGrindRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toGrindCommRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Ring_toGrindRing___redArg(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
