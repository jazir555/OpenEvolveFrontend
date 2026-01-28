// Lean compiler output
// Module: Mathlib.Algebra.Field.ULift
// Imports: public import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.GroupWithZero.ULift public import Mathlib.Algebra.Ring.ULift
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
lean_object* lp_mathlib_ULift_ring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_field(lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionRing___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNNRatCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_field___redArg(lean_object*);
lean_object* lp_mathlib_ULift_groupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semifield___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNNRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNNRatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semifield(lean_object*, lean_object*);
lean_object* lp_mathlib_ULift_semiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instRatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semifield___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNNRatCast___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNNRatCast___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNNRatCast(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_instNNRatCast___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instRatCast___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instRatCast(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_instRatCast___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionSemiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionSemiring___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 4);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 5);
lean_inc(x_5);
lean_inc_ref(x_2);
x_6 = lp_mathlib_ULift_semiring___redArg(x_2);
x_7 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_1);
x_8 = lp_mathlib_ULift_groupWithZero___redArg(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
x_10 = lean_ctor_get(x_8, 2);
lean_inc(x_10);
lean_dec_ref(x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(x_12, 0, x_5);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ULift_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_13, 0, x_4);
x_14 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_14, 0, x_6);
lean_ctor_set(x_14, 1, x_9);
lean_ctor_set(x_14, 2, x_10);
lean_ctor_set(x_14, 3, x_11);
lean_ctor_set(x_14, 4, x_13);
lean_ctor_set(x_14, 5, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_divisionSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semifield___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 3);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semifield___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 5);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semifield___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_1);
x_3 = lp_mathlib_ULift_divisionSemiring___redArg(x_2);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 5);
lean_dec(x_5);
x_6 = lean_ctor_get(x_3, 3);
lean_dec(x_6);
lean_inc_ref(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ULift_semifield___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_ULift_semifield___redArg___lam__1), 3, 1);
lean_closure_set(x_8, 0, x_1);
lean_ctor_set(x_3, 5, x_8);
lean_ctor_set(x_3, 3, x_7);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = lean_ctor_get(x_3, 0);
x_10 = lean_ctor_get(x_3, 1);
x_11 = lean_ctor_get(x_3, 2);
x_12 = lean_ctor_get(x_3, 4);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_3);
lean_inc_ref(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ULift_semifield___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_1);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ULift_semifield___redArg___lam__1), 3, 1);
lean_closure_set(x_14, 0, x_1);
x_15 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_15, 0, x_9);
lean_ctor_set(x_15, 1, x_10);
lean_ctor_set(x_15, 2, x_11);
lean_ctor_set(x_15, 3, x_13);
lean_ctor_set(x_15, 4, x_12);
lean_ctor_set(x_15, 5, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semifield(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_semifield___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionRing___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_1);
x_3 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_2);
x_4 = lp_mathlib_ULift_groupWithZero___redArg(x_3);
x_5 = !lean_is_exclusive(x_1);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 3);
x_8 = lean_ctor_get(x_1, 4);
x_9 = lean_ctor_get(x_1, 5);
x_10 = lean_ctor_get(x_1, 6);
x_11 = lean_ctor_get(x_1, 7);
x_12 = lean_ctor_get(x_1, 2);
lean_dec(x_12);
x_13 = lean_ctor_get(x_1, 1);
lean_dec(x_13);
x_14 = lp_mathlib_ULift_ring___redArg(x_6);
x_15 = lean_ctor_get(x_4, 1);
lean_inc(x_15);
x_16 = lean_ctor_get(x_4, 2);
lean_inc(x_16);
lean_dec_ref(x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_17, 0, x_7);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(x_18, 0, x_10);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionRing___redArg___lam__2), 3, 1);
lean_closure_set(x_19, 0, x_11);
x_20 = lean_alloc_closure((void*)(lp_mathlib_ULift_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_20, 0, x_8);
x_21 = lean_alloc_closure((void*)(lp_mathlib_ULift_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_21, 0, x_9);
lean_ctor_set(x_1, 7, x_19);
lean_ctor_set(x_1, 6, x_18);
lean_ctor_set(x_1, 5, x_21);
lean_ctor_set(x_1, 4, x_20);
lean_ctor_set(x_1, 3, x_17);
lean_ctor_set(x_1, 2, x_16);
lean_ctor_set(x_1, 1, x_15);
lean_ctor_set(x_1, 0, x_14);
return x_1;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_22 = lean_ctor_get(x_1, 0);
x_23 = lean_ctor_get(x_1, 3);
x_24 = lean_ctor_get(x_1, 4);
x_25 = lean_ctor_get(x_1, 5);
x_26 = lean_ctor_get(x_1, 6);
x_27 = lean_ctor_get(x_1, 7);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_1);
x_28 = lp_mathlib_ULift_ring___redArg(x_22);
x_29 = lean_ctor_get(x_4, 1);
lean_inc(x_29);
x_30 = lean_ctor_get(x_4, 2);
lean_inc(x_30);
lean_dec_ref(x_4);
x_31 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_31, 0, x_23);
x_32 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(x_32, 0, x_26);
x_33 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionRing___redArg___lam__2), 3, 1);
lean_closure_set(x_33, 0, x_27);
x_34 = lean_alloc_closure((void*)(lp_mathlib_ULift_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_34, 0, x_24);
x_35 = lean_alloc_closure((void*)(lp_mathlib_ULift_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(x_35, 0, x_25);
x_36 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_36, 0, x_28);
lean_ctor_set(x_36, 1, x_29);
lean_ctor_set(x_36, 2, x_30);
lean_ctor_set(x_36, 3, x_31);
lean_ctor_set(x_36, 4, x_34);
lean_ctor_set(x_36, 5, x_35);
lean_ctor_set(x_36, 6, x_32);
lean_ctor_set(x_36, 7, x_33);
return x_36;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divisionRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_divisionRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_field___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 6);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 7);
lean_inc(x_5);
lean_inc_ref(x_2);
x_6 = lp_mathlib_ULift_ring___redArg(x_2);
x_7 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
x_8 = lp_mathlib_ULift_divisionRing___redArg(x_7);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_10 = lean_ctor_get(x_8, 7);
lean_dec(x_10);
x_11 = lean_ctor_get(x_8, 6);
lean_dec(x_11);
x_12 = lean_ctor_get(x_8, 3);
lean_dec(x_12);
x_13 = lean_ctor_get(x_8, 0);
lean_dec(x_13);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(x_15, 0, x_4);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionRing___redArg___lam__2), 3, 1);
lean_closure_set(x_16, 0, x_5);
lean_ctor_set(x_8, 7, x_16);
lean_ctor_set(x_8, 6, x_15);
lean_ctor_set(x_8, 3, x_14);
lean_ctor_set(x_8, 0, x_6);
return x_8;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_17 = lean_ctor_get(x_8, 1);
x_18 = lean_ctor_get(x_8, 2);
x_19 = lean_ctor_get(x_8, 4);
x_20 = lean_ctor_get(x_8, 5);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_dec(x_8);
x_21 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(x_21, 0, x_3);
x_22 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(x_22, 0, x_4);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ULift_divisionRing___redArg___lam__2), 3, 1);
lean_closure_set(x_23, 0, x_5);
x_24 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_24, 0, x_6);
lean_ctor_set(x_24, 1, x_17);
lean_ctor_set(x_24, 2, x_18);
lean_ctor_set(x_24, 3, x_21);
lean_ctor_set(x_24, 4, x_19);
lean_ctor_set(x_24, 5, x_20);
lean_ctor_set(x_24, 6, x_22);
lean_ctor_set(x_24, 7, x_23);
return x_24;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_field(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_field___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_ULift(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_ULift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
