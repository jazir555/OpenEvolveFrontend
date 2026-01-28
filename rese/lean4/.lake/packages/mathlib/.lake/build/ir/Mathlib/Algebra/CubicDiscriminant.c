// Lean compiler output
// Module: Mathlib.Algebra.CubicDiscriminant
// Imports: public import Init public import Mathlib.Algebra.Polynomial.Splits public import Mathlib.Tactic.IntervalCases
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_discr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_instInhabited___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_disc(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_disc___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_discr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_instZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cubic_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
lean_inc_n(x_2, 3);
x_3 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_2);
lean_ctor_set(x_3, 2, x_2);
lean_ctor_set(x_3, 3, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_instInhabited___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
lean_inc_n(x_1, 3);
x_2 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
lean_ctor_set(x_2, 2, x_1);
lean_ctor_set(x_2, 3, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_instZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
lean_inc_n(x_2, 3);
x_3 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_2);
lean_ctor_set(x_3, 2, x_2);
lean_ctor_set(x_3, 3, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_instZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
lean_inc_n(x_1, 3);
x_2 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
lean_ctor_set(x_2, 2, x_1);
lean_ctor_set(x_2, 3, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_map___redArg(lean_object* x_1, lean_object* x_2) {
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
x_7 = lean_ctor_get(x_2, 3);
lean_inc(x_1);
x_8 = lean_apply_1(x_1, x_4);
lean_inc(x_1);
x_9 = lean_apply_1(x_1, x_5);
lean_inc(x_1);
x_10 = lean_apply_1(x_1, x_6);
x_11 = lean_apply_1(x_1, x_7);
lean_ctor_set(x_2, 3, x_11);
lean_ctor_set(x_2, 2, x_10);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_8);
return x_2;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
x_14 = lean_ctor_get(x_2, 2);
x_15 = lean_ctor_get(x_2, 3);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
lean_inc(x_1);
x_16 = lean_apply_1(x_1, x_12);
lean_inc(x_1);
x_17 = lean_apply_1(x_1, x_13);
lean_inc(x_1);
x_18 = lean_apply_1(x_1, x_14);
x_19 = lean_apply_1(x_1, x_15);
x_20 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_20, 0, x_16);
lean_ctor_set(x_20, 1, x_17);
lean_ctor_set(x_20, 2, x_18);
lean_ctor_set(x_20, 3, x_19);
return x_20;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Cubic_map___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Cubic_map(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_discr___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
lean_inc_ref(x_1);
x_9 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_10 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_9);
x_11 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_11);
lean_dec_ref(x_1);
x_12 = lean_ctor_get(x_9, 1);
lean_inc_ref(x_12);
lean_dec_ref(x_9);
x_13 = lean_ctor_get(x_10, 2);
lean_inc(x_13);
lean_dec_ref(x_10);
x_14 = lean_ctor_get(x_2, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_2, 1);
lean_inc(x_15);
x_16 = lean_ctor_get(x_2, 2);
lean_inc(x_16);
x_17 = lean_ctor_get(x_2, 3);
lean_inc(x_17);
lean_dec_ref(x_2);
x_18 = lean_ctor_get(x_11, 3);
lean_inc(x_18);
lean_dec_ref(x_11);
x_19 = lean_ctor_get(x_12, 0);
lean_inc(x_19);
lean_dec_ref(x_12);
x_20 = lean_unsigned_to_nat(2u);
lean_inc(x_18);
lean_inc(x_15);
x_21 = lean_apply_2(x_18, x_20, x_15);
lean_inc(x_18);
lean_inc(x_16);
x_22 = lean_apply_2(x_18, x_20, x_16);
lean_inc(x_7);
x_23 = lean_apply_2(x_7, x_21, x_22);
x_24 = lean_unsigned_to_nat(4u);
lean_inc(x_19);
x_25 = lean_apply_1(x_19, x_24);
lean_inc(x_7);
lean_inc(x_14);
lean_inc(x_25);
x_26 = lean_apply_2(x_7, x_25, x_14);
x_27 = lean_unsigned_to_nat(3u);
lean_inc(x_18);
lean_inc(x_16);
x_28 = lean_apply_2(x_18, x_27, x_16);
lean_inc(x_7);
x_29 = lean_apply_2(x_7, x_26, x_28);
lean_inc(x_13);
x_30 = lean_apply_2(x_13, x_23, x_29);
lean_inc(x_18);
lean_inc(x_15);
x_31 = lean_apply_2(x_18, x_27, x_15);
lean_inc(x_7);
x_32 = lean_apply_2(x_7, x_25, x_31);
lean_inc(x_7);
lean_inc(x_17);
x_33 = lean_apply_2(x_7, x_32, x_17);
lean_inc(x_13);
x_34 = lean_apply_2(x_13, x_30, x_33);
x_35 = lean_unsigned_to_nat(27u);
lean_inc(x_19);
x_36 = lean_apply_1(x_19, x_35);
lean_inc(x_18);
lean_inc(x_14);
x_37 = lean_apply_2(x_18, x_20, x_14);
lean_inc(x_7);
x_38 = lean_apply_2(x_7, x_36, x_37);
lean_inc(x_17);
x_39 = lean_apply_2(x_18, x_20, x_17);
lean_inc(x_7);
x_40 = lean_apply_2(x_7, x_38, x_39);
x_41 = lean_apply_2(x_13, x_34, x_40);
x_42 = lean_unsigned_to_nat(18u);
x_43 = lean_apply_1(x_19, x_42);
lean_inc(x_7);
x_44 = lean_apply_2(x_7, x_43, x_14);
lean_inc(x_7);
x_45 = lean_apply_2(x_7, x_44, x_15);
lean_inc(x_7);
x_46 = lean_apply_2(x_7, x_45, x_16);
x_47 = lean_apply_2(x_7, x_46, x_17);
x_48 = lean_apply_2(x_8, x_41, x_47);
return x_48;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_discr(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Cubic_discr___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_disc(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Cubic_discr___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cubic_disc___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Cubic_discr___redArg(x_1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Splits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_IntervalCases(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_CubicDiscriminant(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Splits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_IntervalCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
