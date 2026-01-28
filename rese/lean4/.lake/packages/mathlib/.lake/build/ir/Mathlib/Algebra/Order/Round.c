// Lean compiler output
// Module: Mathlib.Algebra.Order.Round
// Imports: public import Init public import Mathlib.Algebra.Order.Floor.Ring import Mathlib.Algebra.Order.Interval.Set.Group
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
lean_object* lp_mathlib_Int_floor___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Int_fract___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_round___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_round(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Int_ceil___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_round___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; 
x_5 = lean_ctor_get(x_2, 6);
lean_inc_ref(x_5);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
x_6 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_7);
x_9 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
lean_inc_ref(x_1);
x_11 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_12 = lean_ctor_get(x_11, 1);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_12, 2);
lean_inc(x_14);
lean_dec_ref(x_12);
x_15 = lean_unsigned_to_nat(2u);
x_16 = lean_apply_1(x_13, x_15);
lean_inc(x_4);
lean_inc_ref(x_3);
x_17 = lp_mathlib_Int_fract___redArg(x_1, x_3, x_4);
x_18 = lean_apply_2(x_10, x_16, x_17);
x_19 = lean_apply_2(x_5, x_18, x_14);
x_20 = lean_unbox(x_19);
if (x_20 == 0)
{
lean_object* x_21; 
x_21 = lp_mathlib_Int_ceil___redArg(x_3, x_4);
return x_21;
}
else
{
lean_object* x_22; 
x_22 = lp_mathlib_Int_floor___redArg(x_3, x_4);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_round(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_round___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Floor_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Interval_Set_Group(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Round(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Floor_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Interval_Set_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
