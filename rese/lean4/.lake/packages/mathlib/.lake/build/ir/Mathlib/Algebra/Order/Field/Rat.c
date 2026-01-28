// Lean compiler output
// Module: Mathlib.Algebra.Order.Field.Rat
// Imports: public import Init public import Mathlib.Algebra.Field.Rat public import Mathlib.Algebra.Order.Nonneg.Field public import Mathlib.Algebra.Order.Ring.Rat
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
lean_object* lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___redArg(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Rat_linearOrder;
static lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0;
extern lean_object* lp_mathlib_Rat_instField;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat;
static lean_object* _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Rat_linearOrder;
x_2 = lp_mathlib_Rat_instField;
x_3 = lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Rat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Field_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0 = _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0();
lean_mark_persistent(lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0);
lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat = _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat();
lean_mark_persistent(lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
