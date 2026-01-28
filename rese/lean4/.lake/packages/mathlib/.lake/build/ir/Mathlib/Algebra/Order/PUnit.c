// Lean compiler output
// Module: Mathlib.Algebra.Order.PUnit
// Imports: public import Init public import Mathlib.Algebra.Group.PUnit public import Mathlib.Algebra.Order.AddGroupWithTop
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
extern lean_object* lp_mathlib_PUnit_addCommGroup;
extern lean_object* lp_mathlib_PUnit_instLinearOrder;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instLinearOrderedAddCommMonoidWithTop;
static lean_object* _init_lp_mathlib_PUnit_instLinearOrderedAddCommMonoidWithTop() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_PUnit_addCommGroup;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_PUnit_instLinearOrder;
x_4 = lean_box(0);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_PUnit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_PUnit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PUnit_instLinearOrderedAddCommMonoidWithTop = _init_lp_mathlib_PUnit_instLinearOrderedAddCommMonoidWithTop();
lean_mark_persistent(lp_mathlib_PUnit_instLinearOrderedAddCommMonoidWithTop);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
