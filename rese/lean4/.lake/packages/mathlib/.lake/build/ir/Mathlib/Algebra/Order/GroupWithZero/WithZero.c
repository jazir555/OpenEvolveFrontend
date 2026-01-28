// Lean compiler output
// Module: Mathlib.Algebra.Order.GroupWithZero.WithZero
// Imports: public import Init public import Mathlib.Algebra.Order.GroupWithZero.Canonical public import Mathlib.Algebra.Order.GroupWithZero.Unbundled.Basic
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
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withZeroUnits___redArg(lean_object*);
lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withZeroUnits(lean_object*, lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withZeroUnits___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderIso_withZeroUnits___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderIso_withZeroUnits___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_1, 5);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_3, x_2);
x_6 = lean_unbox(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withZeroUnits___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_OrderIso_withZeroUnits___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withZeroUnits___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(x_1);
x_3 = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(x_2);
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_6);
x_7 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_6);
x_8 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_withZeroUnits___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_10, 0, x_5);
lean_closure_set(x_10, 1, x_9);
x_11 = lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(x_3, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withZeroUnits(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderIso_withZeroUnits___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Unbundled_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_WithZero(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
