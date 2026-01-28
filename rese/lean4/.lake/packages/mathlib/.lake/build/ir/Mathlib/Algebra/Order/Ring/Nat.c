// Lean compiler output
// Module: Mathlib.Algebra.Order.Ring.Nat
// Imports: public import Init public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Algebra.Order.GroupWithZero.Canonical public import Mathlib.Algebra.Order.Ring.Defs public import Mathlib.Algebra.Ring.Parity public import Mathlib.Order.BooleanAlgebra.Set
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
static lean_object* lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero___closed__0;
extern lean_object* lp_mathlib_Nat_instCommMonoidWithZero;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero;
extern lean_object* lp_mathlib_Nat_instLinearOrder;
static lean_object* _init_lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_mathlib_Nat_instLinearOrder;
x_3 = lp_mathlib_Nat_instCommMonoidWithZero;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Parity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BooleanAlgebra_Set(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BooleanAlgebra_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero___closed__0 = _init_lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero___closed__0);
lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero = _init_lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrderedCommMonoidWithZero);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
