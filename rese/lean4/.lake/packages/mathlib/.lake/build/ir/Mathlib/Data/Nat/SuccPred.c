// Lean compiler output
// Module: Mathlib.Data.Nat.SuccPred
// Imports: public import Init public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Algebra.Ring.Nat public import Mathlib.Algebra.Order.Monoid.Unbundled.WithTop public import Mathlib.Algebra.Order.Sub.Unbundled.Basic public import Mathlib.Algebra.Order.SuccPred public import Mathlib.Data.Fin.Basic public import Mathlib.Order.Nat public import Mathlib.Order.SuccPred.Archimedean public import Mathlib.Order.SuccPred.WithBot
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_instPredOrder;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccOrder___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instPredSubOrder;
static lean_object* lp_mathlib_Nat_instSuccAddOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccOrder___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccOrder;
static lean_object* lp_mathlib_Nat_instPredOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccAddOrder;
lean_object* l_Nat_pred___boxed(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccOrder___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_nat_add(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccOrder___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nat_instSuccOrder___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_instSuccOrder() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_instSuccOrder___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instSuccAddOrder___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_instSuccOrder___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instSuccAddOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instSuccAddOrder___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instPredOrder___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_pred___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instPredOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instPredOrder___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instPredSubOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instPredOrder___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Unbundled_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_WithBot(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instSuccOrder = _init_lp_mathlib_Nat_instSuccOrder();
lean_mark_persistent(lp_mathlib_Nat_instSuccOrder);
lp_mathlib_Nat_instSuccAddOrder___closed__0 = _init_lp_mathlib_Nat_instSuccAddOrder___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instSuccAddOrder___closed__0);
lp_mathlib_Nat_instSuccAddOrder = _init_lp_mathlib_Nat_instSuccAddOrder();
lean_mark_persistent(lp_mathlib_Nat_instSuccAddOrder);
lp_mathlib_Nat_instPredOrder___closed__0 = _init_lp_mathlib_Nat_instPredOrder___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instPredOrder___closed__0);
lp_mathlib_Nat_instPredOrder = _init_lp_mathlib_Nat_instPredOrder();
lean_mark_persistent(lp_mathlib_Nat_instPredOrder);
lp_mathlib_Nat_instPredSubOrder = _init_lp_mathlib_Nat_instPredSubOrder();
lean_mark_persistent(lp_mathlib_Nat_instPredSubOrder);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
