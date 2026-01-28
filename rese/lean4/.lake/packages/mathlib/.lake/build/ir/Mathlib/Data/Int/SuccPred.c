// Lean compiler output
// Module: Mathlib.Data.Int.SuccPred
// Imports: public import Init public import Mathlib.Algebra.Order.Ring.Int public import Mathlib.Data.Nat.SuccPred
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
lean_object* lp_mathlib_Int_succ___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instSuccAddOrder;
LEAN_EXPORT lean_object* lp_mathlib_Int_instPredOrder;
static lean_object* lp_mathlib_Int_instPredOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_instSuccOrder;
static lean_object* lp_mathlib_Int_instSuccOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_instPredSubOrder;
lean_object* lp_mathlib_Int_pred___boxed(lean_object*);
static lean_object* _init_lp_mathlib_Int_instSuccOrder___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Int_succ___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instSuccOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instSuccOrder___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instSuccAddOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instSuccOrder___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instPredOrder___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Int_pred___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instPredOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instPredOrder___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instPredSubOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instPredOrder___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_SuccPred(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_instSuccOrder___closed__0 = _init_lp_mathlib_Int_instSuccOrder___closed__0();
lean_mark_persistent(lp_mathlib_Int_instSuccOrder___closed__0);
lp_mathlib_Int_instSuccOrder = _init_lp_mathlib_Int_instSuccOrder();
lean_mark_persistent(lp_mathlib_Int_instSuccOrder);
lp_mathlib_Int_instSuccAddOrder = _init_lp_mathlib_Int_instSuccAddOrder();
lean_mark_persistent(lp_mathlib_Int_instSuccAddOrder);
lp_mathlib_Int_instPredOrder___closed__0 = _init_lp_mathlib_Int_instPredOrder___closed__0();
lean_mark_persistent(lp_mathlib_Int_instPredOrder___closed__0);
lp_mathlib_Int_instPredOrder = _init_lp_mathlib_Int_instPredOrder();
lean_mark_persistent(lp_mathlib_Int_instPredOrder);
lp_mathlib_Int_instPredSubOrder = _init_lp_mathlib_Int_instPredSubOrder();
lean_mark_persistent(lp_mathlib_Int_instPredSubOrder);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
