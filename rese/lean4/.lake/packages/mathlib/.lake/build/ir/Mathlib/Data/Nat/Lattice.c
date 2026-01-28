// Lean compiler output
// Module: Mathlib.Data.Nat.Lattice
// Imports: public import Init public import Mathlib.Order.ConditionallyCompleteLattice.Finset public import Mathlib.Order.Interval.Finset.Nat
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLattice;
static lean_object* lp_mathlib_Nat_instLattice___closed__0;
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
extern lean_object* lp_mathlib_Nat_instLinearOrder;
static lean_object* _init_lp_mathlib_Nat_instLattice___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instLinearOrder;
x_2 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_instLattice() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instLattice___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instLattice___closed__0 = _init_lp_mathlib_Nat_instLattice___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instLattice___closed__0);
lp_mathlib_Nat_instLattice = _init_lp_mathlib_Nat_instLattice();
lean_mark_persistent(lp_mathlib_Nat_instLattice);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
