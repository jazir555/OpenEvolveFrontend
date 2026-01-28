// Lean compiler output
// Module: Mathlib.Algebra.ContinuedFractions.Computation.Approximations
// Imports: public import Init public import Mathlib.Algebra.ContinuedFractions.Determinant public import Mathlib.Algebra.ContinuedFractions.Computation.CorrectnessTerminating public import Mathlib.Algebra.Order.Ring.Basic public import Mathlib.Data.Nat.Fib.Basic public import Mathlib.Tactic.Monotonicity public import Mathlib.Tactic.GCongr
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
LEAN_EXPORT lean_object* lp_mathlib_SimpContFract_of___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContFract_of(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
lean_object* lp_mathlib_GenContFract_of___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SimpContFract_of(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContFract_of___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SimpContFract_of___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Field_toDivisionRing___redArg(x_2);
x_6 = lp_mathlib_GenContFract_of___redArg(x_5, x_3, x_4, x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SimpContFract_of(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_SimpContFract_of___redArg(x_2, x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContFract_of(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_SimpContFract_of___redArg(x_2, x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContFract_of___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SimpContFract_of___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_ContinuedFractions_Determinant(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_ContinuedFractions_Computation_CorrectnessTerminating(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Fib_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Monotonicity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GCongr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_ContinuedFractions_Computation_Approximations(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_ContinuedFractions_Determinant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_ContinuedFractions_Computation_CorrectnessTerminating(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Fib_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Monotonicity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
