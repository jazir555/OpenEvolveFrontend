// Lean compiler output
// Module: Mathlib.Algebra.Ring.Int.Parity
// Imports: public import Init public import Mathlib.Algebra.Ring.Parity public import Mathlib.Algebra.Ring.Int.Defs public import Mathlib.Algebra.Group.Int.Even
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
uint8_t lp_mathlib_Int_instDecidablePredEven(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredOdd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredOdd___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredOdd(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = lp_mathlib_Int_instDecidablePredEven(x_1);
if (x_2 == 0)
{
uint8_t x_3; 
x_3 = 1;
return x_3;
}
else
{
uint8_t x_4; 
x_4 = 0;
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredOdd___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Int_instDecidablePredOdd(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Parity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Even(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Int_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
