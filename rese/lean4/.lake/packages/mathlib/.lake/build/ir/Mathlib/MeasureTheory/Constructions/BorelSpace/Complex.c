// Lean compiler output
// Module: Mathlib.MeasureTheory.Constructions.BorelSpace.Complex
// Imports: public import Init public import Mathlib.Analysis.Complex.Basic public import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_RCLike_measurableSpace___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RCLike_measurableSpace(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Complex_measurableSpace;
LEAN_EXPORT lean_object* lp_mathlib_RCLike_measurableSpace(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RCLike_measurableSpace___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RCLike_measurableSpace(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Complex_measurableSpace() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Complex_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Constructions_BorelSpace_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Constructions_BorelSpace_Complex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Complex_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Constructions_BorelSpace_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Complex_measurableSpace = _init_lp_mathlib_Complex_measurableSpace();
lean_mark_persistent(lp_mathlib_Complex_measurableSpace);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
