// Lean compiler output
// Module: Mathlib.Analysis.Normed.Operator.BoundedLinearMaps
// Imports: public import Init public import Mathlib.Analysis.Normed.Module.Multilinear.Basic public import Mathlib.Analysis.Normed.Ring.Units public import Mathlib.Analysis.Normed.Operator.Mul
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
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toLinearMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toLinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_inc(x_9);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toLinearMap___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toLinearMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_IsBoundedLinearMap_toLinearMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toLinearMap___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_IsBoundedLinearMap_toLinearMap___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_inc(x_9);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_IsBoundedLinearMap_toContinuousLinearMap___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_Multilinear_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Ring_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Operator_Mul(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Operator_BoundedLinearMaps(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Module_Multilinear_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Ring_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Operator_Mul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
