// Lean compiler output
// Module: Mathlib.Data.Multiset.Antidiagonal
// Imports: public import Init public import Mathlib.Data.Multiset.Powerset
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
lean_object* lp_mathlib_Multiset_powersetAux___redArg(lean_object*);
lean_object* lp_batteries_List_revzip___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_antidiagonal___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_antidiagonal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_antidiagonal___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Multiset_powersetAux___redArg(x_1);
x_3 = lp_batteries_List_revzip___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_antidiagonal(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_antidiagonal___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Powerset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Antidiagonal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
