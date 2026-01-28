// Lean compiler output
// Module: Mathlib.Data.Fintype.Powerset
// Imports: public import Init public import Mathlib.Data.Finset.Powerset public import Mathlib.Data.Fintype.EquivFin
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
LEAN_EXPORT lean_object* lp_mathlib_Finset_fintype(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintype(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_powerset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_fintype(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_powerset___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_fintype___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Finset_powerset___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintype___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Finset_powerset___redArg(x_1);
x_3 = lp_mathlib_Finset_map___redArg(lean_box(0), x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintype(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_fintype___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Powerset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Powerset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
