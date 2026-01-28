// Lean compiler output
// Module: Mathlib.Data.Fintype.BigOperators
// Imports: public import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise public import Mathlib.Algebra.BigOperators.Group.Finset.Sigma public import Mathlib.Algebra.BigOperators.Option public import Mathlib.Data.Fintype.Option public import Mathlib.Data.Fintype.Prod public import Mathlib.Data.Fintype.Sigma public import Mathlib.Data.Fintype.Sum public import Mathlib.Data.Fintype.Vector
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Piecewise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Option(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Vector(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_BigOperators(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Vector(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
