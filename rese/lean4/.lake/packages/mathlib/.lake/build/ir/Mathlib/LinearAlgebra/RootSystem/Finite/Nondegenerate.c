// Lean compiler output
// Module: Mathlib.LinearAlgebra.RootSystem.Finite.Nondegenerate
// Imports: public import Init public import Mathlib.LinearAlgebra.BilinearForm.Basic public import Mathlib.LinearAlgebra.BilinearForm.Orthogonal public import Mathlib.LinearAlgebra.Dimension.Localization public import Mathlib.LinearAlgebra.QuadraticForm.Basic public import Mathlib.LinearAlgebra.RootSystem.BaseChange public import Mathlib.LinearAlgebra.RootSystem.Finite.CanonicalBilinear
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
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_toInvariantForm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_toInvariantForm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RootPairing_RootForm___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_toInvariantForm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_toInvariantForm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_RootPairing_RootForm___redArg(x_8, x_11, x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_toInvariantForm___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RootPairing_RootForm___redArg(x_2, x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_toInvariantForm___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_RootPairing_toInvariantForm(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_13;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_BilinearForm_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_BilinearForm_Orthogonal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Dimension_Localization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_QuadraticForm_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_BaseChange(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Finite_CanonicalBilinear(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Finite_Nondegenerate(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_BilinearForm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_BilinearForm_Orthogonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Dimension_Localization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_QuadraticForm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_BaseChange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Finite_CanonicalBilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
