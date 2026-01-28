// Lean compiler output
// Module: Mathlib.FieldTheory.IsAlgClosed.AlgebraicClosure
// Imports: public import Init public import Mathlib.Algebra.CharP.Algebra public import Mathlib.Data.Multiset.Fintype public import Mathlib.FieldTheory.IsAlgClosed.Basic public import Mathlib.FieldTheory.SplittingField.Construction
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
LEAN_EXPORT lean_object* lp_mathlib_AlgebraicClosure_spanCoeffs___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgebraicClosure_spanCoeffs(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgebraicClosure_spanCoeffs(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgebraicClosure_spanCoeffs___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AlgebraicClosure_spanCoeffs(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_CharP_Algebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Fintype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_FieldTheory_IsAlgClosed_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_FieldTheory_SplittingField_Construction(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_FieldTheory_IsAlgClosed_AlgebraicClosure(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharP_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Fintype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_FieldTheory_IsAlgClosed_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_FieldTheory_SplittingField_Construction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
