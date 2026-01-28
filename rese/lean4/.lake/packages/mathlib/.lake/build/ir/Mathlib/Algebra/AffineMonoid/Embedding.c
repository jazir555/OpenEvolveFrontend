// Lean compiler output
// Module: Mathlib.Algebra.AffineMonoid.Embedding
// Imports: public import Init public import Mathlib.GroupTheory.Finiteness public import Mathlib.GroupTheory.FreeAbelianGroup public import Mathlib.GroupTheory.MonoidLocalization.GrothendieckGroup public import Mathlib.LinearAlgebra.Dimension.Finrank import Mathlib.Algebra.EuclideanDomain.Int import Mathlib.GroupTheory.MonoidLocalization.Finite import Mathlib.LinearAlgebra.Dimension.Free import Mathlib.LinearAlgebra.FreeModule.PID
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
lean_object* initialize_mathlib_Mathlib_GroupTheory_Finiteness(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_FreeAbelianGroup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_GrothendieckGroup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Dimension_Finrank(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Dimension_Free(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_FreeModule_PID(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_AffineMonoid_Embedding(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Finiteness(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_FreeAbelianGroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_GrothendieckGroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Dimension_Finrank(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Dimension_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_FreeModule_PID(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
