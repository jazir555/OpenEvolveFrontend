// Lean compiler output
// Module: Mathlib.RingTheory.Flat.Basic
// Imports: public import Init public import Mathlib.Algebra.Colimit.TensorProduct public import Mathlib.Algebra.Module.Projective public import Mathlib.LinearAlgebra.TensorProduct.RightExactness public import Mathlib.RingTheory.Finiteness.Small public import Mathlib.RingTheory.IsTensorProduct public import Mathlib.RingTheory.TensorProduct.Finite public import Mathlib.RingTheory.Adjoin.FGBaseChange
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
lean_object* initialize_mathlib_Mathlib_Algebra_Colimit_TensorProduct(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Projective(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_RightExactness(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Small(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_IsTensorProduct(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Adjoin_FGBaseChange(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Flat_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Colimit_TensorProduct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Projective(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_RightExactness(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Finiteness_Small(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_IsTensorProduct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Adjoin_FGBaseChange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
