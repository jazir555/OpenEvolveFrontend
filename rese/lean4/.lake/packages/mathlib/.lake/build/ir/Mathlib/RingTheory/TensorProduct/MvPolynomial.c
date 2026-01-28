// Lean compiler output
// Module: Mathlib.RingTheory.TensorProduct.MvPolynomial
// Imports: public import Init public import Mathlib.LinearAlgebra.DirectSum.Finsupp public import Mathlib.Algebra.MvPolynomial.Eval public import Mathlib.RingTheory.TensorProduct.Basic public import Mathlib.Algebra.MvPolynomial.Equiv public import Mathlib.RingTheory.IsTensorProduct
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_DirectSum_Finsupp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Eval(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_IsTensorProduct(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_MvPolynomial(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_DirectSum_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_IsTensorProduct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
