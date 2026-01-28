// Lean compiler output
// Module: Mathlib.RingTheory.TensorProduct.Finite
// Imports: public import Init public import Mathlib.LinearAlgebra.Isomorphisms public import Mathlib.RingTheory.Finiteness.Basic public import Mathlib.RingTheory.Finiteness.Bilinear public import Mathlib.RingTheory.Ideal.Quotient.Basic public import Mathlib.RingTheory.TensorProduct.Maps
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Bilinear(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_Maps(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_Finite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Finiteness_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
