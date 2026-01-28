// Lean compiler output
// Module: Mathlib.RingTheory.SimpleModule.Basic
// Imports: public import Init public import Mathlib.Algebra.DirectSum.Module public import Mathlib.Data.Finite.Card public import Mathlib.Data.Matrix.Mul public import Mathlib.LinearAlgebra.DFinsupp public import Mathlib.LinearAlgebra.Finsupp.Span public import Mathlib.LinearAlgebra.Isomorphisms public import Mathlib.LinearAlgebra.Projection public import Mathlib.Order.Atoms.Finite public import Mathlib.Order.CompactlyGenerated.Intervals public import Mathlib.Order.JordanHolder public import Mathlib.RingTheory.Ideal.Colon public import Mathlib.RingTheory.Noetherian.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_JordanHolderModule_instJordanHolderLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_JordanHolderModule_instJordanHolderLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_JordanHolderModule_instJordanHolderLattice(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_JordanHolderModule_instJordanHolderLattice___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_JordanHolderModule_instJordanHolderLattice(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finite_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Mul(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Span(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Projection(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Atoms_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompactlyGenerated_Intervals(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_JordanHolder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Colon(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Noetherian_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_SimpleModule_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finite_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Mul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Span(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Projection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Atoms_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompactlyGenerated_Intervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_JordanHolder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Colon(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Noetherian_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
