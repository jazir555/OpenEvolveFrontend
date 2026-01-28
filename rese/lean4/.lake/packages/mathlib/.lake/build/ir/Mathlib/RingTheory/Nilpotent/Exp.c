// Lean compiler output
// Module: Mathlib.RingTheory.Nilpotent.Exp
// Imports: public import Init public import Mathlib.Algebra.Algebra.Basic public import Mathlib.Algebra.Algebra.Bilinear public import Mathlib.Algebra.BigOperators.GroupWithZero.Action public import Mathlib.Algebra.Module.BigOperators public import Mathlib.Algebra.Module.Rat public import Mathlib.Data.Nat.Cast.Field public import Mathlib.LinearAlgebra.TensorProduct.Tower public import Mathlib.RingTheory.Nilpotent.Basic public import Mathlib.RingTheory.TensorProduct.Maps public import Mathlib.Tactic.FieldSimp
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Field(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Tower(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Nilpotent_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_Maps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FieldSimp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Nilpotent_Exp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Nilpotent_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FieldSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
