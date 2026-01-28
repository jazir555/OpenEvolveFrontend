// Lean compiler output
// Module: Mathlib.Analysis.SpecificLimits.Normed
// Imports: public import Init public import Mathlib.Algebra.BigOperators.Module public import Mathlib.Algebra.Order.Field.Power public import Mathlib.Algebra.Polynomial.Monic public import Mathlib.Analysis.Asymptotics.Lemmas public import Mathlib.Analysis.Normed.Ring.InfiniteSum public import Mathlib.Analysis.Normed.Module.Basic public import Mathlib.Analysis.Normed.Order.Lattice public import Mathlib.Analysis.SpecificLimits.Basic public import Mathlib.Data.List.TFAE public import Mathlib.Data.Nat.Choose.Bounds public import Mathlib.Order.Filter.AtTopBot.ModEq public import Mathlib.RingTheory.Polynomial.Pochhammer public import Mathlib.Tactic.NoncommRing
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Field_Power(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Monic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Asymptotics_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Ring_InfiniteSum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Order_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_SpecificLimits_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_TFAE(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Choose_Bounds(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_AtTopBot_ModEq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Polynomial_Pochhammer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NoncommRing(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_SpecificLimits_Normed(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Field_Power(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Monic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Asymptotics_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Ring_InfiniteSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Module_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_SpecificLimits_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_TFAE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Choose_Bounds(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_AtTopBot_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Polynomial_Pochhammer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NoncommRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
