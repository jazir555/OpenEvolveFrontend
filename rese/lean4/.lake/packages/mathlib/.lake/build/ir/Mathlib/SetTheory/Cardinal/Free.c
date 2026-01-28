// Lean compiler output
// Module: Mathlib.SetTheory.Cardinal.Free
// Imports: public import Init public import Mathlib.Algebra.FreeAbelianGroup.Finsupp public import Mathlib.Algebra.Ring.TransferInstance public import Mathlib.Data.Finsupp.Fintype public import Mathlib.Data.ZMod.Defs public import Mathlib.GroupTheory.FreeGroup.Reduce public import Mathlib.RingTheory.FreeCommRing public import Mathlib.SetTheory.Cardinal.Arithmetic public import Mathlib.SetTheory.Cardinal.Finsupp
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
lean_object* initialize_mathlib_Mathlib_Algebra_FreeAbelianGroup_Finsupp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_TransferInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Fintype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Reduce(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_FreeCommRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Arithmetic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Finsupp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Free(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_FreeAbelianGroup_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Fintype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Reduce(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_FreeCommRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Arithmetic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
