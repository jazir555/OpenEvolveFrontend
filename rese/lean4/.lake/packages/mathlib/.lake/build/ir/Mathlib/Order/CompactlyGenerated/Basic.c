// Lean compiler output
// Module: Mathlib.Order.CompactlyGenerated.Basic
// Imports: public import Init public import Mathlib.Order.Atoms public import Mathlib.Order.OrderIsoNat public import Mathlib.Order.RelIso.Set public import Mathlib.Order.SupClosed public import Mathlib.Order.SupIndep public import Mathlib.Order.Zorn public import Mathlib.Data.Finset.Order public import Mathlib.Order.Interval.Set.OrderIso public import Mathlib.Data.Finite.Set public import Mathlib.Tactic.TFAE
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
lean_object* initialize_mathlib_Mathlib_Order_Atoms(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_OrderIsoNat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelIso_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SupClosed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SupIndep(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Zorn(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_OrderIso(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finite_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TFAE(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_CompactlyGenerated_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Atoms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_OrderIsoNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelIso_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SupClosed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SupIndep(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Zorn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_OrderIso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finite_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TFAE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
