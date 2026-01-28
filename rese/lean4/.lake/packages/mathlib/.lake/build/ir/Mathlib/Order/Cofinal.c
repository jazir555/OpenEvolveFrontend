// Lean compiler output
// Module: Mathlib.Order.Cofinal
// Imports: public import Init public import Mathlib.Order.GaloisConnection.Basic public import Mathlib.Order.Interval.Set.Basic public import Mathlib.Order.WellFounded
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
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSubtypeSetIsCofinal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSubtypeSetIsCofinal___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSubtypeSetIsCofinal(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSubtypeSetIsCofinal___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instInhabitedSubtypeSetIsCofinal(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Cofinal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
