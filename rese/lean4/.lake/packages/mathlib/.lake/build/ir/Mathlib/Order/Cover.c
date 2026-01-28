// Lean compiler output
// Module: Mathlib.Order.Cover
// Imports: public import Init public import Mathlib.Order.Antisymmetrization public import Mathlib.Order.Hom.WithTopBot public import Mathlib.Order.Interval.Set.OrdConnected public import Mathlib.Order.Interval.Set.WithBotTop
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
LEAN_EXPORT lean_object* lp_mathlib_Bool_instDecidableRelCovBy___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Bool_instDecidableRelCovBy(uint8_t, uint8_t);
uint8_t l_Bool_instDecidableLt(uint8_t, uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_Bool_instDecidableRelWCovBy(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool_instDecidableRelWCovBy___boxed(lean_object*, lean_object*);
uint8_t l_Bool_instDecidableLe(uint8_t, uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_Bool_instDecidableRelWCovBy(uint8_t x_1, uint8_t x_2) {
_start:
{
uint8_t x_3; 
x_3 = l_Bool_instDecidableLe(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_instDecidableRelWCovBy___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_mathlib_Bool_instDecidableRelWCovBy(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_instDecidableRelCovBy(uint8_t x_1, uint8_t x_2) {
_start:
{
uint8_t x_3; 
x_3 = l_Bool_instDecidableLt(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_instDecidableRelCovBy___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_mathlib_Bool_instDecidableRelCovBy(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Antisymmetrization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_WithTopBot(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_WithBotTop(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Antisymmetrization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_WithTopBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_WithBotTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
