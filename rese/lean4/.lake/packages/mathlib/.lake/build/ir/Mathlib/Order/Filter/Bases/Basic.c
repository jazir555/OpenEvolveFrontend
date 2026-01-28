// Lean compiler output
// Module: Mathlib.Order.Filter.Bases.Basic
// Imports: public import Init public import Mathlib.Data.Set.Sigma public import Mathlib.Order.Filter.Defs public import Mathlib.Order.Filter.Map public import Mathlib.Order.Interval.Set.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_instMembershipSetFilterBasis(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_IsBasis_filterBasis(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_IsBasis_filter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_asBasis(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FilterBasis_filter(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedFilterBasisNat;
LEAN_EXPORT lean_object* lp_mathlib_instMembershipSetFilterBasis(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instInhabitedFilterBasisNat() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_asBasis(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_IsBasis_filterBasis(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FilterBasis_filter(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_IsBasis_filter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Map(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Filter_Bases_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instInhabitedFilterBasisNat = _init_lp_mathlib_instInhabitedFilterBasisNat();
lean_mark_persistent(lp_mathlib_instInhabitedFilterBasisNat);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
