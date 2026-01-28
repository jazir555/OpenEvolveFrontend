// Lean compiler output
// Module: Mathlib.Topology.Bornology.Basic
// Imports: public import Init public import Mathlib.Order.Filter.Cofinite
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
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBornology(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBornology___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bornology_cofinite(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bornology_cobounded_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bornology_cobounded_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBornologyPUnit;
LEAN_EXPORT lean_object* lp_mathlib_Bornology_ofBounded(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bornology_ofBounded_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bornology_cobounded_x27(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bornology_cobounded_x27___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bornology_ofBounded(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bornology_ofBounded_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
static lean_object* _init_lp_mathlib_instBornologyPUnit() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bornology_cofinite(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBornology(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBornology___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Cofinite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Bornology_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Cofinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instBornologyPUnit = _init_lp_mathlib_instBornologyPUnit();
lean_mark_persistent(lp_mathlib_instBornologyPUnit);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
