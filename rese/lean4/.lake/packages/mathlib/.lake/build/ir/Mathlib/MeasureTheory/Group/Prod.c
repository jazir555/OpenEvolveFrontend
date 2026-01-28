// Lean compiler output
// Module: Mathlib.MeasureTheory.Group.Prod
// Imports: public import Init public import Mathlib.MeasureTheory.Group.Measure public import Mathlib.MeasureTheory.Measure.Prod
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
lean_object* lp_mathlib_Equiv_subRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearDivRight___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_divRight(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodShear___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearAddRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearMulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_mulLeft___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearDivRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearMulRight___redArg(lean_object*);
static lean_object* lp_mathlib_MeasurableEquiv_shearMulRight___redArg___closed__0;
lean_object* lp_mathlib_Equiv_addLeft___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearSubRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearSubRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearAddRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_MeasurableEquiv_shearMulRight___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearMulRight___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_MeasurableEquiv_shearMulRight___redArg___closed__0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulLeft___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_Equiv_prodShear___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearMulRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MeasurableEquiv_shearMulRight___redArg(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearAddRight___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_MeasurableEquiv_shearMulRight___redArg___closed__0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_addLeft___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_Equiv_prodShear___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearAddRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MeasurableEquiv_shearAddRight___redArg(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearDivRight___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_MeasurableEquiv_shearMulRight___redArg___closed__0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_divRight), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_Equiv_prodShear___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearDivRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MeasurableEquiv_shearDivRight___redArg(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearSubRight___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_MeasurableEquiv_shearMulRight___redArg___closed__0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_subRight), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_Equiv_prodShear___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableEquiv_shearSubRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MeasurableEquiv_shearSubRight___redArg(x_3);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_Measure(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Group_Measure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MeasurableEquiv_shearMulRight___redArg___closed__0 = _init_lp_mathlib_MeasurableEquiv_shearMulRight___redArg___closed__0();
lean_mark_persistent(lp_mathlib_MeasurableEquiv_shearMulRight___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
