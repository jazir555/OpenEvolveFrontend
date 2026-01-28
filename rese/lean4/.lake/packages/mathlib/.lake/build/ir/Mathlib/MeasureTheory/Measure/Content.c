// Lean compiler output
// Module: Mathlib.MeasureTheory.Measure.Content
// Imports: public import Init public import Mathlib.MeasureTheory.Measure.MeasureSpace public import Mathlib.MeasureTheory.Measure.Regular public import Mathlib.Topology.Sets.Compacts
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
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_instInhabitedContent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Content_instFunLikeCompactsENNReal___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_instInhabitedContent___lam__0___boxed(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1850581184____hygCtx___hyg_8_;
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_instInhabitedContent___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Content_instFunLikeCompactsENNReal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_instInhabitedContent___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_instInhabitedContent___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MeasureTheory_instInhabitedContent___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_instInhabitedContent(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1850581184____hygCtx___hyg_8_;
x_4 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_instInhabitedContent___lam__0___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Content_instFunLikeCompactsENNReal___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Content_instFunLikeCompactsENNReal(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Content_instFunLikeCompactsENNReal___lam__0), 2, 0);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_MeasureSpace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Regular(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Sets_Compacts(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Content(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_MeasureSpace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Sets_Compacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
