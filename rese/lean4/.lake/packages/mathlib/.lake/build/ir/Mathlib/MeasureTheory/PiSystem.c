// Lean compiler output
// Module: Mathlib.MeasureTheory.PiSystem
// Imports: public import Init public import Mathlib.Logic.Encodable.Lattice public import Mathlib.MeasureTheory.MeasurableSpace.Defs public import Mathlib.Order.Disjointed
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
static lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_toMeasurableSpace(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_ofMeasurableSpace(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_instLEDynkinSystem(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_generate(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_instPartialOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_restrictOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_instLEDynkinSystem(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
static lean_object* _init_lp_mathlib_MeasurableSpace_DynkinSystem_instPartialOrder___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_instPartialOrder(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MeasurableSpace_DynkinSystem_instPartialOrder___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_ofMeasurableSpace(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_generate(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_instInhabited(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_toMeasurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_DynkinSystem_restrictOn(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Disjointed(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_PiSystem(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Encodable_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Disjointed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MeasurableSpace_DynkinSystem_instPartialOrder___closed__0 = _init_lp_mathlib_MeasurableSpace_DynkinSystem_instPartialOrder___closed__0();
lean_mark_persistent(lp_mathlib_MeasurableSpace_DynkinSystem_instPartialOrder___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
