// Lean compiler output
// Module: Mathlib.Topology.Algebra.Constructions
// Imports: public import Init public import Mathlib.Topology.Separation.Hausdorff public import Mathlib.Topology.Homeomorph.Lemmas
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
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opHomeomorph(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instTopologicalSpaceUnits(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_instTopologicalSpaceAddUnits(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddOpposite_opHomeomorph___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instTopologicalSpaceAddOpposite(lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_opEquiv(lean_object*);
static lean_object* lp_mathlib_MulOpposite_opHomeomorph___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opHomeomorph(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_instTopologicalSpaceAddUnits___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instTopologicalSpaceMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instTopologicalSpaceUnits___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instTopologicalSpaceMulOpposite(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instTopologicalSpaceAddOpposite(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
static lean_object* _init_lp_mathlib_MulOpposite_opHomeomorph___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opHomeomorph(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_opHomeomorph___closed__0;
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddOpposite_opHomeomorph___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddOpposite_opEquiv(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opHomeomorph(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_opHomeomorph___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instTopologicalSpaceUnits(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instTopologicalSpaceUnits___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Units_instTopologicalSpaceUnits(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_instTopologicalSpaceAddUnits(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_instTopologicalSpaceAddUnits___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddUnits_instTopologicalSpaceAddUnits(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Separation_Hausdorff(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Homeomorph_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Constructions(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Separation_Hausdorff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Homeomorph_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MulOpposite_opHomeomorph___closed__0 = _init_lp_mathlib_MulOpposite_opHomeomorph___closed__0();
lean_mark_persistent(lp_mathlib_MulOpposite_opHomeomorph___closed__0);
lp_mathlib_AddOpposite_opHomeomorph___closed__0 = _init_lp_mathlib_AddOpposite_opHomeomorph___closed__0();
lean_mark_persistent(lp_mathlib_AddOpposite_opHomeomorph___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
