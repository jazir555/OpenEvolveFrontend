// Lean compiler output
// Module: Mathlib.Topology.UniformSpace.Separation
// Imports: public import Init public import Mathlib.Tactic.ApplyFun public import Mathlib.Topology.Separation.Regular public import Mathlib.Topology.UniformSpace.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instUniformSpace(lean_object*, lean_object*);
static lean_object* lp_mathlib_SeparationQuotient_instUniformSpace___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instUniformSpace___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_SeparationQuotient_instUniformSpace___closed__0() {
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
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instUniformSpace(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SeparationQuotient_instUniformSpace___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instUniformSpace___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SeparationQuotient_instUniformSpace(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Separation_Regular(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Separation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Separation_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_SeparationQuotient_instUniformSpace___closed__0 = _init_lp_mathlib_SeparationQuotient_instUniformSpace___closed__0();
lean_mark_persistent(lp_mathlib_SeparationQuotient_instUniformSpace___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
