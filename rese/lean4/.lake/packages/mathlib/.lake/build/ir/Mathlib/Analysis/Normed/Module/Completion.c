// Lean compiler output
// Module: Mathlib.Analysis.Normed.Module.Completion
// Imports: public import Init public import Mathlib.Analysis.Normed.Group.Completion public import Mathlib.Analysis.Normed.Operator.NormedSpace public import Mathlib.Topology.Algebra.UniformRing public import Mathlib.Topology.Algebra.UniformField
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
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toComplL___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toComplL(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62___redArg(lean_object*);
lean_object* lp_mathlib_UniformSpace_Completion_coe_x27___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toComplL___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_UniformSpace_Completion_coe_x27___boxed), 3, 2);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62___redArg(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toComplL(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62___redArg(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toComplL___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_UniformSpace_Completion_toCompl_u2097_u1d62___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UniformSpace_Completion_toComplL___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_UniformSpace_Completion_toComplL(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Completion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Operator_NormedSpace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_UniformRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_UniformField(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_Completion(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Group_Completion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Operator_NormedSpace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_UniformRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_UniformField(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
