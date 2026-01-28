// Lean compiler output
// Module: Mathlib.Topology.Algebra.GroupWithZero
// Imports: public import Init public import Mathlib.Algebra.Group.Pi.Lemmas public import Mathlib.Algebra.GroupWithZero.Units.Equiv public import Mathlib.Topology.Algebra.Monoid public import Mathlib.Topology.Homeomorph.Lemmas
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
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight_u2080___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft_u2080(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight_u2080(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_mulLeft_u2080___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_mulRight_u2080___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft_u2080___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft_u2080(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_mulLeft_u2080___redArg(x_3, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft_u2080___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_mulLeft_u2080___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight_u2080(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_mulRight_u2080___redArg(x_3, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight_u2080___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_mulRight_u2080___redArg(x_1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Monoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Homeomorph_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Algebra_GroupWithZero(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Homeomorph_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
