// Lean compiler output
// Module: Mathlib.Analysis.Normed.Group.Continuity
// Imports: public import Init public import Mathlib.Analysis.Normed.Group.Basic public import Mathlib.Topology.Algebra.Ring.Real public import Mathlib.Topology.Metrizable.Uniformity public import Mathlib.Topology.Sequences
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
LEAN_EXPORT lean_object* lp_mathlib_NormedCommGroup_toENormedCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedAddCommGroup_toENormedAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedAddGroup_toENormedAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeminormedAddGroup_toContinuousENorm(lean_object*, lean_object*);
lean_object* lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedCommGroup_toENormedCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeminormedGroup_toContinuousENorm___redArg(lean_object*);
lean_object* lp_mathlib_NormedCommGroup_toNormedGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedAddGroup_toENormedAddMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedGroup_toENormedMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedGroup_toENormedMonoid(lean_object*, lean_object*);
lean_object* lp_mathlib_NormedAddGroup_toSeminormedAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeminormedAddGroup_toContinuousENorm___redArg(lean_object*);
lean_object* lp_mathlib_NormedGroup_toSeminormedGroup___redArg(lean_object*);
lean_object* lp_mathlib_SeminormedAddGroup_toNNNorm___redArg(lean_object*);
lean_object* lp_mathlib_SeminormedGroup_toNNNorm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeminormedGroup_toContinuousENorm(lean_object*, lean_object*);
lean_object* lp_mathlib_NNNorm_toENorm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedAddCommGroup_toENormedAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeminormedGroup_toContinuousENorm___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_SeminormedGroup_toNNNorm___redArg(x_1);
x_3 = lp_mathlib_NNNorm_toENorm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeminormedGroup_toContinuousENorm(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SeminormedGroup_toContinuousENorm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeminormedAddGroup_toContinuousENorm___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_SeminormedAddGroup_toNNNorm___redArg(x_1);
x_3 = lp_mathlib_NNNorm_toENorm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeminormedAddGroup_toContinuousENorm(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SeminormedAddGroup_toContinuousENorm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedGroup_toENormedMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_NormedGroup_toSeminormedGroup___redArg(x_1);
x_3 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_SeminormedGroup_toContinuousENorm___redArg(x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedGroup_toENormedMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NormedGroup_toENormedMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedAddGroup_toENormedAddMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_NormedAddGroup_toSeminormedAddGroup___redArg(x_1);
x_3 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_SeminormedAddGroup_toContinuousENorm___redArg(x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedAddGroup_toENormedAddMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NormedAddGroup_toENormedAddMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedCommGroup_toENormedCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_NormedCommGroup_toNormedGroup___redArg(x_1);
x_3 = lp_mathlib_NormedGroup_toENormedMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedCommGroup_toENormedCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NormedCommGroup_toENormedCommMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedAddCommGroup_toENormedAddCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(x_1);
x_3 = lp_mathlib_NormedAddGroup_toENormedAddMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedAddCommGroup_toENormedAddCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NormedAddCommGroup_toENormedAddCommMonoid___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Ring_Real(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Metrizable_Uniformity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Sequences(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Continuity(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Ring_Real(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Metrizable_Uniformity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Sequences(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
