// Lean compiler output
// Module: Mathlib.Topology.OpenPartialHomeomorph.Constructions
// Imports: public import Init public import Mathlib.Topology.OpenPartialHomeomorph.Composition
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
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_const___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_transHomeomorph___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialEquiv_disjointUnion___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_transHomeomorph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialEquiv_single___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_pi___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_disjointUnion___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_piecewise___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_disjointUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialEquiv_transEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_PartialEquiv_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_const(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_piecewise(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_pi___redArg(lean_object*);
lean_object* lp_mathlib_PartialEquiv_piecewise___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialEquiv_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_const(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_PartialEquiv_single___redArg(x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_const___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PartialEquiv_single___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_prod(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_PartialEquiv_prod___redArg(x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_prod___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PartialEquiv_prod___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_pi___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_pi___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_OpenPartialHomeomorph_pi___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_PartialEquiv_pi___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_pi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_OpenPartialHomeomorph_pi___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_pi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_OpenPartialHomeomorph_pi(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_piecewise(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_PartialEquiv_piecewise___redArg(x_5, x_6, x_9, x_10);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_piecewise___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PartialEquiv_piecewise___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_disjointUnion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_PartialEquiv_disjointUnion___redArg(x_5, x_6, x_7, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_disjointUnion___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PartialEquiv_disjointUnion___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_transHomeomorph(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_PartialEquiv_transEquiv___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_transHomeomorph___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PartialEquiv_transEquiv___redArg(x_1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_OpenPartialHomeomorph_Composition(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_OpenPartialHomeomorph_Constructions(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_OpenPartialHomeomorph_Composition(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
