// Lean compiler output
// Module: Mathlib.Topology.Bornology.Constructions
// Imports: public import Init public import Mathlib.Algebra.Group.TypeTags.Basic public import Mathlib.Topology.Bornology.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Bornology_induced___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBornology(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBornologyOrderDual___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBornologyAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBornology___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBornologyMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBornologyMultiplicative(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBornology(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBornologySubtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBornologyOrderDual(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBornologyAdditive(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bornology_induced(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBornology(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBornology(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBornology___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_instBornology(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bornology_induced(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bornology_induced___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Bornology_induced(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBornologySubtype(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBornologyAdditive(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBornologyAdditive___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBornologyMultiplicative(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBornologyMultiplicative___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBornologyOrderDual(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBornologyOrderDual___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Bornology_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Bornology_Constructions(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Bornology_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
