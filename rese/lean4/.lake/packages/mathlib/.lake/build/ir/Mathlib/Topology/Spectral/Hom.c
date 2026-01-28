// Lean compiler output
// Module: Mathlib.Topology.Spectral.Hom
// Imports: public import Init public import Mathlib.Tactic.StacksAttribute public import Mathlib.Topology.ContinuousMap.Basic public import Mathlib.Topology.Maps.Proper.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_toContinuousMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSpectralMapOfSpectralMapClass___redArg(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_id(lean_object*, lean_object*);
lean_object* lp_mathlib_ContinuousMap_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_toContinuousMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSpectralMapOfSpectralMapClass___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_toContinuousMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_comp___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_SpectralMap_id___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_instFunLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_toContinuousMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSpectralMapOfSpectralMapClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSpectralMapOfSpectralMapClass___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSpectralMapOfSpectralMapClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSpectralMapOfSpectralMapClass___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSpectralMapOfSpectralMapClass(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_instCoeTCSpectralMapOfSpectralMapClass___redArg(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_toContinuousMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_toContinuousMap___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_toContinuousMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_SpectralMap_toContinuousMap(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_toContinuousMap___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SpectralMap_toContinuousMap___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_instFunLike___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_instFunLike(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_SpectralMap_instFunLike___lam__0), 2, 0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_copy(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_inc(x_6);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_copy___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_copy___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_SpectralMap_copy(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec(x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_copy___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SpectralMap_copy___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_SpectralMap_id___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_id(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SpectralMap_id___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SpectralMap_id___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_comp___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_ContinuousMap_comp___redArg(x_1, x_2);
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_comp___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_SpectralMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SpectralMap_comp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_SpectralMap_comp___redArg(x_7, x_8);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_StacksAttribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_ContinuousMap_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Maps_Proper_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Spectral_Hom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_StacksAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_ContinuousMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Maps_Proper_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_SpectralMap_id___closed__0 = _init_lp_mathlib_SpectralMap_id___closed__0();
lean_mark_persistent(lp_mathlib_SpectralMap_id___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
