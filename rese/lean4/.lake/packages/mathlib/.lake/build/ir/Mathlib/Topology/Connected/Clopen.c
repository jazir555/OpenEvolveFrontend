// Lean compiler output
// Module: Mathlib.Topology.Connected.Clopen
// Imports: public import Init public import Mathlib.Data.Set.Subset public import Mathlib.Topology.Clopen public import Mathlib.Topology.Compactness.Compact public import Mathlib.Topology.Connected.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_mk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instCoeTC(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instTopologicalSpace(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instCoeTC___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_connectedComponentSetoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_connectedComponentSetoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_mk(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_mk___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_mk___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ConnectedComponents_mk(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_mk___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ConnectedComponents_mk___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instCoeTC(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ConnectedComponents_mk___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instCoeTC___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ConnectedComponents_mk___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instInhabited___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instInhabited___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ConnectedComponents_instInhabited(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instInhabited___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ConnectedComponents_instInhabited___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConnectedComponents_instTopologicalSpace(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Subset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Clopen(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Compactness_Compact(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Connected_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Connected_Clopen(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Subset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Clopen(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Compactness_Compact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Connected_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
