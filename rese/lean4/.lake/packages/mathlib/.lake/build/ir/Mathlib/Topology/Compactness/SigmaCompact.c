// Lean compiler output
// Module: Mathlib.Topology.Compactness.SigmaCompact
// Imports: public import Init public import Mathlib.Topology.Bases public import Mathlib.Topology.Compactness.LocallyCompact public import Mathlib.Topology.Compactness.LocallyFinite
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
LEAN_EXPORT lean_object* lp_mathlib_CompactExhaustion_shiftr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompactExhaustion_instFunLikeNatSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompactExhaustion_instFunLikeNatSet(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompactExhaustion_shiftr(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Bases(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Compactness_LocallyCompact(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Compactness_LocallyFinite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Compactness_SigmaCompact(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Bases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Compactness_LocallyCompact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Compactness_LocallyFinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
