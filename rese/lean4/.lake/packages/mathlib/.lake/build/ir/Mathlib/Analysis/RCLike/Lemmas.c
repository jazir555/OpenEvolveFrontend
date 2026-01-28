// Lean compiler output
// Module: Mathlib.Analysis.RCLike.Lemmas
// Imports: public import Init public import Mathlib.Analysis.Normed.Module.FiniteDimension public import Mathlib.Analysis.RCLike.Basic public import Mathlib.Topology.Instances.RealVectorSpace
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
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_RCLike_x20instance;
static lean_object* _init_lp_mathlib_LibraryNote_RCLike_x20instance() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_FiniteDimension(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_RCLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Instances_RealVectorSpace(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_RCLike_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Module_FiniteDimension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_RCLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Instances_RealVectorSpace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_RCLike_x20instance = _init_lp_mathlib_LibraryNote_RCLike_x20instance();
lean_mark_persistent(lp_mathlib_LibraryNote_RCLike_x20instance);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
