// Lean compiler output
// Module: Mathlib.Topology.Spectral.Prespectral
// Imports: public import Init public import Mathlib.Order.Ideal public import Mathlib.Topology.Sets.Compacts public import Mathlib.Topology.Sets.OpenCover public import Mathlib.Topology.Spectral.Hom
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
LEAN_EXPORT lean_object* lp_mathlib_PrespectralSpace_opensEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrespectralSpace_opensEquiv___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_TopologicalSpace_Opens_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrespectralSpace_opensEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrespectralSpace_opensEquiv___redArg___lam__0(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrespectralSpace_opensEquiv___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrespectralSpace_opensEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrespectralSpace_opensEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_TopologicalSpace_Opens_instCompleteLattice(lean_box(0), x_1);
x_3 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_PrespectralSpace_opensEquiv___redArg___lam__0), 1, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_PrespectralSpace_opensEquiv___redArg___lam__1), 2, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrespectralSpace_opensEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_PrespectralSpace_opensEquiv___redArg(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Ideal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Sets_Compacts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Sets_OpenCover(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Spectral_Hom(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Spectral_Prespectral(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Ideal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Sets_Compacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Sets_OpenCover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Spectral_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
