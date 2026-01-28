// Lean compiler output
// Module: Mathlib.Topology.UniformSpace.CompactConvergence
// Imports: public import Init public import Mathlib.Topology.CompactOpen public import Mathlib.Topology.Compactness.CompactlyCoherentSpace public import Mathlib.Topology.Maps.Proper.Basic public import Mathlib.Topology.UniformSpace.Compact public import Mathlib.Topology.UniformSpace.UniformConvergenceTopology
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
LEAN_EXPORT lean_object* lp_mathlib_UniformEquiv_arrowCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compactConvergenceUniformSpace___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ContinuousMap_compactConvergenceUniformSpace___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_toUniformOnFunIsCompact(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ContinuousMap_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UniformEquiv_arrowCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Homeomorph_instEquivLike(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_UniformOnFun_ofFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UniformEquiv_arrowCongr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UniformEquiv_arrowCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compactConvergenceUniformSpace(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_UniformOnFun_ofFun(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg___lam__0), 2, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_apply_2(x_4, x_5, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_toUniformOnFunIsCompact(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_toUniformOnFunIsCompact(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
static lean_object* _init_lp_mathlib_ContinuousMap_compactConvergenceUniformSpace___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compactConvergenceUniformSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ContinuousMap_compactConvergenceUniformSpace___closed__0;
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compactConvergenceUniformSpace___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ContinuousMap_compactConvergenceUniformSpace(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UniformEquiv_arrowCongr___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_apply_1(x_1, x_2);
x_8 = lp_mathlib_Equiv_symm___redArg(x_3);
x_9 = lean_apply_1(x_4, x_8);
x_10 = lp_mathlib_ContinuousMap_comp___redArg(x_5, x_9);
x_11 = lp_mathlib_ContinuousMap_comp___redArg(x_7, x_10);
x_12 = lean_apply_1(x_11, x_6);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UniformEquiv_arrowCongr___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lp_mathlib_Equiv_symm___redArg(x_1);
x_8 = lean_apply_1(x_2, x_7);
x_9 = lean_apply_1(x_3, x_4);
x_10 = lp_mathlib_ContinuousMap_comp___redArg(x_5, x_9);
x_11 = lp_mathlib_ContinuousMap_comp___redArg(x_8, x_10);
x_12 = lean_apply_1(x_11, x_6);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UniformEquiv_arrowCongr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_ctor_get(x_4, 0);
lean_inc(x_8);
lean_dec_ref(x_4);
x_9 = lp_mathlib_Homeomorph_instEquivLike(lean_box(0), lean_box(0), x_7, x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_Homeomorph_instEquivLike(lean_box(0), lean_box(0), x_3, x_1);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lp_mathlib_Homeomorph_instEquivLike(lean_box(0), lean_box(0), x_8, x_7);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib_Homeomorph_instEquivLike(lean_box(0), lean_box(0), x_1, x_3);
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = lean_ctor_get(x_15, 1);
lean_dec(x_18);
lean_inc_ref(x_5);
lean_inc_ref(x_6);
x_19 = lean_alloc_closure((void*)(lp_mathlib_UniformEquiv_arrowCongr___redArg___lam__0), 6, 4);
lean_closure_set(x_19, 0, x_10);
lean_closure_set(x_19, 1, x_6);
lean_closure_set(x_19, 2, x_5);
lean_closure_set(x_19, 3, x_12);
x_20 = lean_alloc_closure((void*)(lp_mathlib_UniformEquiv_arrowCongr___redArg___lam__1), 6, 4);
lean_closure_set(x_20, 0, x_6);
lean_closure_set(x_20, 1, x_14);
lean_closure_set(x_20, 2, x_17);
lean_closure_set(x_20, 3, x_5);
lean_ctor_set(x_15, 1, x_20);
lean_ctor_set(x_15, 0, x_19);
return x_15;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_21 = lean_ctor_get(x_15, 0);
lean_inc(x_21);
lean_dec(x_15);
lean_inc_ref(x_5);
lean_inc_ref(x_6);
x_22 = lean_alloc_closure((void*)(lp_mathlib_UniformEquiv_arrowCongr___redArg___lam__0), 6, 4);
lean_closure_set(x_22, 0, x_10);
lean_closure_set(x_22, 1, x_6);
lean_closure_set(x_22, 2, x_5);
lean_closure_set(x_22, 3, x_12);
x_23 = lean_alloc_closure((void*)(lp_mathlib_UniformEquiv_arrowCongr___redArg___lam__1), 6, 4);
lean_closure_set(x_23, 0, x_6);
lean_closure_set(x_23, 1, x_14);
lean_closure_set(x_23, 2, x_21);
lean_closure_set(x_23, 3, x_5);
x_24 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_23);
return x_24;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_UniformEquiv_arrowCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_UniformEquiv_arrowCongr___redArg(x_3, x_4, x_7, x_8, x_9, x_10);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_CompactOpen(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Compactness_CompactlyCoherentSpace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Maps_Proper_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Compact(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_UniformConvergenceTopology(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_CompactConvergence(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_CompactOpen(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Compactness_CompactlyCoherentSpace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Maps_Proper_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_Compact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_UniformConvergenceTopology(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg___closed__0 = _init_lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg___closed__0();
lean_mark_persistent(lp_mathlib_ContinuousMap_toUniformOnFunIsCompact___redArg___closed__0);
lp_mathlib_ContinuousMap_compactConvergenceUniformSpace___closed__0 = _init_lp_mathlib_ContinuousMap_compactConvergenceUniformSpace___closed__0();
lean_mark_persistent(lp_mathlib_ContinuousMap_compactConvergenceUniformSpace___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
