// Lean compiler output
// Module: Mathlib.Topology.UniformSpace.Real
// Imports: public import Init public import Mathlib.Topology.ContinuousMap.Basic public import Mathlib.Topology.MetricSpace.Cauchy
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
LEAN_EXPORT lean_object* lp_mathlib_NNReal_instTopologicalSpace;
extern lean_object* lp_mathlib_instPseudoMetricSpaceNNReal;
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeNNRealReal;
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeNNRealReal___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeNNRealReal___lam__0(lean_object*);
static lean_object* _init_lp_mathlib_NNReal_instTopologicalSpace() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instPseudoMetricSpaceNNReal;
x_2 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeNNRealReal___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeNNRealReal___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ContinuousMap_coeNNRealReal___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ContinuousMap_coeNNRealReal() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_coeNNRealReal___lam__0___boxed), 1, 0);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_ContinuousMap_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Cauchy(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Real(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_ContinuousMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Cauchy(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_NNReal_instTopologicalSpace = _init_lp_mathlib_NNReal_instTopologicalSpace();
lean_mark_persistent(lp_mathlib_NNReal_instTopologicalSpace);
lp_mathlib_ContinuousMap_coeNNRealReal = _init_lp_mathlib_ContinuousMap_coeNNRealReal();
lean_mark_persistent(lp_mathlib_ContinuousMap_coeNNRealReal);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
