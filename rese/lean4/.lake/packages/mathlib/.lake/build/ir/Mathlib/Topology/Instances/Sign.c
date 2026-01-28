// Lean compiler output
// Module: Mathlib.Topology.Instances.Sign
// Imports: public import Init public import Mathlib.Data.Sign.Defs public import Mathlib.Topology.Order.Basic
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
lean_object* lp_mathlib_TopologicalSpace_instCompleteLattice(lean_object*);
static lean_object* lp_mathlib_instTopologicalSpaceSignType___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceSignType;
static lean_object* _init_lp_mathlib_instTopologicalSpaceSignType___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_TopologicalSpace_instCompleteLattice(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_instTopologicalSpaceSignType() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instTopologicalSpaceSignType___closed__0;
x_2 = lean_ctor_get(x_1, 3);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sign_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Instances_Sign(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sign_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instTopologicalSpaceSignType___closed__0 = _init_lp_mathlib_instTopologicalSpaceSignType___closed__0();
lean_mark_persistent(lp_mathlib_instTopologicalSpaceSignType___closed__0);
lp_mathlib_instTopologicalSpaceSignType = _init_lp_mathlib_instTopologicalSpaceSignType();
lean_mark_persistent(lp_mathlib_instTopologicalSpaceSignType);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
