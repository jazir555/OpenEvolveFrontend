// Lean compiler output
// Module: Mathlib.Topology.Order.Real
// Imports: public import Init public import Mathlib.Data.EReal.Basic public import Mathlib.Topology.Order.T5
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
LEAN_EXPORT lean_object* lp_mathlib_EReal_instTopologicalSpace;
LEAN_EXPORT lean_object* lp_mathlib_ENNReal_instTopologicalSpace;
static lean_object* _init_lp_mathlib_EReal_instTopologicalSpace() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ENNReal_instTopologicalSpace() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_EReal_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_T5(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Order_Real(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_EReal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_T5(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_EReal_instTopologicalSpace = _init_lp_mathlib_EReal_instTopologicalSpace();
lean_mark_persistent(lp_mathlib_EReal_instTopologicalSpace);
lp_mathlib_ENNReal_instTopologicalSpace = _init_lp_mathlib_ENNReal_instTopologicalSpace();
lean_mark_persistent(lp_mathlib_ENNReal_instTopologicalSpace);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
