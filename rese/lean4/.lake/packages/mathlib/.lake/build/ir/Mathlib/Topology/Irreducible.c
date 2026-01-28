// Lean compiler output
// Module: Mathlib.Topology.Irreducible
// Imports: public import Init public import Mathlib.Order.Minimal public import Mathlib.Order.Zorn public import Mathlib.Topology.ContinuousOn public import Mathlib.Tactic.StacksAttribute
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
LEAN_EXPORT lean_object* lp_mathlib_irreducibleComponentsEquivOfIsPreirreducibleFiber(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_irreducibleComponentsEquivOfIsPreirreducibleFiber___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_irreducibleComponentsEquivOfIsPreirreducibleFiber___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_irreducibleComponentsEquivOfIsPreirreducibleFiber(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, lean_box(0));
lean_ctor_set(x_10, 1, lean_box(0));
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_irreducibleComponentsEquivOfIsPreirreducibleFiber___lam__0(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_irreducibleComponentsEquivOfIsPreirreducibleFiber___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_irreducibleComponentsEquivOfIsPreirreducibleFiber(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_5);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Minimal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Zorn(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_ContinuousOn(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_StacksAttribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Irreducible(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Minimal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Zorn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_ContinuousOn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_StacksAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
