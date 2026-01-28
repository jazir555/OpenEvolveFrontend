// Lean compiler output
// Module: Mathlib.Algebra.Module.Opposite
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Action.Opposite public import Mathlib.Algebra.Module.Defs public import Mathlib.Algebra.Ring.Opposite
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
lean_object* lp_mathlib_MonoidWithZero_toOppositeMulActionWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instModule___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toOppositeModule(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toOppositeModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toOppositeModule___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_1);
x_3 = lp_mathlib_MonoidWithZero_toOppositeMulActionWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toOppositeModule(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Semiring_toOppositeModule___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instModule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MulOpposite_instSMul___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instModule___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulOpposite_instSMul___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instModule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MulOpposite_instModule(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
