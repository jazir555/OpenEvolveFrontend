// Lean compiler output
// Module: Mathlib.Algebra.Field.Action.ConjAct
// Imports: public import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.GroupWithZero.Action.ConjAct public import Mathlib.Algebra.GroupWithZero.Action.Defs
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
lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_distribMulAction_u2080(lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_ConjAct_mulAction_u2080___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_distribMulAction_u2080___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_distribMulAction_u2080___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_distribMulAction_u2080___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_distribMulAction_u2080___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_1);
x_3 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_2);
x_4 = lp_mathlib_ConjAct_mulAction_u2080___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_distribMulAction_u2080(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ConjAct_distribMulAction_u2080___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_distribMulAction_u2080___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ConjAct_distribMulAction_u2080(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_distribMulAction_u2080___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ConjAct_distribMulAction_u2080___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_ConjAct(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_Action_ConjAct(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_ConjAct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
