// Lean compiler output
// Module: Mathlib.Algebra.Ring.AddAut
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Action.Basic public import Mathlib.Algebra.GroupWithZero.Action.Units public import Mathlib.Algebra.Group.Units.Opposite public import Mathlib.Algebra.Module.Opposite
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
lean_object* lp_mathlib_DistribMulAction_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DistribMulAction_toAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Units_instSMul___redArg(lean_object*);
lean_object* lp_mathlib_Units_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulLeft(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Units_opEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulLeft___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulRight(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toOppositeModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulLeft___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_Units_instDivInvMonoid___redArg(x_3);
x_5 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_6 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lp_mathlib_Semiring_toModule___redArg(x_1);
x_9 = lp_mathlib_Units_instSMul___redArg(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_toAddEquiv___boxed), 6, 5);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, x_4);
lean_closure_set(x_10, 3, x_7);
lean_closure_set(x_10, 4, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulLeft(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddAut_mulLeft___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulRight___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_inc_ref(x_4);
x_5 = lp_mathlib_MulOpposite_instMonoid___redArg(x_4);
x_6 = lp_mathlib_Units_instDivInvMonoid___redArg(x_5);
x_7 = lp_mathlib_Units_opEquiv(lean_box(0), x_4);
lean_dec_ref(x_4);
x_8 = lp_mathlib_Equiv_symm___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_Semiring_toOppositeModule___redArg(x_1);
x_11 = lp_mathlib_Units_instSMul___redArg(x_10);
x_12 = lean_apply_1(x_9, x_2);
x_13 = lp_mathlib_DistribMulAction_toAddEquiv___redArg(x_6, x_11, x_12);
lean_dec_ref(x_6);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulRight(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddAut_mulRight___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_AddAut(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
