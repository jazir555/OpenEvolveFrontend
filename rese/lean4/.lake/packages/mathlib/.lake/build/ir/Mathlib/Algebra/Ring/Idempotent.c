// Lean compiler output
// Module: Mathlib.Algebra.Ring.Idempotent
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Idempotent public import Mathlib.Algebra.Ring.Defs public import Mathlib.Order.Notation public import Mathlib.Tactic.Convert public import Mathlib.Algebra.Group.Torsion
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
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instHasComplSubtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instHasComplSubtype___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instHasComplSubtype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instHasComplSubtype___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instHasComplSubtype___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(x_1);
x_3 = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(x_2);
lean_dec_ref(x_2);
x_4 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_3);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
lean_dec_ref(x_3);
x_6 = lean_ctor_get(x_4, 2);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lean_ctor_get(x_5, 2);
lean_inc(x_7);
lean_dec_ref(x_5);
x_8 = lean_alloc_closure((void*)(lp_mathlib_IsIdempotentElem_instHasComplSubtype___redArg___lam__0), 3, 2);
lean_closure_set(x_8, 0, x_6);
lean_closure_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instHasComplSubtype(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_IsIdempotentElem_instHasComplSubtype___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Idempotent(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Idempotent(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
