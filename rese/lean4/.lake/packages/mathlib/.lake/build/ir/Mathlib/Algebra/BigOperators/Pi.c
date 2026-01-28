// Lean compiler output
// Module: Mathlib.Algebra.BigOperators.Pi
// Imports: public import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Lemmas public import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise public import Mathlib.Algebra.BigOperators.GroupWithZero.Finset public import Mathlib.Algebra.Group.Action.Pi public import Mathlib.Algebra.Notation.Indicator public import Mathlib.Algebra.Ring.Pi public import Mathlib.Data.Finset.Lattice.Fold public import Mathlib.Data.Fintype.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_prod___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_single___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_mulSingle___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_evalAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_instCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_evalMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_Monoid_toMulOneClass___redArg(x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lp_mathlib_MonoidHom_mulSingle___redArg(x_1, x_2, x_4);
x_7 = lp_mathlib_MonoidHom_comp___redArg(x_3, x_6);
x_8 = lean_apply_1(x_7, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_2);
x_4 = lean_apply_1(x_1, x_2);
x_5 = lp_mathlib_Pi_evalMonoidHom___redArg(x_2);
x_6 = lp_mathlib_MonoidHom_comp___redArg(x_4, x_5);
x_7 = lean_apply_1(x_6, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__2), 3, 1);
lean_closure_set(x_5, 0, x_3);
x_6 = lp_mathlib_Finset_prod___redArg(x_1, x_2, x_5);
x_7 = lean_apply_1(x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__3(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__0), 2, 1);
lean_closure_set(x_5, 0, x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__1), 5, 2);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_5);
x_7 = lp_mathlib_MonoidHom_instCommMonoid___redArg(x_4);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__3___boxed), 4, 2);
lean_closure_set(x_8, 0, x_7);
lean_closure_set(x_8, 1, x_1);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Pi_monoidHomMulEquiv___redArg(x_2, x_3, x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lp_mathlib_AddMonoidHom_single___redArg(x_1, x_2, x_4);
x_7 = lp_mathlib_AddMonoidHom_comp___redArg(x_3, x_6);
x_8 = lean_apply_1(x_7, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_2);
x_4 = lean_apply_1(x_1, x_2);
x_5 = lp_mathlib_Pi_evalAddMonoidHom___redArg(x_2);
x_6 = lp_mathlib_AddMonoidHom_comp___redArg(x_4, x_5);
x_7 = lean_apply_1(x_6, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__2), 3, 1);
lean_closure_set(x_5, 0, x_3);
x_6 = lp_mathlib_Finset_sum___redArg(x_1, x_2, x_5);
x_7 = lean_apply_1(x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__3(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__0), 2, 1);
lean_closure_set(x_5, 0, x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__1), 5, 2);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_5);
x_7 = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(x_4);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__3___boxed), 4, 2);
lean_closure_set(x_8, 0, x_7);
lean_closure_set(x_8, 1, x_1);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Pi_addMonoidHomAddEquiv___redArg(x_2, x_3, x_5, x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Piecewise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Indicator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Indicator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
