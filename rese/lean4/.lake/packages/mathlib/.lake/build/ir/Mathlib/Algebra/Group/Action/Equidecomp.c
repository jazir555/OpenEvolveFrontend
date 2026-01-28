// Lean compiler output
// Module: Mathlib.Algebra.Group.Action.Equidecomp
// Imports: public import Init public import Mathlib.Algebra.Group.Action.Defs public import Mathlib.Logic.Equiv.PartialEquiv public import Mathlib.Algebra.Group.Pointwise.Finset.Basic
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
lean_object* lp_mathlib_PartialEquiv_IsImage_restr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_instCoeFunForall___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_restr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_instCoeFunForall___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Equidecomp_refl___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_refl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_refl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_instCoeFunForall(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_restr___redArg(lean_object*);
lean_object* lp_mathlib_PartialEquiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_restr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_instCoeFunForall___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_instCoeFunForall(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equidecomp_instCoeFunForall___lam__0), 2, 0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_instCoeFunForall___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equidecomp_instCoeFunForall(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_restr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_PartialEquiv_IsImage_restr___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_restr___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_PartialEquiv_IsImage_restr___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_restr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equidecomp_restr(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_3);
return x_6;
}
}
static lean_object* _init_lp_mathlib_Equidecomp_refl___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_PartialEquiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_refl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equidecomp_refl___closed__0;
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equidecomp_refl___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equidecomp_refl(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_PartialEquiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Equidecomp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_PartialEquiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equidecomp_refl___closed__0 = _init_lp_mathlib_Equidecomp_refl___closed__0();
lean_mark_persistent(lp_mathlib_Equidecomp_refl___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
