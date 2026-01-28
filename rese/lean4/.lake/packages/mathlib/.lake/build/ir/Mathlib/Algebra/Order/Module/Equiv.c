// Lean compiler output
// Module: Mathlib.Algebra.Order.Module.Equiv
// Imports: public import Init public import Mathlib.Algebra.Module.Equiv.Basic public import Mathlib.Algebra.Order.Group.Equiv public import Mathlib.Algebra.Order.Module.Synonym
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
LEAN_EXPORT lean_object* lp_mathlib_toLexLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_toLexLinearEquiv___closed__1;
static lean_object* lp_mathlib_toLexLinearEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_toLexLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddEquiv_toLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ofLexLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ofLexLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
static lean_object* _init_lp_mathlib_toLexLinearEquiv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_toLexLinearEquiv___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_toLexLinearEquiv___closed__0;
x_2 = lp_mathlib_AddEquiv_toLinearEquiv___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toLexLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_toLexLinearEquiv___closed__1;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toLexLinearEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_toLexLinearEquiv(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ofLexLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_toLexLinearEquiv___closed__1;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ofLexLinearEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ofLexLinearEquiv(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Module_Synonym(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Module_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Module_Synonym(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_toLexLinearEquiv___closed__0 = _init_lp_mathlib_toLexLinearEquiv___closed__0();
lean_mark_persistent(lp_mathlib_toLexLinearEquiv___closed__0);
lp_mathlib_toLexLinearEquiv___closed__1 = _init_lp_mathlib_toLexLinearEquiv___closed__1();
lean_mark_persistent(lp_mathlib_toLexLinearEquiv___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
