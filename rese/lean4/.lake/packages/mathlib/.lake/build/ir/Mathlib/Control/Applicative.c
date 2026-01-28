// Lean compiler output
// Module: Mathlib.Control.Applicative
// Imports: public import Init public import Mathlib.Algebra.Group.Defs public import Mathlib.Control.Functor public import Mathlib.Control.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeAddConstOfZeroOfAdd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeAddConstOfZeroOfAdd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Functor_Const_functor(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_box(0);
x_7 = lean_apply_1(x_5, x_6);
x_8 = lean_apply_2(x_1, x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_apply_4(x_1, lean_box(0), lean_box(0), x_4, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Functor_Const_functor(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__1), 5, 1);
lean_closure_set(x_4, 0, x_2);
lean_inc_ref(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__2), 5, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0;
lean_inc_ref(x_5);
x_7 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_4);
lean_ctor_set(x_7, 3, x_5);
lean_ctor_set(x_7, 4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instApplicativeConstOfOneOfMul___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeAddConstOfZeroOfAdd___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__1), 5, 1);
lean_closure_set(x_4, 0, x_2);
lean_inc_ref(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__2), 5, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0;
lean_inc_ref(x_5);
x_7 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_4);
lean_ctor_set(x_7, 3, x_5);
lean_ctor_set(x_7, 4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeAddConstOfZeroOfAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instApplicativeAddConstOfZeroOfAdd___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_Applicative(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0 = _init_lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0();
lean_mark_persistent(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
