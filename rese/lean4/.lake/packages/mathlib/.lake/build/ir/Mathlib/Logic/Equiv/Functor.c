// Lean compiler output
// Module: Mathlib.Logic.Equiv.Functor
// Imports: public import Init public import Mathlib.Control.Bifunctor public import Mathlib.Logic.Equiv.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_mapEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_mapEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapEquiv___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_mapEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapEquiv___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
lean_dec(x_5);
lean_inc_ref(x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Functor_mapEquiv___redArg___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_2);
lean_inc(x_4);
x_7 = lean_apply_3(x_4, lean_box(0), lean_box(0), x_6);
x_8 = lp_mathlib_Equiv_symm___redArg(x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Functor_mapEquiv___redArg___lam__1), 2, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lean_apply_3(x_4, lean_box(0), lean_box(0), x_9);
lean_ctor_set(x_1, 1, x_10);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_11 = lean_ctor_get(x_1, 0);
lean_inc(x_11);
lean_dec(x_1);
lean_inc_ref(x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Functor_mapEquiv___redArg___lam__0), 2, 1);
lean_closure_set(x_12, 0, x_2);
lean_inc(x_11);
x_13 = lean_apply_3(x_11, lean_box(0), lean_box(0), x_12);
x_14 = lp_mathlib_Equiv_symm___redArg(x_2);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Functor_mapEquiv___redArg___lam__1), 2, 1);
lean_closure_set(x_15, 0, x_14);
x_16 = lean_apply_3(x_11, lean_box(0), lean_box(0), x_15);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_13);
lean_ctor_set(x_17, 1, x_16);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Functor_mapEquiv___redArg(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_mapEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_mapEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Functor_mapEquiv___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_2);
lean_inc_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Bifunctor_mapEquiv___redArg___lam__1), 2, 1);
lean_closure_set(x_5, 0, x_3);
lean_inc(x_1);
x_6 = lean_apply_6(x_1, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_4, x_5);
x_7 = lp_mathlib_Equiv_symm___redArg(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Functor_mapEquiv___redArg___lam__1), 2, 1);
lean_closure_set(x_8, 0, x_7);
x_9 = lp_mathlib_Equiv_symm___redArg(x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Functor_mapEquiv___redArg___lam__1), 2, 1);
lean_closure_set(x_10, 0, x_9);
x_11 = lean_apply_6(x_1, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_8, x_10);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_6);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_mapEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Bifunctor_mapEquiv___redArg(x_6, x_8, x_9);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Bifunctor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Functor(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Bifunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
