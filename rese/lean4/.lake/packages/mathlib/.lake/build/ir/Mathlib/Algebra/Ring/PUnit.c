// Lean compiler output
// Module: Mathlib.Algebra.Ring.PUnit
// Imports: public import Init public import Mathlib.Algebra.Group.PUnit public import Mathlib.Algebra.Ring.Defs
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
static lean_object* lp_mathlib_PUnit_commRing___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing;
lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object*);
extern lean_object* lp_mathlib_PUnit_addCommGroup;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Int_castDef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_cancelCommMonoidWithZero;
lean_object* lp_mathlib_PUnit_commGroup___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
static lean_object* _init_lp_mathlib_PUnit_commRing___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PUnit_commGroup___lam__0), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_PUnit_commRing___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_PUnit_commRing___lam__1(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_PUnit_commRing() {
_start:
{
lean_object* x_1; uint8_t x_2; 
x_1 = lp_mathlib_PUnit_addCommGroup;
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lean_ctor_get(x_1, 3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_PUnit_commRing___lam__0___boxed), 1, 0);
x_8 = lp_mathlib_PUnit_commRing___closed__0;
x_9 = lean_box(0);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_3);
lean_ctor_set(x_10, 1, x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_PUnit_commRing___lam__1___boxed), 3, 1);
lean_closure_set(x_11, 0, x_9);
lean_inc_ref(x_7);
lean_ctor_set(x_1, 3, x_11);
lean_ctor_set(x_1, 2, x_7);
lean_ctor_set(x_1, 1, x_9);
lean_ctor_set(x_1, 0, x_10);
lean_inc(x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_7);
lean_closure_set(x_12, 2, x_4);
x_13 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_13, 0, x_1);
lean_ctor_set(x_13, 1, x_4);
lean_ctor_set(x_13, 2, x_5);
lean_ctor_set(x_13, 3, x_6);
lean_ctor_set(x_13, 4, x_12);
return x_13;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_14 = lean_ctor_get(x_1, 0);
x_15 = lean_ctor_get(x_1, 1);
x_16 = lean_ctor_get(x_1, 2);
x_17 = lean_ctor_get(x_1, 3);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_1);
x_18 = lean_alloc_closure((void*)(lp_mathlib_PUnit_commRing___lam__0___boxed), 1, 0);
x_19 = lp_mathlib_PUnit_commRing___closed__0;
x_20 = lean_box(0);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_14);
lean_ctor_set(x_21, 1, x_19);
x_22 = lean_alloc_closure((void*)(lp_mathlib_PUnit_commRing___lam__1___boxed), 3, 1);
lean_closure_set(x_22, 0, x_20);
lean_inc_ref(x_18);
x_23 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_20);
lean_ctor_set(x_23, 2, x_18);
lean_ctor_set(x_23, 3, x_22);
lean_inc(x_15);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, x_18);
lean_closure_set(x_24, 2, x_15);
x_25 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_25, 0, x_23);
lean_ctor_set(x_25, 1, x_15);
lean_ctor_set(x_25, 2, x_16);
lean_ctor_set(x_25, 3, x_17);
lean_ctor_set(x_25, 4, x_24);
return x_25;
}
}
}
static lean_object* _init_lp_mathlib_PUnit_cancelCommMonoidWithZero() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_PUnit_commRing;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_PUnit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_PUnit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PUnit_commRing___closed__0 = _init_lp_mathlib_PUnit_commRing___closed__0();
lean_mark_persistent(lp_mathlib_PUnit_commRing___closed__0);
lp_mathlib_PUnit_commRing = _init_lp_mathlib_PUnit_commRing();
lean_mark_persistent(lp_mathlib_PUnit_commRing);
lp_mathlib_PUnit_cancelCommMonoidWithZero = _init_lp_mathlib_PUnit_cancelCommMonoidWithZero();
lean_mark_persistent(lp_mathlib_PUnit_cancelCommMonoidWithZero);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
