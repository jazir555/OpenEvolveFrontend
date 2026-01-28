// Lean compiler output
// Module: Mathlib.RingTheory.DedekindDomain.Ideal.Basic
// Imports: public import Init public import Mathlib.Algebra.Algebra.Subalgebra.Pointwise public import Mathlib.RingTheory.DedekindDomain.Basic public import Mathlib.RingTheory.FractionalIdeal.Inverse public import Mathlib.RingTheory.Spectrum.Prime.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_FractionalIdeal_commSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toEuclideanDomain___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_box(0);
x_6 = lp_mathlib_Submodule_mul___redArg(x_4);
x_7 = l_npowRec___redArg(x_5, x_6, x_2, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_Field_toEuclideanDomain___redArg(x_2);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 2);
lean_dec(x_7);
x_8 = lean_ctor_get(x_4, 1);
lean_dec(x_8);
x_9 = lean_box(0);
lean_inc_ref(x_6);
x_10 = lp_mathlib_FractionalIdeal_commSemiring___redArg(x_1, x_9, x_6, x_3);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc(x_12);
lean_dec_ref(x_10);
x_13 = !lean_is_exclusive(x_11);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_14 = lean_ctor_get(x_11, 0);
x_15 = lean_ctor_get(x_11, 1);
x_16 = lean_alloc_closure((void*)(lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_16, 0, x_6);
lean_ctor_set(x_4, 2, x_16);
lean_ctor_set(x_4, 1, x_12);
lean_ctor_set(x_4, 0, x_15);
x_17 = lean_ctor_get(x_14, 1);
lean_inc(x_17);
lean_dec_ref(x_14);
lean_ctor_set(x_11, 1, x_17);
lean_ctor_set(x_11, 0, x_4);
return x_11;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_18 = lean_ctor_get(x_11, 0);
x_19 = lean_ctor_get(x_11, 1);
lean_inc(x_19);
lean_inc(x_18);
lean_dec(x_11);
x_20 = lean_alloc_closure((void*)(lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_20, 0, x_6);
lean_ctor_set(x_4, 2, x_20);
lean_ctor_set(x_4, 1, x_12);
lean_ctor_set(x_4, 0, x_19);
x_21 = lean_ctor_get(x_18, 1);
lean_inc(x_21);
lean_dec_ref(x_18);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_4);
lean_ctor_set(x_22, 1, x_21);
return x_22;
}
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_23 = lean_ctor_get(x_4, 0);
lean_inc(x_23);
lean_dec(x_4);
x_24 = lean_box(0);
lean_inc_ref(x_23);
x_25 = lp_mathlib_FractionalIdeal_commSemiring___redArg(x_1, x_24, x_23, x_3);
x_26 = lean_ctor_get(x_25, 0);
lean_inc_ref(x_26);
x_27 = lean_ctor_get(x_25, 1);
lean_inc(x_27);
lean_dec_ref(x_25);
x_28 = lean_ctor_get(x_26, 0);
lean_inc_ref(x_28);
x_29 = lean_ctor_get(x_26, 1);
lean_inc(x_29);
if (lean_is_exclusive(x_26)) {
 lean_ctor_release(x_26, 0);
 lean_ctor_release(x_26, 1);
 x_30 = x_26;
} else {
 lean_dec_ref(x_26);
 x_30 = lean_box(0);
}
x_31 = lean_alloc_closure((void*)(lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_31, 0, x_23);
x_32 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_32, 0, x_29);
lean_ctor_set(x_32, 1, x_27);
lean_ctor_set(x_32, 2, x_31);
x_33 = lean_ctor_get(x_28, 1);
lean_inc(x_33);
lean_dec_ref(x_28);
if (lean_is_scalar(x_30)) {
 x_34 = lean_alloc_ctor(0, 2, 0);
} else {
 x_34 = x_30;
}
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_33);
return x_34;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg(x_3, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FractionalIdeal_cancelCommMonoidWithZero___redArg(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_DedekindDomain_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_FractionalIdeal_Inverse(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_DedekindDomain_Ideal_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_DedekindDomain_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_FractionalIdeal_Inverse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
