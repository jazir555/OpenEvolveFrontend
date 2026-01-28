// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.TransferInstance
// Imports: public import Init public import Mathlib.Algebra.Group.TransferInstance public import Mathlib.Algebra.GroupWithZero.InjSurj
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoidWithZero___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoidWithZero___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoidWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroupWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_zero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroupWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroOneClass___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroOneClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc(x_5);
x_8 = lean_apply_1(x_5, x_3);
x_9 = lean_apply_1(x_5, x_4);
x_10 = lean_apply_2(x_2, x_8, x_9);
x_11 = lean_apply_1(x_7, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroupWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(x_4);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_5, 0);
x_8 = lean_ctor_get(x_5, 1);
lean_inc_ref(x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0), 4, 2);
lean_closure_set(x_9, 0, x_3);
lean_closure_set(x_9, 1, x_7);
x_10 = lp_mathlib_Equiv_zero___redArg(x_3, x_8);
lean_ctor_set(x_5, 1, x_10);
lean_ctor_set(x_5, 0, x_9);
return x_5;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_ctor_get(x_5, 0);
x_12 = lean_ctor_get(x_5, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_5);
lean_inc_ref(x_3);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0), 4, 2);
lean_closure_set(x_13, 0, x_3);
lean_closure_set(x_13, 1, x_11);
x_14 = lp_mathlib_Equiv_zero___redArg(x_3, x_12);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroupWithZero___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(x_2);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0), 4, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_5);
x_8 = lp_mathlib_Equiv_zero___redArg(x_1, x_6);
lean_ctor_set(x_3, 1, x_8);
lean_ctor_set(x_3, 0, x_7);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_3, 0);
x_10 = lean_ctor_get(x_3, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_3);
lean_inc_ref(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0), 4, 2);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_9);
x_12 = lp_mathlib_Equiv_zero___redArg(x_1, x_10);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroClass(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_3);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0), 4, 2);
lean_closure_set(x_8, 0, x_3);
lean_closure_set(x_8, 1, x_6);
x_9 = lp_mathlib_Equiv_zero___redArg(x_3, x_7);
lean_ctor_set(x_4, 1, x_9);
lean_ctor_set(x_4, 0, x_8);
return x_4;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_4, 0);
x_11 = lean_ctor_get(x_4, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_4);
lean_inc_ref(x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0), 4, 2);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_10);
x_13 = lp_mathlib_Equiv_zero___redArg(x_3, x_11);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroClass___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0), 4, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_4);
x_7 = lp_mathlib_Equiv_zero___redArg(x_1, x_5);
lean_ctor_set(x_2, 1, x_7);
lean_ctor_set(x_2, 0, x_6);
return x_2;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_2, 0);
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_2);
lean_inc_ref(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Equiv_semigroupWithZero___redArg___lam__0), 4, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_8);
x_11 = lp_mathlib_Equiv_zero___redArg(x_1, x_9);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
lean_inc(x_6);
x_7 = lean_apply_1(x_6, x_4);
x_8 = lean_apply_1(x_6, x_5);
x_9 = lean_apply_2(x_2, x_7, x_8);
x_10 = lean_apply_1(x_3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroOneClass(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc_ref(x_4);
x_5 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_4);
x_6 = !lean_is_exclusive(x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_7 = lean_ctor_get(x_4, 0);
x_8 = lean_ctor_get(x_4, 1);
lean_dec(x_8);
x_9 = lean_ctor_get(x_5, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_5, 1);
lean_inc(x_10);
lean_dec_ref(x_5);
x_11 = !lean_is_exclusive(x_7);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_12 = lean_ctor_get(x_7, 0);
x_13 = lean_ctor_get(x_7, 1);
lean_dec(x_13);
lean_inc_ref(x_3);
x_14 = lp_mathlib_Equiv_symm___redArg(x_3);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_3);
x_16 = lp_mathlib_Equiv_zero___redArg(x_3, x_10);
lean_inc(x_15);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_17, 0, x_3);
lean_closure_set(x_17, 1, x_9);
lean_closure_set(x_17, 2, x_15);
x_18 = lean_apply_1(x_15, x_12);
lean_ctor_set(x_7, 1, x_17);
lean_ctor_set(x_7, 0, x_18);
lean_ctor_set(x_4, 1, x_16);
return x_4;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_19 = lean_ctor_get(x_7, 0);
lean_inc(x_19);
lean_dec(x_7);
lean_inc_ref(x_3);
x_20 = lp_mathlib_Equiv_symm___redArg(x_3);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_inc_ref(x_3);
x_22 = lp_mathlib_Equiv_zero___redArg(x_3, x_10);
lean_inc(x_21);
x_23 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_23, 0, x_3);
lean_closure_set(x_23, 1, x_9);
lean_closure_set(x_23, 2, x_21);
x_24 = lean_apply_1(x_21, x_19);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_24);
lean_ctor_set(x_25, 1, x_23);
lean_ctor_set(x_4, 1, x_22);
lean_ctor_set(x_4, 0, x_25);
return x_4;
}
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_26 = lean_ctor_get(x_4, 0);
lean_inc(x_26);
lean_dec(x_4);
x_27 = lean_ctor_get(x_5, 0);
lean_inc(x_27);
x_28 = lean_ctor_get(x_5, 1);
lean_inc(x_28);
lean_dec_ref(x_5);
x_29 = lean_ctor_get(x_26, 0);
lean_inc(x_29);
if (lean_is_exclusive(x_26)) {
 lean_ctor_release(x_26, 0);
 lean_ctor_release(x_26, 1);
 x_30 = x_26;
} else {
 lean_dec_ref(x_26);
 x_30 = lean_box(0);
}
lean_inc_ref(x_3);
x_31 = lp_mathlib_Equiv_symm___redArg(x_3);
x_32 = lean_ctor_get(x_31, 0);
lean_inc(x_32);
lean_dec_ref(x_31);
lean_inc_ref(x_3);
x_33 = lp_mathlib_Equiv_zero___redArg(x_3, x_28);
lean_inc(x_32);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_34, 0, x_3);
lean_closure_set(x_34, 1, x_27);
lean_closure_set(x_34, 2, x_32);
x_35 = lean_apply_1(x_32, x_29);
if (lean_is_scalar(x_30)) {
 x_36 = lean_alloc_ctor(0, 2, 0);
} else {
 x_36 = x_30;
}
lean_ctor_set(x_36, 0, x_35);
lean_ctor_set(x_36, 1, x_34);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_36);
lean_ctor_set(x_37, 1, x_33);
return x_37;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulZeroOneClass___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_2);
x_3 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_2);
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_2, 1);
lean_dec(x_6);
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_3, 1);
lean_inc(x_8);
lean_dec_ref(x_3);
x_9 = !lean_is_exclusive(x_5);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_10 = lean_ctor_get(x_5, 0);
x_11 = lean_ctor_get(x_5, 1);
lean_dec(x_11);
lean_inc_ref(x_1);
x_12 = lp_mathlib_Equiv_symm___redArg(x_1);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc_ref(x_1);
x_14 = lp_mathlib_Equiv_zero___redArg(x_1, x_8);
lean_inc(x_13);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_15, 0, x_1);
lean_closure_set(x_15, 1, x_7);
lean_closure_set(x_15, 2, x_13);
x_16 = lean_apply_1(x_13, x_10);
lean_ctor_set(x_5, 1, x_15);
lean_ctor_set(x_5, 0, x_16);
lean_ctor_set(x_2, 1, x_14);
return x_2;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_17 = lean_ctor_get(x_5, 0);
lean_inc(x_17);
lean_dec(x_5);
lean_inc_ref(x_1);
x_18 = lp_mathlib_Equiv_symm___redArg(x_1);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
lean_inc_ref(x_1);
x_20 = lp_mathlib_Equiv_zero___redArg(x_1, x_8);
lean_inc(x_19);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_21, 0, x_1);
lean_closure_set(x_21, 1, x_7);
lean_closure_set(x_21, 2, x_19);
x_22 = lean_apply_1(x_19, x_17);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_21);
lean_ctor_set(x_2, 1, x_20);
lean_ctor_set(x_2, 0, x_23);
return x_2;
}
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_24 = lean_ctor_get(x_2, 0);
lean_inc(x_24);
lean_dec(x_2);
x_25 = lean_ctor_get(x_3, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_3, 1);
lean_inc(x_26);
lean_dec_ref(x_3);
x_27 = lean_ctor_get(x_24, 0);
lean_inc(x_27);
if (lean_is_exclusive(x_24)) {
 lean_ctor_release(x_24, 0);
 lean_ctor_release(x_24, 1);
 x_28 = x_24;
} else {
 lean_dec_ref(x_24);
 x_28 = lean_box(0);
}
lean_inc_ref(x_1);
x_29 = lp_mathlib_Equiv_symm___redArg(x_1);
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
lean_dec_ref(x_29);
lean_inc_ref(x_1);
x_31 = lp_mathlib_Equiv_zero___redArg(x_1, x_26);
lean_inc(x_30);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_32, 0, x_1);
lean_closure_set(x_32, 1, x_25);
lean_closure_set(x_32, 2, x_30);
x_33 = lean_apply_1(x_30, x_27);
if (lean_is_scalar(x_28)) {
 x_34 = lean_alloc_ctor(0, 2, 0);
} else {
 x_34 = x_28;
}
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_32);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_34);
lean_ctor_set(x_35, 1, x_31);
return x_35;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoidWithZero___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 2);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_apply_1(x_6, x_5);
x_9 = lean_apply_2(x_7, x_4, x_8);
x_10 = lean_apply_1(x_3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
lean_inc_ref(x_4);
x_5 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_4);
lean_inc_ref(x_5);
x_6 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_5);
x_7 = !lean_is_exclusive(x_5);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_8 = lean_ctor_get(x_5, 0);
x_9 = lean_ctor_get(x_5, 1);
lean_dec(x_9);
x_10 = lean_ctor_get(x_6, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_6, 1);
lean_inc(x_11);
lean_dec_ref(x_6);
x_12 = !lean_is_exclusive(x_8);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; 
x_13 = lean_ctor_get(x_8, 0);
x_14 = lean_ctor_get(x_8, 1);
lean_dec(x_14);
lean_inc_ref(x_3);
x_15 = lp_mathlib_Equiv_symm___redArg(x_3);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc_ref(x_3);
x_17 = lp_mathlib_Equiv_zero___redArg(x_3, x_11);
lean_inc(x_16);
lean_inc_ref(x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_18, 0, x_3);
lean_closure_set(x_18, 1, x_10);
lean_closure_set(x_18, 2, x_16);
lean_inc(x_16);
x_19 = lean_apply_1(x_16, x_13);
lean_inc(x_19);
lean_ctor_set(x_8, 1, x_18);
lean_ctor_set(x_8, 0, x_19);
lean_ctor_set(x_5, 1, x_17);
x_20 = !lean_is_exclusive(x_4);
if (x_20 == 0)
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_21 = lean_ctor_get(x_4, 0);
x_22 = lean_ctor_get(x_4, 1);
lean_dec(x_22);
x_23 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_5);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
x_25 = lean_ctor_get(x_23, 1);
lean_inc(x_25);
lean_dec_ref(x_23);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_26, 0, x_3);
lean_closure_set(x_26, 1, x_21);
lean_closure_set(x_26, 2, x_16);
x_27 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_27, 0, x_24);
lean_ctor_set(x_27, 1, x_19);
lean_ctor_set(x_27, 2, x_26);
lean_ctor_set(x_4, 1, x_25);
lean_ctor_set(x_4, 0, x_27);
return x_4;
}
else
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_28 = lean_ctor_get(x_4, 0);
lean_inc(x_28);
lean_dec(x_4);
x_29 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_5);
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
x_31 = lean_ctor_get(x_29, 1);
lean_inc(x_31);
lean_dec_ref(x_29);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_32, 0, x_3);
lean_closure_set(x_32, 1, x_28);
lean_closure_set(x_32, 2, x_16);
x_33 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_33, 0, x_30);
lean_ctor_set(x_33, 1, x_19);
lean_ctor_set(x_33, 2, x_32);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_31);
return x_34;
}
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_35 = lean_ctor_get(x_8, 0);
lean_inc(x_35);
lean_dec(x_8);
lean_inc_ref(x_3);
x_36 = lp_mathlib_Equiv_symm___redArg(x_3);
x_37 = lean_ctor_get(x_36, 0);
lean_inc(x_37);
lean_dec_ref(x_36);
lean_inc_ref(x_3);
x_38 = lp_mathlib_Equiv_zero___redArg(x_3, x_11);
lean_inc(x_37);
lean_inc_ref(x_3);
x_39 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_39, 0, x_3);
lean_closure_set(x_39, 1, x_10);
lean_closure_set(x_39, 2, x_37);
lean_inc(x_37);
x_40 = lean_apply_1(x_37, x_35);
lean_inc(x_40);
x_41 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_39);
lean_ctor_set(x_5, 1, x_38);
lean_ctor_set(x_5, 0, x_41);
x_42 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_42);
if (lean_is_exclusive(x_4)) {
 lean_ctor_release(x_4, 0);
 lean_ctor_release(x_4, 1);
 x_43 = x_4;
} else {
 lean_dec_ref(x_4);
 x_43 = lean_box(0);
}
x_44 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_5);
x_45 = lean_ctor_get(x_44, 0);
lean_inc(x_45);
x_46 = lean_ctor_get(x_44, 1);
lean_inc(x_46);
lean_dec_ref(x_44);
x_47 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_47, 0, x_3);
lean_closure_set(x_47, 1, x_42);
lean_closure_set(x_47, 2, x_37);
x_48 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_48, 0, x_45);
lean_ctor_set(x_48, 1, x_40);
lean_ctor_set(x_48, 2, x_47);
if (lean_is_scalar(x_43)) {
 x_49 = lean_alloc_ctor(0, 2, 0);
} else {
 x_49 = x_43;
}
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_46);
return x_49;
}
}
else
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; 
x_50 = lean_ctor_get(x_5, 0);
lean_inc(x_50);
lean_dec(x_5);
x_51 = lean_ctor_get(x_6, 0);
lean_inc(x_51);
x_52 = lean_ctor_get(x_6, 1);
lean_inc(x_52);
lean_dec_ref(x_6);
x_53 = lean_ctor_get(x_50, 0);
lean_inc(x_53);
if (lean_is_exclusive(x_50)) {
 lean_ctor_release(x_50, 0);
 lean_ctor_release(x_50, 1);
 x_54 = x_50;
} else {
 lean_dec_ref(x_50);
 x_54 = lean_box(0);
}
lean_inc_ref(x_3);
x_55 = lp_mathlib_Equiv_symm___redArg(x_3);
x_56 = lean_ctor_get(x_55, 0);
lean_inc(x_56);
lean_dec_ref(x_55);
lean_inc_ref(x_3);
x_57 = lp_mathlib_Equiv_zero___redArg(x_3, x_52);
lean_inc(x_56);
lean_inc_ref(x_3);
x_58 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_58, 0, x_3);
lean_closure_set(x_58, 1, x_51);
lean_closure_set(x_58, 2, x_56);
lean_inc(x_56);
x_59 = lean_apply_1(x_56, x_53);
lean_inc(x_59);
if (lean_is_scalar(x_54)) {
 x_60 = lean_alloc_ctor(0, 2, 0);
} else {
 x_60 = x_54;
}
lean_ctor_set(x_60, 0, x_59);
lean_ctor_set(x_60, 1, x_58);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_60);
lean_ctor_set(x_61, 1, x_57);
x_62 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_62);
if (lean_is_exclusive(x_4)) {
 lean_ctor_release(x_4, 0);
 lean_ctor_release(x_4, 1);
 x_63 = x_4;
} else {
 lean_dec_ref(x_4);
 x_63 = lean_box(0);
}
x_64 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_61);
x_65 = lean_ctor_get(x_64, 0);
lean_inc(x_65);
x_66 = lean_ctor_get(x_64, 1);
lean_inc(x_66);
lean_dec_ref(x_64);
x_67 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_67, 0, x_3);
lean_closure_set(x_67, 1, x_62);
lean_closure_set(x_67, 2, x_56);
x_68 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_68, 0, x_65);
lean_ctor_set(x_68, 1, x_59);
lean_ctor_set(x_68, 2, x_67);
if (lean_is_scalar(x_63)) {
 x_69 = lean_alloc_ctor(0, 2, 0);
} else {
 x_69 = x_63;
}
lean_ctor_set(x_69, 0, x_68);
lean_ctor_set(x_69, 1, x_66);
return x_69;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoidWithZero___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
lean_inc_ref(x_2);
x_3 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_2);
lean_inc_ref(x_3);
x_4 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_3);
x_5 = !lean_is_exclusive(x_3);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_6 = lean_ctor_get(x_3, 0);
x_7 = lean_ctor_get(x_3, 1);
lean_dec(x_7);
x_8 = lean_ctor_get(x_4, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_4, 1);
lean_inc(x_9);
lean_dec_ref(x_4);
x_10 = !lean_is_exclusive(x_6);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; uint8_t x_18; 
x_11 = lean_ctor_get(x_6, 0);
x_12 = lean_ctor_get(x_6, 1);
lean_dec(x_12);
lean_inc_ref(x_1);
x_13 = lp_mathlib_Equiv_symm___redArg(x_1);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc_ref(x_1);
x_15 = lp_mathlib_Equiv_zero___redArg(x_1, x_9);
lean_inc(x_14);
lean_inc_ref(x_1);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_16, 0, x_1);
lean_closure_set(x_16, 1, x_8);
lean_closure_set(x_16, 2, x_14);
lean_inc(x_14);
x_17 = lean_apply_1(x_14, x_11);
lean_inc(x_17);
lean_ctor_set(x_6, 1, x_16);
lean_ctor_set(x_6, 0, x_17);
lean_ctor_set(x_3, 1, x_15);
x_18 = !lean_is_exclusive(x_2);
if (x_18 == 0)
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_19 = lean_ctor_get(x_2, 0);
x_20 = lean_ctor_get(x_2, 1);
lean_dec(x_20);
x_21 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_3);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
x_23 = lean_ctor_get(x_21, 1);
lean_inc(x_23);
lean_dec_ref(x_21);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_24, 0, x_1);
lean_closure_set(x_24, 1, x_19);
lean_closure_set(x_24, 2, x_14);
x_25 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_25, 0, x_22);
lean_ctor_set(x_25, 1, x_17);
lean_ctor_set(x_25, 2, x_24);
lean_ctor_set(x_2, 1, x_23);
lean_ctor_set(x_2, 0, x_25);
return x_2;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_26 = lean_ctor_get(x_2, 0);
lean_inc(x_26);
lean_dec(x_2);
x_27 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_3);
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
x_29 = lean_ctor_get(x_27, 1);
lean_inc(x_29);
lean_dec_ref(x_27);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_30, 0, x_1);
lean_closure_set(x_30, 1, x_26);
lean_closure_set(x_30, 2, x_14);
x_31 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_31, 0, x_28);
lean_ctor_set(x_31, 1, x_17);
lean_ctor_set(x_31, 2, x_30);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_29);
return x_32;
}
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; 
x_33 = lean_ctor_get(x_6, 0);
lean_inc(x_33);
lean_dec(x_6);
lean_inc_ref(x_1);
x_34 = lp_mathlib_Equiv_symm___redArg(x_1);
x_35 = lean_ctor_get(x_34, 0);
lean_inc(x_35);
lean_dec_ref(x_34);
lean_inc_ref(x_1);
x_36 = lp_mathlib_Equiv_zero___redArg(x_1, x_9);
lean_inc(x_35);
lean_inc_ref(x_1);
x_37 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_37, 0, x_1);
lean_closure_set(x_37, 1, x_8);
lean_closure_set(x_37, 2, x_35);
lean_inc(x_35);
x_38 = lean_apply_1(x_35, x_33);
lean_inc(x_38);
x_39 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_39, 0, x_38);
lean_ctor_set(x_39, 1, x_37);
lean_ctor_set(x_3, 1, x_36);
lean_ctor_set(x_3, 0, x_39);
x_40 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_40);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_41 = x_2;
} else {
 lean_dec_ref(x_2);
 x_41 = lean_box(0);
}
x_42 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_3);
x_43 = lean_ctor_get(x_42, 0);
lean_inc(x_43);
x_44 = lean_ctor_get(x_42, 1);
lean_inc(x_44);
lean_dec_ref(x_42);
x_45 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_45, 0, x_1);
lean_closure_set(x_45, 1, x_40);
lean_closure_set(x_45, 2, x_35);
x_46 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_46, 0, x_43);
lean_ctor_set(x_46, 1, x_38);
lean_ctor_set(x_46, 2, x_45);
if (lean_is_scalar(x_41)) {
 x_47 = lean_alloc_ctor(0, 2, 0);
} else {
 x_47 = x_41;
}
lean_ctor_set(x_47, 0, x_46);
lean_ctor_set(x_47, 1, x_44);
return x_47;
}
}
else
{
lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; 
x_48 = lean_ctor_get(x_3, 0);
lean_inc(x_48);
lean_dec(x_3);
x_49 = lean_ctor_get(x_4, 0);
lean_inc(x_49);
x_50 = lean_ctor_get(x_4, 1);
lean_inc(x_50);
lean_dec_ref(x_4);
x_51 = lean_ctor_get(x_48, 0);
lean_inc(x_51);
if (lean_is_exclusive(x_48)) {
 lean_ctor_release(x_48, 0);
 lean_ctor_release(x_48, 1);
 x_52 = x_48;
} else {
 lean_dec_ref(x_48);
 x_52 = lean_box(0);
}
lean_inc_ref(x_1);
x_53 = lp_mathlib_Equiv_symm___redArg(x_1);
x_54 = lean_ctor_get(x_53, 0);
lean_inc(x_54);
lean_dec_ref(x_53);
lean_inc_ref(x_1);
x_55 = lp_mathlib_Equiv_zero___redArg(x_1, x_50);
lean_inc(x_54);
lean_inc_ref(x_1);
x_56 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_56, 0, x_1);
lean_closure_set(x_56, 1, x_49);
lean_closure_set(x_56, 2, x_54);
lean_inc(x_54);
x_57 = lean_apply_1(x_54, x_51);
lean_inc(x_57);
if (lean_is_scalar(x_52)) {
 x_58 = lean_alloc_ctor(0, 2, 0);
} else {
 x_58 = x_52;
}
lean_ctor_set(x_58, 0, x_57);
lean_ctor_set(x_58, 1, x_56);
x_59 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_59, 0, x_58);
lean_ctor_set(x_59, 1, x_55);
x_60 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_60);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_61 = x_2;
} else {
 lean_dec_ref(x_2);
 x_61 = lean_box(0);
}
x_62 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_59);
x_63 = lean_ctor_get(x_62, 0);
lean_inc(x_63);
x_64 = lean_ctor_get(x_62, 1);
lean_inc(x_64);
lean_dec_ref(x_62);
x_65 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_65, 0, x_1);
lean_closure_set(x_65, 1, x_60);
lean_closure_set(x_65, 2, x_54);
x_66 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_66, 0, x_63);
lean_ctor_set(x_66, 1, x_57);
lean_ctor_set(x_66, 2, x_65);
if (lean_is_scalar(x_61)) {
 x_67 = lean_alloc_ctor(0, 2, 0);
} else {
 x_67 = x_61;
}
lean_ctor_set(x_67, 0, x_66);
lean_ctor_set(x_67, 1, x_64);
return x_67;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_5 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_4);
lean_inc_ref(x_5);
x_6 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_5);
lean_inc_ref(x_6);
x_7 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_6);
x_8 = !lean_is_exclusive(x_6);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
lean_dec(x_10);
x_11 = lean_ctor_get(x_7, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_7, 1);
lean_inc(x_12);
lean_dec_ref(x_7);
x_13 = !lean_is_exclusive(x_9);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_14 = lean_ctor_get(x_9, 0);
x_15 = lean_ctor_get(x_9, 1);
lean_dec(x_15);
lean_inc_ref(x_3);
x_16 = lp_mathlib_Equiv_symm___redArg(x_3);
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
lean_inc_ref(x_3);
x_18 = lp_mathlib_Equiv_zero___redArg(x_3, x_12);
lean_inc(x_17);
lean_inc_ref(x_3);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_19, 0, x_3);
lean_closure_set(x_19, 1, x_11);
lean_closure_set(x_19, 2, x_17);
lean_inc(x_17);
x_20 = lean_apply_1(x_17, x_14);
lean_inc(x_20);
lean_ctor_set(x_9, 1, x_19);
lean_ctor_set(x_9, 0, x_20);
lean_ctor_set(x_6, 1, x_18);
x_21 = !lean_is_exclusive(x_5);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; 
x_22 = lean_ctor_get(x_5, 0);
x_23 = lean_ctor_get(x_5, 1);
lean_dec(x_23);
x_24 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_6);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_24, 1);
lean_inc(x_26);
lean_dec_ref(x_24);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_27, 0, x_3);
lean_closure_set(x_27, 1, x_22);
lean_closure_set(x_27, 2, x_17);
lean_inc_ref(x_27);
x_28 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_28, 0, x_25);
lean_ctor_set(x_28, 1, x_20);
lean_ctor_set(x_28, 2, x_27);
lean_ctor_set(x_5, 1, x_26);
lean_ctor_set(x_5, 0, x_28);
x_29 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_5);
lean_inc_ref(x_29);
x_30 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_29);
x_31 = lean_ctor_get(x_29, 0);
lean_inc_ref(x_31);
lean_dec_ref(x_29);
x_32 = lean_ctor_get(x_30, 0);
lean_inc(x_32);
x_33 = lean_ctor_get(x_30, 1);
lean_inc(x_33);
lean_dec_ref(x_30);
x_34 = !lean_is_exclusive(x_31);
if (x_34 == 0)
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_35 = lean_ctor_get(x_31, 0);
x_36 = lean_ctor_get(x_31, 1);
lean_dec(x_36);
x_37 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_37, 0, x_27);
x_38 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_38, 0, x_32);
lean_ctor_set(x_38, 1, x_35);
lean_ctor_set(x_38, 2, x_37);
lean_ctor_set(x_31, 1, x_33);
lean_ctor_set(x_31, 0, x_38);
return x_31;
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_39 = lean_ctor_get(x_31, 0);
lean_inc(x_39);
lean_dec(x_31);
x_40 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_40, 0, x_27);
x_41 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_41, 0, x_32);
lean_ctor_set(x_41, 1, x_39);
lean_ctor_set(x_41, 2, x_40);
x_42 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_42, 0, x_41);
lean_ctor_set(x_42, 1, x_33);
return x_42;
}
}
else
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_43 = lean_ctor_get(x_5, 0);
lean_inc(x_43);
lean_dec(x_5);
x_44 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_6);
x_45 = lean_ctor_get(x_44, 0);
lean_inc(x_45);
x_46 = lean_ctor_get(x_44, 1);
lean_inc(x_46);
lean_dec_ref(x_44);
x_47 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_47, 0, x_3);
lean_closure_set(x_47, 1, x_43);
lean_closure_set(x_47, 2, x_17);
lean_inc_ref(x_47);
x_48 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_48, 0, x_45);
lean_ctor_set(x_48, 1, x_20);
lean_ctor_set(x_48, 2, x_47);
x_49 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_46);
x_50 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_49);
lean_inc_ref(x_50);
x_51 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_50);
x_52 = lean_ctor_get(x_50, 0);
lean_inc_ref(x_52);
lean_dec_ref(x_50);
x_53 = lean_ctor_get(x_51, 0);
lean_inc(x_53);
x_54 = lean_ctor_get(x_51, 1);
lean_inc(x_54);
lean_dec_ref(x_51);
x_55 = lean_ctor_get(x_52, 0);
lean_inc(x_55);
if (lean_is_exclusive(x_52)) {
 lean_ctor_release(x_52, 0);
 lean_ctor_release(x_52, 1);
 x_56 = x_52;
} else {
 lean_dec_ref(x_52);
 x_56 = lean_box(0);
}
x_57 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_57, 0, x_47);
x_58 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_58, 0, x_53);
lean_ctor_set(x_58, 1, x_55);
lean_ctor_set(x_58, 2, x_57);
if (lean_is_scalar(x_56)) {
 x_59 = lean_alloc_ctor(0, 2, 0);
} else {
 x_59 = x_56;
}
lean_ctor_set(x_59, 0, x_58);
lean_ctor_set(x_59, 1, x_54);
return x_59;
}
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; 
x_60 = lean_ctor_get(x_9, 0);
lean_inc(x_60);
lean_dec(x_9);
lean_inc_ref(x_3);
x_61 = lp_mathlib_Equiv_symm___redArg(x_3);
x_62 = lean_ctor_get(x_61, 0);
lean_inc(x_62);
lean_dec_ref(x_61);
lean_inc_ref(x_3);
x_63 = lp_mathlib_Equiv_zero___redArg(x_3, x_12);
lean_inc(x_62);
lean_inc_ref(x_3);
x_64 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_64, 0, x_3);
lean_closure_set(x_64, 1, x_11);
lean_closure_set(x_64, 2, x_62);
lean_inc(x_62);
x_65 = lean_apply_1(x_62, x_60);
lean_inc(x_65);
x_66 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_66, 0, x_65);
lean_ctor_set(x_66, 1, x_64);
lean_ctor_set(x_6, 1, x_63);
lean_ctor_set(x_6, 0, x_66);
x_67 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_67);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 lean_ctor_release(x_5, 1);
 x_68 = x_5;
} else {
 lean_dec_ref(x_5);
 x_68 = lean_box(0);
}
x_69 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_6);
x_70 = lean_ctor_get(x_69, 0);
lean_inc(x_70);
x_71 = lean_ctor_get(x_69, 1);
lean_inc(x_71);
lean_dec_ref(x_69);
x_72 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_72, 0, x_3);
lean_closure_set(x_72, 1, x_67);
lean_closure_set(x_72, 2, x_62);
lean_inc_ref(x_72);
x_73 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_73, 0, x_70);
lean_ctor_set(x_73, 1, x_65);
lean_ctor_set(x_73, 2, x_72);
if (lean_is_scalar(x_68)) {
 x_74 = lean_alloc_ctor(0, 2, 0);
} else {
 x_74 = x_68;
}
lean_ctor_set(x_74, 0, x_73);
lean_ctor_set(x_74, 1, x_71);
x_75 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_74);
lean_inc_ref(x_75);
x_76 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_75);
x_77 = lean_ctor_get(x_75, 0);
lean_inc_ref(x_77);
lean_dec_ref(x_75);
x_78 = lean_ctor_get(x_76, 0);
lean_inc(x_78);
x_79 = lean_ctor_get(x_76, 1);
lean_inc(x_79);
lean_dec_ref(x_76);
x_80 = lean_ctor_get(x_77, 0);
lean_inc(x_80);
if (lean_is_exclusive(x_77)) {
 lean_ctor_release(x_77, 0);
 lean_ctor_release(x_77, 1);
 x_81 = x_77;
} else {
 lean_dec_ref(x_77);
 x_81 = lean_box(0);
}
x_82 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_82, 0, x_72);
x_83 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_83, 0, x_78);
lean_ctor_set(x_83, 1, x_80);
lean_ctor_set(x_83, 2, x_82);
if (lean_is_scalar(x_81)) {
 x_84 = lean_alloc_ctor(0, 2, 0);
} else {
 x_84 = x_81;
}
lean_ctor_set(x_84, 0, x_83);
lean_ctor_set(x_84, 1, x_79);
return x_84;
}
}
else
{
lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; 
x_85 = lean_ctor_get(x_6, 0);
lean_inc(x_85);
lean_dec(x_6);
x_86 = lean_ctor_get(x_7, 0);
lean_inc(x_86);
x_87 = lean_ctor_get(x_7, 1);
lean_inc(x_87);
lean_dec_ref(x_7);
x_88 = lean_ctor_get(x_85, 0);
lean_inc(x_88);
if (lean_is_exclusive(x_85)) {
 lean_ctor_release(x_85, 0);
 lean_ctor_release(x_85, 1);
 x_89 = x_85;
} else {
 lean_dec_ref(x_85);
 x_89 = lean_box(0);
}
lean_inc_ref(x_3);
x_90 = lp_mathlib_Equiv_symm___redArg(x_3);
x_91 = lean_ctor_get(x_90, 0);
lean_inc(x_91);
lean_dec_ref(x_90);
lean_inc_ref(x_3);
x_92 = lp_mathlib_Equiv_zero___redArg(x_3, x_87);
lean_inc(x_91);
lean_inc_ref(x_3);
x_93 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_93, 0, x_3);
lean_closure_set(x_93, 1, x_86);
lean_closure_set(x_93, 2, x_91);
lean_inc(x_91);
x_94 = lean_apply_1(x_91, x_88);
lean_inc(x_94);
if (lean_is_scalar(x_89)) {
 x_95 = lean_alloc_ctor(0, 2, 0);
} else {
 x_95 = x_89;
}
lean_ctor_set(x_95, 0, x_94);
lean_ctor_set(x_95, 1, x_93);
x_96 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_96, 0, x_95);
lean_ctor_set(x_96, 1, x_92);
x_97 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_97);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 lean_ctor_release(x_5, 1);
 x_98 = x_5;
} else {
 lean_dec_ref(x_5);
 x_98 = lean_box(0);
}
x_99 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_96);
x_100 = lean_ctor_get(x_99, 0);
lean_inc(x_100);
x_101 = lean_ctor_get(x_99, 1);
lean_inc(x_101);
lean_dec_ref(x_99);
x_102 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_102, 0, x_3);
lean_closure_set(x_102, 1, x_97);
lean_closure_set(x_102, 2, x_91);
lean_inc_ref(x_102);
x_103 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_103, 0, x_100);
lean_ctor_set(x_103, 1, x_94);
lean_ctor_set(x_103, 2, x_102);
if (lean_is_scalar(x_98)) {
 x_104 = lean_alloc_ctor(0, 2, 0);
} else {
 x_104 = x_98;
}
lean_ctor_set(x_104, 0, x_103);
lean_ctor_set(x_104, 1, x_101);
x_105 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_104);
lean_inc_ref(x_105);
x_106 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_105);
x_107 = lean_ctor_get(x_105, 0);
lean_inc_ref(x_107);
lean_dec_ref(x_105);
x_108 = lean_ctor_get(x_106, 0);
lean_inc(x_108);
x_109 = lean_ctor_get(x_106, 1);
lean_inc(x_109);
lean_dec_ref(x_106);
x_110 = lean_ctor_get(x_107, 0);
lean_inc(x_110);
if (lean_is_exclusive(x_107)) {
 lean_ctor_release(x_107, 0);
 lean_ctor_release(x_107, 1);
 x_111 = x_107;
} else {
 lean_dec_ref(x_107);
 x_111 = lean_box(0);
}
x_112 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_112, 0, x_102);
x_113 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_113, 0, x_108);
lean_ctor_set(x_113, 1, x_110);
lean_ctor_set(x_113, 2, x_112);
if (lean_is_scalar(x_111)) {
 x_114 = lean_alloc_ctor(0, 2, 0);
} else {
 x_114 = x_111;
}
lean_ctor_set(x_114, 0, x_113);
lean_ctor_set(x_114, 1, x_109);
return x_114;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoidWithZero___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_2);
lean_inc_ref(x_3);
x_4 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_3);
lean_inc_ref(x_4);
x_5 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_4);
x_6 = !lean_is_exclusive(x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_7 = lean_ctor_get(x_4, 0);
x_8 = lean_ctor_get(x_4, 1);
lean_dec(x_8);
x_9 = lean_ctor_get(x_5, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_5, 1);
lean_inc(x_10);
lean_dec_ref(x_5);
x_11 = !lean_is_exclusive(x_7);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_12 = lean_ctor_get(x_7, 0);
x_13 = lean_ctor_get(x_7, 1);
lean_dec(x_13);
lean_inc_ref(x_1);
x_14 = lp_mathlib_Equiv_symm___redArg(x_1);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_1);
x_16 = lp_mathlib_Equiv_zero___redArg(x_1, x_10);
lean_inc(x_15);
lean_inc_ref(x_1);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_17, 0, x_1);
lean_closure_set(x_17, 1, x_9);
lean_closure_set(x_17, 2, x_15);
lean_inc(x_15);
x_18 = lean_apply_1(x_15, x_12);
lean_inc(x_18);
lean_ctor_set(x_7, 1, x_17);
lean_ctor_set(x_7, 0, x_18);
lean_ctor_set(x_4, 1, x_16);
x_19 = !lean_is_exclusive(x_3);
if (x_19 == 0)
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_20 = lean_ctor_get(x_3, 0);
x_21 = lean_ctor_get(x_3, 1);
lean_dec(x_21);
x_22 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_4);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
x_24 = lean_ctor_get(x_22, 1);
lean_inc(x_24);
lean_dec_ref(x_22);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_25, 0, x_1);
lean_closure_set(x_25, 1, x_20);
lean_closure_set(x_25, 2, x_15);
lean_inc_ref(x_25);
x_26 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_26, 0, x_23);
lean_ctor_set(x_26, 1, x_18);
lean_ctor_set(x_26, 2, x_25);
lean_ctor_set(x_3, 1, x_24);
lean_ctor_set(x_3, 0, x_26);
x_27 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_3);
lean_inc_ref(x_27);
x_28 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_27);
x_29 = lean_ctor_get(x_27, 0);
lean_inc_ref(x_29);
lean_dec_ref(x_27);
x_30 = lean_ctor_get(x_28, 0);
lean_inc(x_30);
x_31 = lean_ctor_get(x_28, 1);
lean_inc(x_31);
lean_dec_ref(x_28);
x_32 = !lean_is_exclusive(x_29);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_33 = lean_ctor_get(x_29, 0);
x_34 = lean_ctor_get(x_29, 1);
lean_dec(x_34);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_35, 0, x_25);
x_36 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_36, 0, x_30);
lean_ctor_set(x_36, 1, x_33);
lean_ctor_set(x_36, 2, x_35);
lean_ctor_set(x_29, 1, x_31);
lean_ctor_set(x_29, 0, x_36);
return x_29;
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_37 = lean_ctor_get(x_29, 0);
lean_inc(x_37);
lean_dec(x_29);
x_38 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_38, 0, x_25);
x_39 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_39, 0, x_30);
lean_ctor_set(x_39, 1, x_37);
lean_ctor_set(x_39, 2, x_38);
x_40 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_40, 0, x_39);
lean_ctor_set(x_40, 1, x_31);
return x_40;
}
}
else
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_41 = lean_ctor_get(x_3, 0);
lean_inc(x_41);
lean_dec(x_3);
x_42 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_4);
x_43 = lean_ctor_get(x_42, 0);
lean_inc(x_43);
x_44 = lean_ctor_get(x_42, 1);
lean_inc(x_44);
lean_dec_ref(x_42);
x_45 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_45, 0, x_1);
lean_closure_set(x_45, 1, x_41);
lean_closure_set(x_45, 2, x_15);
lean_inc_ref(x_45);
x_46 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_46, 0, x_43);
lean_ctor_set(x_46, 1, x_18);
lean_ctor_set(x_46, 2, x_45);
x_47 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_47, 0, x_46);
lean_ctor_set(x_47, 1, x_44);
x_48 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_47);
lean_inc_ref(x_48);
x_49 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_48);
x_50 = lean_ctor_get(x_48, 0);
lean_inc_ref(x_50);
lean_dec_ref(x_48);
x_51 = lean_ctor_get(x_49, 0);
lean_inc(x_51);
x_52 = lean_ctor_get(x_49, 1);
lean_inc(x_52);
lean_dec_ref(x_49);
x_53 = lean_ctor_get(x_50, 0);
lean_inc(x_53);
if (lean_is_exclusive(x_50)) {
 lean_ctor_release(x_50, 0);
 lean_ctor_release(x_50, 1);
 x_54 = x_50;
} else {
 lean_dec_ref(x_50);
 x_54 = lean_box(0);
}
x_55 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_55, 0, x_45);
x_56 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_56, 0, x_51);
lean_ctor_set(x_56, 1, x_53);
lean_ctor_set(x_56, 2, x_55);
if (lean_is_scalar(x_54)) {
 x_57 = lean_alloc_ctor(0, 2, 0);
} else {
 x_57 = x_54;
}
lean_ctor_set(x_57, 0, x_56);
lean_ctor_set(x_57, 1, x_52);
return x_57;
}
}
else
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; 
x_58 = lean_ctor_get(x_7, 0);
lean_inc(x_58);
lean_dec(x_7);
lean_inc_ref(x_1);
x_59 = lp_mathlib_Equiv_symm___redArg(x_1);
x_60 = lean_ctor_get(x_59, 0);
lean_inc(x_60);
lean_dec_ref(x_59);
lean_inc_ref(x_1);
x_61 = lp_mathlib_Equiv_zero___redArg(x_1, x_10);
lean_inc(x_60);
lean_inc_ref(x_1);
x_62 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_62, 0, x_1);
lean_closure_set(x_62, 1, x_9);
lean_closure_set(x_62, 2, x_60);
lean_inc(x_60);
x_63 = lean_apply_1(x_60, x_58);
lean_inc(x_63);
x_64 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_64, 0, x_63);
lean_ctor_set(x_64, 1, x_62);
lean_ctor_set(x_4, 1, x_61);
lean_ctor_set(x_4, 0, x_64);
x_65 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_65);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 x_66 = x_3;
} else {
 lean_dec_ref(x_3);
 x_66 = lean_box(0);
}
x_67 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_4);
x_68 = lean_ctor_get(x_67, 0);
lean_inc(x_68);
x_69 = lean_ctor_get(x_67, 1);
lean_inc(x_69);
lean_dec_ref(x_67);
x_70 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_70, 0, x_1);
lean_closure_set(x_70, 1, x_65);
lean_closure_set(x_70, 2, x_60);
lean_inc_ref(x_70);
x_71 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_71, 0, x_68);
lean_ctor_set(x_71, 1, x_63);
lean_ctor_set(x_71, 2, x_70);
if (lean_is_scalar(x_66)) {
 x_72 = lean_alloc_ctor(0, 2, 0);
} else {
 x_72 = x_66;
}
lean_ctor_set(x_72, 0, x_71);
lean_ctor_set(x_72, 1, x_69);
x_73 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_72);
lean_inc_ref(x_73);
x_74 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_73);
x_75 = lean_ctor_get(x_73, 0);
lean_inc_ref(x_75);
lean_dec_ref(x_73);
x_76 = lean_ctor_get(x_74, 0);
lean_inc(x_76);
x_77 = lean_ctor_get(x_74, 1);
lean_inc(x_77);
lean_dec_ref(x_74);
x_78 = lean_ctor_get(x_75, 0);
lean_inc(x_78);
if (lean_is_exclusive(x_75)) {
 lean_ctor_release(x_75, 0);
 lean_ctor_release(x_75, 1);
 x_79 = x_75;
} else {
 lean_dec_ref(x_75);
 x_79 = lean_box(0);
}
x_80 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_80, 0, x_70);
x_81 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_81, 0, x_76);
lean_ctor_set(x_81, 1, x_78);
lean_ctor_set(x_81, 2, x_80);
if (lean_is_scalar(x_79)) {
 x_82 = lean_alloc_ctor(0, 2, 0);
} else {
 x_82 = x_79;
}
lean_ctor_set(x_82, 0, x_81);
lean_ctor_set(x_82, 1, x_77);
return x_82;
}
}
else
{
lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; 
x_83 = lean_ctor_get(x_4, 0);
lean_inc(x_83);
lean_dec(x_4);
x_84 = lean_ctor_get(x_5, 0);
lean_inc(x_84);
x_85 = lean_ctor_get(x_5, 1);
lean_inc(x_85);
lean_dec_ref(x_5);
x_86 = lean_ctor_get(x_83, 0);
lean_inc(x_86);
if (lean_is_exclusive(x_83)) {
 lean_ctor_release(x_83, 0);
 lean_ctor_release(x_83, 1);
 x_87 = x_83;
} else {
 lean_dec_ref(x_83);
 x_87 = lean_box(0);
}
lean_inc_ref(x_1);
x_88 = lp_mathlib_Equiv_symm___redArg(x_1);
x_89 = lean_ctor_get(x_88, 0);
lean_inc(x_89);
lean_dec_ref(x_88);
lean_inc_ref(x_1);
x_90 = lp_mathlib_Equiv_zero___redArg(x_1, x_85);
lean_inc(x_89);
lean_inc_ref(x_1);
x_91 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulZeroOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_91, 0, x_1);
lean_closure_set(x_91, 1, x_84);
lean_closure_set(x_91, 2, x_89);
lean_inc(x_89);
x_92 = lean_apply_1(x_89, x_86);
lean_inc(x_92);
if (lean_is_scalar(x_87)) {
 x_93 = lean_alloc_ctor(0, 2, 0);
} else {
 x_93 = x_87;
}
lean_ctor_set(x_93, 0, x_92);
lean_ctor_set(x_93, 1, x_91);
x_94 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_94, 0, x_93);
lean_ctor_set(x_94, 1, x_90);
x_95 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_95);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 x_96 = x_3;
} else {
 lean_dec_ref(x_3);
 x_96 = lean_box(0);
}
x_97 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_94);
x_98 = lean_ctor_get(x_97, 0);
lean_inc(x_98);
x_99 = lean_ctor_get(x_97, 1);
lean_inc(x_99);
lean_dec_ref(x_97);
x_100 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoidWithZero___redArg___lam__1), 5, 3);
lean_closure_set(x_100, 0, x_1);
lean_closure_set(x_100, 1, x_95);
lean_closure_set(x_100, 2, x_89);
lean_inc_ref(x_100);
x_101 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_101, 0, x_98);
lean_ctor_set(x_101, 1, x_92);
lean_ctor_set(x_101, 2, x_100);
if (lean_is_scalar(x_96)) {
 x_102 = lean_alloc_ctor(0, 2, 0);
} else {
 x_102 = x_96;
}
lean_ctor_set(x_102, 0, x_101);
lean_ctor_set(x_102, 1, x_99);
x_103 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_102);
lean_inc_ref(x_103);
x_104 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_103);
x_105 = lean_ctor_get(x_103, 0);
lean_inc_ref(x_105);
lean_dec_ref(x_103);
x_106 = lean_ctor_get(x_104, 0);
lean_inc(x_106);
x_107 = lean_ctor_get(x_104, 1);
lean_inc(x_107);
lean_dec_ref(x_104);
x_108 = lean_ctor_get(x_105, 0);
lean_inc(x_108);
if (lean_is_exclusive(x_105)) {
 lean_ctor_release(x_105, 0);
 lean_ctor_release(x_105, 1);
 x_109 = x_105;
} else {
 lean_dec_ref(x_105);
 x_109 = lean_box(0);
}
x_110 = lean_alloc_closure((void*)(lp_mathlib_Equiv_commMonoidWithZero___redArg___lam__2), 3, 1);
lean_closure_set(x_110, 0, x_100);
x_111 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_111, 0, x_106);
lean_ctor_set(x_111, 1, x_108);
lean_ctor_set(x_111, 2, x_110);
if (lean_is_scalar(x_109)) {
 x_112 = lean_alloc_ctor(0, 2, 0);
} else {
 x_112 = x_109;
}
lean_ctor_set(x_112, 0, x_111);
lean_ctor_set(x_112, 1, x_107);
return x_112;
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_TransferInstance(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
