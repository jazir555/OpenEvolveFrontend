// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Pi
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Defs public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Algebra.Group.Pi.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instZero___redArg(lean_object*);
lean_object* lp_mathlib_Pi_semigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_single___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_single___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_monoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Pi_mulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lp_mathlib_Pi_instMul___redArg(x_2);
x_5 = lp_mathlib_Pi_instZero___redArg(x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_mulZeroClass___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_single___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 5);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_4);
lean_closure_set(x_5, 3, x_2);
lean_closure_set(x_5, 4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_single(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MulHom_single___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroOneClass___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_mulOneClass___redArg(x_2);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_3, 1);
lean_dec(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroOneClass___redArg___lam__1), 2, 1);
lean_closure_set(x_6, 0, x_1);
lean_inc_ref(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lp_mathlib_Pi_instMul___redArg(x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(x_9, 0, x_6);
x_10 = lp_mathlib_Pi_instZero___redArg(x_9);
lean_ctor_set(x_3, 1, x_8);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_3);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_12 = lean_ctor_get(x_3, 0);
lean_inc(x_12);
lean_dec(x_3);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroOneClass___redArg___lam__1), 2, 1);
lean_closure_set(x_13, 0, x_1);
lean_inc_ref(x_13);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(x_14, 0, x_13);
x_15 = lp_mathlib_Pi_instMul___redArg(x_14);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(x_16, 0, x_13);
x_17 = lp_mathlib_Pi_instZero___redArg(x_16);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_12);
lean_ctor_set(x_18, 1, x_15);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_17);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_mulZeroOneClass___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_3);
x_5 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidWithZero___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidWithZero___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lp_mathlib_Pi_monoid___redArg(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(x_5, 0, x_3);
x_6 = lp_mathlib_Pi_instZero___redArg(x_5);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_4);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_monoidWithZero___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_commMonoidWithZero___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_2);
x_3 = lp_mathlib_Pi_monoidWithZero___redArg(x_2);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 1);
lean_dec(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidWithZero___redArg___lam__1), 2, 1);
lean_closure_set(x_6, 0, x_2);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lp_mathlib_Pi_instZero___redArg(x_7);
lean_ctor_set(x_3, 1, x_8);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_3, 0);
lean_inc(x_9);
lean_dec(x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidWithZero___redArg___lam__1), 2, 1);
lean_closure_set(x_10, 0, x_2);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(x_11, 0, x_10);
x_12 = lp_mathlib_Pi_instZero___redArg(x_11);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_9);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_commMonoidWithZero___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_semigroupWithZero___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_semigroupWithZero___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lp_mathlib_Pi_semigroup___redArg(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(x_5, 0, x_3);
x_6 = lp_mathlib_Pi_instZero___redArg(x_5);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_4);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_semigroupWithZero___redArg(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
