// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Opposite
// Imports: public import Init public import Mathlib.Algebra.Group.Opposite public import Mathlib.Algebra.GroupWithZero.InjSurj public import Mathlib.Algebra.GroupWithZero.NeZero
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
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroOneClass(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instMul___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroupWithZero(lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroClass___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lp_mathlib_MulOpposite_instMul___redArg(x_3);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_1);
x_7 = lp_mathlib_MulOpposite_instMul___redArg(x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instMulZeroClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_MulOpposite_instMulOne___redArg(x_2);
x_4 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_1);
x_5 = lp_mathlib_MulOpposite_instMulZeroClass___redArg(x_4);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_dec(x_7);
lean_ctor_set(x_5, 0, x_3);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_3);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroOneClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroupWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(x_1);
x_4 = lp_mathlib_MulOpposite_instMulZeroClass___redArg(x_3);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_4, 0);
lean_dec(x_6);
x_7 = lp_mathlib_MulOpposite_instMul___redArg(x_2);
lean_ctor_set(x_4, 0, x_7);
return x_4;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_4, 1);
lean_inc(x_8);
lean_dec(x_4);
x_9 = lp_mathlib_MulOpposite_instMul___redArg(x_2);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_8);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroupWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instSemigroupWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_MulOpposite_instMonoid___redArg(x_2);
x_4 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_1);
x_5 = lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(x_4);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_dec(x_7);
lean_ctor_set(x_5, 0, x_3);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_3);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoidWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instMonoidWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_MulOpposite_instMonoidWithZero___redArg(x_2);
x_5 = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(x_1);
x_6 = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_6, 3);
lean_dec(x_8);
x_9 = lean_ctor_get(x_6, 0);
lean_dec(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instGroupWithZero___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_3);
lean_ctor_set(x_6, 3, x_10);
lean_ctor_set(x_6, 0, x_4);
return x_6;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_6);
x_13 = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instGroupWithZero___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_3);
x_14 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_14, 0, x_4);
lean_ctor_set(x_14, 1, x_11);
lean_ctor_set(x_14, 2, x_12);
lean_ctor_set(x_14, 3, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instGroupWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroClass___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lp_mathlib_AddOpposite_instMul___redArg(x_3);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_1);
x_7 = lp_mathlib_AddOpposite_instMul___redArg(x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instMulZeroClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_AddOpposite_instMulOneClass___redArg(x_2);
x_4 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_1);
x_5 = lp_mathlib_AddOpposite_instMulZeroClass___redArg(x_4);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_dec(x_7);
lean_ctor_set(x_5, 0, x_3);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_3);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroOneClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroupWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(x_1);
x_4 = lp_mathlib_AddOpposite_instMulZeroClass___redArg(x_3);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_4, 0);
lean_dec(x_6);
x_7 = lp_mathlib_AddOpposite_instMul___redArg(x_2);
lean_ctor_set(x_4, 0, x_7);
return x_4;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_4, 1);
lean_inc(x_8);
lean_dec(x_4);
x_9 = lp_mathlib_AddOpposite_instMul___redArg(x_2);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_8);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroupWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instSemigroupWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_AddOpposite_instMonoid___redArg(x_2);
x_4 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_1);
x_5 = lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(x_4);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_dec(x_7);
lean_ctor_set(x_5, 0, x_3);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_3);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoidWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instMonoidWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_AddOpposite_instMonoidWithZero___redArg(x_2);
x_5 = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(x_1);
x_6 = lp_mathlib_AddOpposite_instDivInvMonoid___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_6, 3);
lean_dec(x_8);
x_9 = lean_ctor_get(x_6, 0);
lean_dec(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instGroupWithZero___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_3);
lean_ctor_set(x_6, 3, x_10);
lean_ctor_set(x_6, 0, x_4);
return x_6;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_6);
x_13 = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instGroupWithZero___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_3);
x_14 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_14, 0, x_4);
lean_ctor_set(x_14, 1, x_11);
lean_ctor_set(x_14, 2, x_12);
lean_ctor_set(x_14, 3, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddOpposite_instGroupWithZero___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
