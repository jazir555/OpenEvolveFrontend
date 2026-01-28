// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Nat
// Imports: public import Init public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Algebra.GroupWithZero.Defs public import Mathlib.Tactic.Spread
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_instMulZeroOneClass;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instMulZeroClass;
static lean_object* lp_mathlib_Nat_instMulZeroClass___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemigroupWithZero;
extern lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instMonoidWithZero;
lean_object* l_Nat_mul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCancelCommMonoidWithZero;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoidWithZero;
extern lean_object* lp_mathlib_Nat_instCommMonoid;
static lean_object* lp_mathlib_Nat_instMulZeroClass___closed__1;
static lean_object* _init_lp_mathlib_Nat_instMulZeroClass___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid;
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_instMulZeroClass___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instMulZeroClass() {
_start:
{
lean_object* x_1; uint8_t x_2; 
x_1 = lp_mathlib_Nat_instMulZeroClass___closed__0;
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
lean_dec(x_4);
x_5 = lp_mathlib_Nat_instMulZeroClass___closed__1;
lean_ctor_set(x_1, 1, x_3);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec(x_1);
x_7 = lp_mathlib_Nat_instMulZeroClass___closed__1;
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
static lean_object* _init_lp_mathlib_Nat_instSemigroupWithZero() {
_start:
{
lean_object* x_1; uint8_t x_2; 
x_1 = lp_mathlib_Nat_instMulZeroClass;
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_dec(x_3);
x_4 = lp_mathlib_Nat_instMulZeroClass___closed__1;
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec(x_1);
x_6 = lp_mathlib_Nat_instMulZeroClass___closed__1;
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_5);
return x_7;
}
}
}
static lean_object* _init_lp_mathlib_Nat_instMonoidWithZero() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; 
x_1 = lp_mathlib_Nat_instCommMonoid;
x_2 = lp_mathlib_Nat_instMulZeroClass;
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_2, 0);
lean_dec(x_4);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
else
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec(x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_1);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
}
static lean_object* _init_lp_mathlib_Nat_instCommMonoidWithZero() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; 
x_1 = lp_mathlib_Nat_instCommMonoid;
x_2 = lp_mathlib_Nat_instMonoidWithZero;
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_2, 0);
lean_dec(x_4);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
else
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec(x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_1);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
}
static lean_object* _init_lp_mathlib_Nat_instCancelCommMonoidWithZero() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instCommMonoidWithZero;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instMulZeroOneClass() {
_start:
{
lean_object* x_1; uint8_t x_2; 
x_1 = lp_mathlib_Nat_instMulZeroClass;
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_unsigned_to_nat(1u);
lean_ctor_set(x_1, 1, x_3);
lean_ctor_set(x_1, 0, x_5);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_1);
lean_ctor_set(x_6, 1, x_4);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_9 = lean_unsigned_to_nat(1u);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_8);
return x_11;
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instMulZeroClass___closed__0 = _init_lp_mathlib_Nat_instMulZeroClass___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instMulZeroClass___closed__0);
lp_mathlib_Nat_instMulZeroClass___closed__1 = _init_lp_mathlib_Nat_instMulZeroClass___closed__1();
lean_mark_persistent(lp_mathlib_Nat_instMulZeroClass___closed__1);
lp_mathlib_Nat_instMulZeroClass = _init_lp_mathlib_Nat_instMulZeroClass();
lean_mark_persistent(lp_mathlib_Nat_instMulZeroClass);
lp_mathlib_Nat_instSemigroupWithZero = _init_lp_mathlib_Nat_instSemigroupWithZero();
lean_mark_persistent(lp_mathlib_Nat_instSemigroupWithZero);
lp_mathlib_Nat_instMonoidWithZero = _init_lp_mathlib_Nat_instMonoidWithZero();
lean_mark_persistent(lp_mathlib_Nat_instMonoidWithZero);
lp_mathlib_Nat_instCommMonoidWithZero = _init_lp_mathlib_Nat_instCommMonoidWithZero();
lean_mark_persistent(lp_mathlib_Nat_instCommMonoidWithZero);
lp_mathlib_Nat_instCancelCommMonoidWithZero = _init_lp_mathlib_Nat_instCancelCommMonoidWithZero();
lean_mark_persistent(lp_mathlib_Nat_instCancelCommMonoidWithZero);
lp_mathlib_Nat_instMulZeroOneClass = _init_lp_mathlib_Nat_instMulZeroOneClass();
lean_mark_persistent(lp_mathlib_Nat_instMulZeroOneClass);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
