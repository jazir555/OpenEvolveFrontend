// Lean compiler output
// Module: Mathlib.Algebra.Group.Hom.End
// Imports: public import Init public import Mathlib.Algebra.Group.Hom.Instances public import Mathlib.Algebra.Ring.Defs
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
lean_object* lp_mathlib_AddMonoidHom_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_End_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_id___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_id___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0;
lean_object* lp_mathlib_Nat_iterate___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lp_mathlib_OneHom_id___lam__0(x_3);
x_6 = lean_apply_2(x_4, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_OneHom_id___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(x_1);
x_4 = lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0;
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instSemiring___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lp_mathlib_Nat_iterate___redArg(x_4, x_1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lp_mathlib_AddMonoid_End_instMonoid___redArg(x_2);
lean_inc_ref(x_1);
x_4 = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(x_1);
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instSemiring___redArg___lam__1), 3, 0);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_4);
lean_ctor_set(x_8, 1, x_5);
x_9 = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_9, 0, x_1);
x_10 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_6);
lean_ctor_set(x_10, 2, x_9);
lean_ctor_set(x_10, 3, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddMonoid_End_instSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_OneHom_id___lam__0(x_3);
x_5 = lean_apply_2(x_1, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddMonoid_End_instRing___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_3, x_4);
x_6 = lean_apply_2(x_1, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_AddMonoid_End_instSemiring___redArg(x_2);
x_5 = lp_mathlib_AddMonoidHom_instAddCommGroup___redArg(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 2);
lean_inc(x_7);
lean_dec_ref(x_5);
lean_inc(x_3);
x_8 = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instRing___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_8, 0, x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instRing___redArg___lam__1), 4, 1);
lean_closure_set(x_9, 0, x_3);
x_10 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_10, 0, x_4);
lean_ctor_set(x_10, 1, x_6);
lean_ctor_set(x_10, 2, x_7);
lean_ctor_set(x_10, 3, x_9);
lean_ctor_set(x_10, 4, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddMonoid_End_instRing___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_End(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0 = _init_lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0();
lean_mark_persistent(lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
