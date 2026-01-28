// Lean compiler output
// Module: Mathlib.Algebra.Group.Nat.Defs
// Imports: public import Init public import Mathlib.Algebra.Group.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_instOne;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instMulOneClass;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoid___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemigroup;
static lean_object* lp_mathlib_Nat_instAddCancelCommMonoid___closed__1;
static lean_object* lp_mathlib_Nat_instAddCancelCommMonoid___closed__0;
static lean_object* lp_mathlib_Nat_instMulOneClass___closed__1;
lean_object* l_Nat_mul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoid___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoid;
lean_object* lean_nat_pow(lean_object*, lean_object*);
static lean_object* lp_mathlib_Nat_instAddCancelCommMonoid___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddCommMonoid;
static lean_object* lp_mathlib_Nat_instMulOneClass___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddCommSemigroup;
static lean_object* _init_lp_mathlib_Nat_instMulOneClass___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instMulOneClass___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Nat_instMulOneClass___closed__0;
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Nat_instMulOneClass() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instMulOneClass___closed__1;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddCancelCommMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddCancelCommMonoid___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_add___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddCancelCommMonoid___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid___closed__0;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_mathlib_Nat_instAddCancelCommMonoid___closed__1;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddCancelCommMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid___closed__2;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoid___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_nat_pow(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoid___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_instCommMonoid___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Nat_instCommMonoid() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_instCommMonoid___lam__0___boxed), 2, 0);
x_2 = lp_mathlib_Nat_instMulOneClass___closed__0;
x_3 = lean_unsigned_to_nat(1u);
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddCommMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instCommMonoid;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instCommSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instCommMonoid;
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_instSemigroup() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instMulOneClass___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddCommSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid;
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddSemigroup() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid___closed__1;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instOne() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(1u);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instMulOneClass___closed__0 = _init_lp_mathlib_Nat_instMulOneClass___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instMulOneClass___closed__0);
lp_mathlib_Nat_instMulOneClass___closed__1 = _init_lp_mathlib_Nat_instMulOneClass___closed__1();
lean_mark_persistent(lp_mathlib_Nat_instMulOneClass___closed__1);
lp_mathlib_Nat_instMulOneClass = _init_lp_mathlib_Nat_instMulOneClass();
lean_mark_persistent(lp_mathlib_Nat_instMulOneClass);
lp_mathlib_Nat_instAddCancelCommMonoid___closed__0 = _init_lp_mathlib_Nat_instAddCancelCommMonoid___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instAddCancelCommMonoid___closed__0);
lp_mathlib_Nat_instAddCancelCommMonoid___closed__1 = _init_lp_mathlib_Nat_instAddCancelCommMonoid___closed__1();
lean_mark_persistent(lp_mathlib_Nat_instAddCancelCommMonoid___closed__1);
lp_mathlib_Nat_instAddCancelCommMonoid___closed__2 = _init_lp_mathlib_Nat_instAddCancelCommMonoid___closed__2();
lean_mark_persistent(lp_mathlib_Nat_instAddCancelCommMonoid___closed__2);
lp_mathlib_Nat_instAddCancelCommMonoid = _init_lp_mathlib_Nat_instAddCancelCommMonoid();
lean_mark_persistent(lp_mathlib_Nat_instAddCancelCommMonoid);
lp_mathlib_Nat_instCommMonoid = _init_lp_mathlib_Nat_instCommMonoid();
lean_mark_persistent(lp_mathlib_Nat_instCommMonoid);
lp_mathlib_Nat_instAddCommMonoid = _init_lp_mathlib_Nat_instAddCommMonoid();
lean_mark_persistent(lp_mathlib_Nat_instAddCommMonoid);
lp_mathlib_Nat_instAddMonoid = _init_lp_mathlib_Nat_instAddMonoid();
lean_mark_persistent(lp_mathlib_Nat_instAddMonoid);
lp_mathlib_Nat_instMonoid = _init_lp_mathlib_Nat_instMonoid();
lean_mark_persistent(lp_mathlib_Nat_instMonoid);
lp_mathlib_Nat_instCommSemigroup = _init_lp_mathlib_Nat_instCommSemigroup();
lean_mark_persistent(lp_mathlib_Nat_instCommSemigroup);
lp_mathlib_Nat_instSemigroup = _init_lp_mathlib_Nat_instSemigroup();
lean_mark_persistent(lp_mathlib_Nat_instSemigroup);
lp_mathlib_Nat_instAddCommSemigroup = _init_lp_mathlib_Nat_instAddCommSemigroup();
lean_mark_persistent(lp_mathlib_Nat_instAddCommSemigroup);
lp_mathlib_Nat_instAddSemigroup = _init_lp_mathlib_Nat_instAddSemigroup();
lean_mark_persistent(lp_mathlib_Nat_instAddSemigroup);
lp_mathlib_Nat_instOne = _init_lp_mathlib_Nat_instOne();
lean_mark_persistent(lp_mathlib_Nat_instOne);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
