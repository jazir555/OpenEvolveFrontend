// Lean compiler output
// Module: Mathlib.Algebra.Ring.Nat
// Imports: public import Init public import Mathlib.Algebra.CharZero.Defs public import Mathlib.Algebra.GroupWithZero.Nat public import Mathlib.Algebra.Ring.Defs public import Mathlib.Data.Nat.Basic
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
extern lean_object* lp_mathlib_Nat_instMulZeroOneClass;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddCommMonoidWithOne;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemiring___lam__0___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Nat_instNonAssocSemiring___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne___lam__0(lean_object*);
static lean_object* lp_mathlib_Nat_instDistrib___closed__0;
extern lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instNonUnitalSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instNonAssocSemiring;
lean_object* l_Nat_mul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instNonUnitalNonAssocSemiring;
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDistrib;
static lean_object* lp_mathlib_Nat_instDistrib___closed__1;
lean_object* lean_nat_pow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemiring;
static lean_object* lp_mathlib_Nat_instDistrib___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne___lam__0___boxed(lean_object*);
static lean_object* lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemiring___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nat_instAddMonoidWithOne___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddMonoidWithOne() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_instAddMonoidWithOne___lam__0___boxed), 1, 0);
x_2 = lp_mathlib_Nat_instAddCancelCommMonoid;
x_3 = lean_unsigned_to_nat(1u);
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddCommMonoidWithOne() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instAddMonoidWithOne;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instDistrib___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instDistrib___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_add___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instDistrib___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Nat_instDistrib___closed__1;
x_2 = lp_mathlib_Nat_instDistrib___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Nat_instDistrib() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instDistrib___closed__2;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Nat_instDistrib___closed__0;
x_2 = lp_mathlib_Nat_instAddCancelCommMonoid;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Nat_instNonUnitalNonAssocSemiring() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instNonUnitalSemiring() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instNonUnitalNonAssocSemiring;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instNonAssocSemiring___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_instAddMonoidWithOne___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instNonAssocSemiring() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_mathlib_Nat_instNonUnitalNonAssocSemiring;
x_2 = lp_mathlib_Nat_instMulZeroOneClass;
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_Nat_instNonAssocSemiring___closed__0;
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_1);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemiring___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_nat_pow(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemiring___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_instSemiring___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Nat_instSemiring() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_mathlib_Nat_instNonUnitalNonAssocSemiring;
x_2 = lp_mathlib_Nat_instNonAssocSemiring;
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 2);
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Nat_instSemiring___lam__0___boxed), 2, 0);
x_6 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_6, 0, x_1);
lean_ctor_set(x_6, 1, x_3);
lean_ctor_set(x_6, 2, x_4);
lean_ctor_set(x_6, 3, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_Nat_instCommSemiring() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instSemiring;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instAddMonoidWithOne = _init_lp_mathlib_Nat_instAddMonoidWithOne();
lean_mark_persistent(lp_mathlib_Nat_instAddMonoidWithOne);
lp_mathlib_Nat_instAddCommMonoidWithOne = _init_lp_mathlib_Nat_instAddCommMonoidWithOne();
lean_mark_persistent(lp_mathlib_Nat_instAddCommMonoidWithOne);
lp_mathlib_Nat_instDistrib___closed__0 = _init_lp_mathlib_Nat_instDistrib___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instDistrib___closed__0);
lp_mathlib_Nat_instDistrib___closed__1 = _init_lp_mathlib_Nat_instDistrib___closed__1();
lean_mark_persistent(lp_mathlib_Nat_instDistrib___closed__1);
lp_mathlib_Nat_instDistrib___closed__2 = _init_lp_mathlib_Nat_instDistrib___closed__2();
lean_mark_persistent(lp_mathlib_Nat_instDistrib___closed__2);
lp_mathlib_Nat_instDistrib = _init_lp_mathlib_Nat_instDistrib();
lean_mark_persistent(lp_mathlib_Nat_instDistrib);
lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0 = _init_lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0);
lp_mathlib_Nat_instNonUnitalNonAssocSemiring = _init_lp_mathlib_Nat_instNonUnitalNonAssocSemiring();
lean_mark_persistent(lp_mathlib_Nat_instNonUnitalNonAssocSemiring);
lp_mathlib_Nat_instNonUnitalSemiring = _init_lp_mathlib_Nat_instNonUnitalSemiring();
lean_mark_persistent(lp_mathlib_Nat_instNonUnitalSemiring);
lp_mathlib_Nat_instNonAssocSemiring___closed__0 = _init_lp_mathlib_Nat_instNonAssocSemiring___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instNonAssocSemiring___closed__0);
lp_mathlib_Nat_instNonAssocSemiring = _init_lp_mathlib_Nat_instNonAssocSemiring();
lean_mark_persistent(lp_mathlib_Nat_instNonAssocSemiring);
lp_mathlib_Nat_instSemiring = _init_lp_mathlib_Nat_instSemiring();
lean_mark_persistent(lp_mathlib_Nat_instSemiring);
lp_mathlib_Nat_instCommSemiring = _init_lp_mathlib_Nat_instCommSemiring();
lean_mark_persistent(lp_mathlib_Nat_instCommSemiring);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
