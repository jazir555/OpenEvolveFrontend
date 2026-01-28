// Lean compiler output
// Module: Mathlib.Algebra.Order.Nonneg.Basic
// Imports: public import Init public import Mathlib.Algebra.Order.GroupWithZero.Unbundled.Basic public import Mathlib.Algebra.Order.Monoid.Unbundled.Pow public import Mathlib.Algebra.Order.ZeroLEOne public import Mathlib.Algebra.Ring.Defs public import Mathlib.Algebra.Ring.InjSurj public import Mathlib.Data.Nat.Cast.Order.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___redArg___boxed(lean_object*);
lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_Nonneg_coeRingHom___closed__0;
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addCancelCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Nonneg_inhabited(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nonneg_inhabited___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Nonneg_zero(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nonneg_zero___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_add___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_add___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_add(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_nsmul___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_nsmul(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Nonneg_one(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_3);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nonneg_one___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_mul___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_mul(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lp_mathlib_Nonneg_add___redArg(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lp_mathlib_Function_Injective_addMonoid___redArg(x_4, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_addMonoid___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_addMonoid(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nonneg_coeAddMonoidHom___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_coeAddMonoidHom___lam__0___boxed), 1, 0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_coeAddMonoidHom(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lp_mathlib_Nonneg_add___redArg(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lp_mathlib_Function_Injective_addMonoid___redArg(x_4, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_addCommMonoid___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_addCommMonoid(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lp_mathlib_Nonneg_add___redArg(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lp_mathlib_Function_Injective_addCancelCommMonoid___redArg(x_4, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_addCancelCommMonoid___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_addCancelCommMonoid(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_natCast___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Nonneg_natCast___redArg(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Nonneg_natCast(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 2);
lean_inc(x_3);
x_4 = lp_mathlib_Nonneg_natCast___redArg(x_1);
x_5 = lp_mathlib_Nonneg_addMonoid___redArg(x_2);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
lean_ctor_set(x_6, 2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Nonneg_addMonoidWithOne___redArg(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Nonneg_addMonoidWithOne(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_3, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_pow___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Nonneg_pow___redArg(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Nonneg_pow(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 3);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_2 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
lean_inc_ref(x_2);
x_3 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_3, 2);
lean_inc(x_5);
x_6 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_4);
lean_dec_ref(x_4);
x_7 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_2);
lean_inc_ref(x_7);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
x_10 = !lean_is_exclusive(x_7);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_11 = lean_ctor_get(x_7, 0);
x_12 = lean_ctor_get(x_7, 1);
lean_dec(x_12);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_1);
x_14 = lp_mathlib_Nonneg_add___redArg(x_6);
x_15 = lp_mathlib_Nonneg_mul___redArg(x_8);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(x_16, 0, x_11);
x_17 = lp_mathlib_Nonneg_natCast___redArg(x_3);
x_18 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_17);
x_19 = lp_mathlib_Function_Injective_addMonoid___redArg(x_14, x_9, x_16);
lean_ctor_set(x_7, 1, x_15);
lean_ctor_set(x_7, 0, x_19);
x_20 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_20, 0, x_7);
lean_ctor_set(x_20, 1, x_5);
lean_ctor_set(x_20, 2, x_18);
lean_ctor_set(x_20, 3, x_13);
return x_20;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_21 = lean_ctor_get(x_7, 0);
lean_inc(x_21);
lean_dec(x_7);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_22, 0, x_1);
x_23 = lp_mathlib_Nonneg_add___redArg(x_6);
x_24 = lp_mathlib_Nonneg_mul___redArg(x_8);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(x_25, 0, x_21);
x_26 = lp_mathlib_Nonneg_natCast___redArg(x_3);
x_27 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_27, 0, lean_box(0));
lean_closure_set(x_27, 1, x_26);
x_28 = lp_mathlib_Function_Injective_addMonoid___redArg(x_23, x_9, x_25);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_24);
x_30 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_30, 0, x_29);
lean_ctor_set(x_30, 1, x_5);
lean_ctor_set(x_30, 2, x_27);
lean_ctor_set(x_30, 3, x_22);
return x_30;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_semiring___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_semiring(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Nonneg_semiring___redArg(x_1);
x_3 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_monoidWithZero___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_monoidWithZero(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_3);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Nonneg_coeRingHom___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_coeAddMonoidHom___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeRingHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_coeRingHom___closed__0;
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeRingHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_coeRingHom(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_2 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
lean_inc_ref(x_2);
x_3 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_3, 2);
lean_inc(x_5);
x_6 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_4);
lean_dec_ref(x_4);
x_7 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_2);
lean_inc_ref(x_7);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
x_10 = !lean_is_exclusive(x_7);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_11 = lean_ctor_get(x_7, 0);
x_12 = lean_ctor_get(x_7, 1);
lean_dec(x_12);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_1);
x_14 = lp_mathlib_Nonneg_add___redArg(x_6);
x_15 = lp_mathlib_Nonneg_mul___redArg(x_8);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(x_16, 0, x_11);
x_17 = lp_mathlib_Nonneg_natCast___redArg(x_3);
x_18 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_17);
x_19 = lp_mathlib_Function_Injective_addMonoid___redArg(x_14, x_9, x_16);
lean_ctor_set(x_7, 1, x_15);
lean_ctor_set(x_7, 0, x_19);
x_20 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_20, 0, x_7);
lean_ctor_set(x_20, 1, x_5);
lean_ctor_set(x_20, 2, x_18);
lean_ctor_set(x_20, 3, x_13);
return x_20;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_21 = lean_ctor_get(x_7, 0);
lean_inc(x_21);
lean_dec(x_7);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_22, 0, x_1);
x_23 = lp_mathlib_Nonneg_add___redArg(x_6);
x_24 = lp_mathlib_Nonneg_mul___redArg(x_8);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(x_25, 0, x_21);
x_26 = lp_mathlib_Nonneg_natCast___redArg(x_3);
x_27 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_27, 0, lean_box(0));
lean_closure_set(x_27, 1, x_26);
x_28 = lp_mathlib_Function_Injective_addMonoid___redArg(x_23, x_9, x_25);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_24);
x_30 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_30, 0, x_29);
lean_ctor_set(x_30, 1, x_5);
lean_ctor_set(x_30, 2, x_27);
lean_ctor_set(x_30, 3, x_22);
return x_30;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_commSemiring___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_commSemiring(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Nonneg_commSemiring___redArg(x_1);
x_3 = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_commMonoidWithZero___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Nonneg_commMonoidWithZero(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_apply_2(x_4, x_3, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_toNonneg___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_apply_2(x_1, x_4, x_5);
x_7 = lp_mathlib_Nonneg_toNonneg___redArg(x_2, x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Nonneg_sub___redArg___lam__0), 5, 3);
lean_closure_set(x_4, 0, x_3);
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nonneg_sub___redArg(x_2, x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Unbundled_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nonneg_coeRingHom___closed__0 = _init_lp_mathlib_Nonneg_coeRingHom___closed__0();
lean_mark_persistent(lp_mathlib_Nonneg_coeRingHom___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
