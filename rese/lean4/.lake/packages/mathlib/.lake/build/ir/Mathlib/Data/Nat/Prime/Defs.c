// Lean compiler output
// Module: Mathlib.Data.Nat.Prime.Defs
// Imports: public import Init public import Mathlib.Algebra.Group.Nat.Units public import Mathlib.Algebra.GroupWithZero.Nat public import Mathlib.Algebra.Prime.Defs public import Mathlib.Data.Nat.Sqrt public import Mathlib.Order.Basic
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
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidablePredIrreducible(lean_object*);
uint8_t l_Nat_decidable__dvd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_coeNat___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_instRepr___lam__0(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFac(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_inhabitedPrimes;
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime_x27___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFac___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_instRepr___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFacAux(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidablePredPrime(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidablePredPrime___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidablePredIrreducible___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_coeNat;
uint8_t lp_mathlib_Nat_decidableLoHi___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFacAux___boxed(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableEqPrimes(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_instRepr;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableEqPrimes___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_coeNat___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime_x27(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime___lam__0(lean_object* x_1, uint8_t x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = l_Nat_decidable__dvd(x_3, x_1);
if (x_4 == 0)
{
return x_2;
}
else
{
uint8_t x_5; 
x_5 = 0;
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_4 = lean_unbox(x_2);
x_5 = lp_mathlib_Nat_decidablePrime___lam__0(x_1, x_4, x_3);
lean_dec(x_3);
lean_dec(x_1);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_unsigned_to_nat(2u);
x_3 = lean_nat_dec_le(x_2, x_1);
if (x_3 == 0)
{
lean_dec(x_1);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_box(x_3);
lean_inc(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Nat_decidablePrime___lam__0___boxed), 3, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_4);
x_6 = lp_mathlib_Nat_decidableLoHi___redArg(x_2, x_1, x_5);
lean_dec(x_1);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Nat_decidablePrime(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFacAux(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_nat_mul(x_2, x_2);
x_4 = lean_nat_dec_lt(x_1, x_3);
lean_dec(x_3);
if (x_4 == 0)
{
uint8_t x_5; 
x_5 = l_Nat_decidable__dvd(x_2, x_1);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_unsigned_to_nat(2u);
x_7 = lean_nat_add(x_2, x_6);
lean_dec(x_2);
x_2 = x_7;
goto _start;
}
else
{
return x_2;
}
}
else
{
lean_dec(x_2);
lean_inc(x_1);
return x_1;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFacAux___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_minFacAux(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFac(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_unsigned_to_nat(2u);
x_3 = l_Nat_decidable__dvd(x_2, x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_unsigned_to_nat(3u);
x_5 = lp_mathlib_Nat_minFacAux(x_1, x_4);
return x_5;
}
else
{
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFac___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nat_minFac(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime_x27(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_unsigned_to_nat(2u);
x_3 = lean_nat_dec_le(x_2, x_1);
if (x_3 == 0)
{
return x_3;
}
else
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_Nat_minFac(x_1);
x_5 = lean_nat_dec_eq(x_4, x_1);
lean_dec(x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime_x27___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Nat_decidablePrime_x27(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidablePredPrime(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = lp_mathlib_Nat_decidablePrime_x27(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidablePredPrime___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Nat_instDecidablePredPrime(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidablePredIrreducible(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = lp_mathlib_Nat_decidablePrime_x27(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidablePredIrreducible___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Nat_instDecidablePredIrreducible(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableEqPrimes(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableEqPrimes___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Nat_instDecidableEqPrimes(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_instRepr___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = l_Nat_reprFast(x_1);
x_4 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_instRepr___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_Primes_instRepr___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Nat_Primes_instRepr() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_Primes_instRepr___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_Primes_inhabitedPrimes() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(2u);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_coeNat___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_coeNat___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nat_Primes_coeNat___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_Primes_coeNat() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_Primes_coeNat___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Nat_monoid_primePow___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_monoid_primePow___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Prime_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Sqrt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Prime_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Prime_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_Primes_instRepr = _init_lp_mathlib_Nat_Primes_instRepr();
lean_mark_persistent(lp_mathlib_Nat_Primes_instRepr);
lp_mathlib_Nat_Primes_inhabitedPrimes = _init_lp_mathlib_Nat_Primes_inhabitedPrimes();
lean_mark_persistent(lp_mathlib_Nat_Primes_inhabitedPrimes);
lp_mathlib_Nat_Primes_coeNat = _init_lp_mathlib_Nat_Primes_coeNat();
lean_mark_persistent(lp_mathlib_Nat_Primes_coeNat);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
