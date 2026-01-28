// Lean compiler output
// Module: Mathlib.Algebra.GCDMonoid.Nat
// Imports: public import Init public import Mathlib.Algebra.GCDMonoid.Basic public import Mathlib.Algebra.Order.Group.Unbundled.Int public import Mathlib.Algebra.Ring.Int.Units public import Mathlib.Algebra.GroupWithZero.Nat
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
lean_object* l_Int_gcd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNormalizedGCDMonoidNat;
lean_object* l_Int_lcm(lean_object*, lean_object*);
static lean_object* lp_mathlib_instGCDMonoidNat___closed__0;
static lean_object* lp_mathlib_Int_instNormalizedGCDMonoid___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat___lam__1(lean_object*);
static lean_object* lp_mathlib_Int_normalizationMonoid___lam__0___closed__4;
static lean_object* lp_mathlib_Int_normalizationMonoid___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat;
LEAN_EXPORT lean_object* lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_normalizationMonoid___lam__0___boxed(lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* l_Nat_gcd___boxed(lean_object*, lean_object*);
lean_object* l_instNatCastInt___lam__0(lean_object*);
static lean_object* lp_mathlib_Int_normalizationMonoid___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat___lam__0(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___at___00associatesIntEquivNat_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_normalizationMonoid;
LEAN_EXPORT lean_object* lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0___lam__0(lean_object*);
lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instNormalizedGCDMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid;
lean_object* l_Nat_lcm___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_instNormalizedGCDMonoidNat___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instGCDMonoidNat;
LEAN_EXPORT lean_object* lp_mathlib_Int_normalizationMonoid___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_normalizationMonoid___lam__0___closed__2;
lean_object* lean_nat_abs(lean_object*);
extern lean_object* lp_mathlib_Nat_instCommMonoidWithZero;
lean_object* lean_int_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__1___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_normalizationMonoid___lam__0___closed__0;
static lean_object* lp_mathlib_instGCDMonoidNat___closed__2;
static lean_object* lp_mathlib_instNormalizedGCDMonoidNat___closed__0;
static lean_object* lp_mathlib_Int_instNormalizedGCDMonoid___closed__0;
static lean_object* lp_mathlib_instGCDMonoidNat___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00associatesIntEquivNat_spec__2(lean_object*);
lean_object* lean_int_neg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__0___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_instGCDMonoidNat___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_gcd___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instGCDMonoidNat___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_lcm___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instGCDMonoidNat___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instGCDMonoidNat___closed__1;
x_2 = lp_mathlib_instGCDMonoidNat___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instGCDMonoidNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instGCDMonoidNat___closed__2;
return x_1;
}
}
static lean_object* _init_lp_mathlib_instNormalizedGCDMonoidNat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instCommMonoidWithZero;
x_2 = lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instNormalizedGCDMonoidNat___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instGCDMonoidNat;
x_2 = lp_mathlib_instNormalizedGCDMonoidNat___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instNormalizedGCDMonoidNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instNormalizedGCDMonoidNat___closed__1;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_normalizationMonoid___lam__0___closed__1;
x_2 = lean_int_neg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_normalizationMonoid___lam__0___closed__2;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_normalizationMonoid___lam__0___closed__1;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_normalizationMonoid___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lp_mathlib_Int_normalizationMonoid___lam__0___closed__0;
x_3 = lean_int_dec_le(x_2, x_1);
if (x_3 == 0)
{
lean_object* x_4; 
x_4 = lp_mathlib_Int_normalizationMonoid___lam__0___closed__3;
return x_4;
}
else
{
lean_object* x_5; 
x_5 = lp_mathlib_Int_normalizationMonoid___lam__0___closed__4;
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_normalizationMonoid___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Int_normalizationMonoid___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_normalizationMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Int_normalizationMonoid___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = l_Int_lcm(x_1, x_2);
x_4 = l_instNatCastInt___lam__0(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = l_Int_gcd(x_1, x_2);
x_4 = l_instNatCastInt___lam__0(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Int_instGCDMonoid___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Int_instGCDMonoid___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Int_instGCDMonoid() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Int_instGCDMonoid___lam__0___boxed), 2, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Int_instGCDMonoid___lam__1___boxed), 2, 0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Int_instNormalizedGCDMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Int_normalizationMonoid___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instNormalizedGCDMonoid___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Int_instGCDMonoid;
x_2 = lp_mathlib_Int_instNormalizedGCDMonoid___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Int_instNormalizedGCDMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instNormalizedGCDMonoid___closed__1;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lp_mathlib_Int_normalizationMonoid___lam__0___closed__0;
x_3 = lean_int_dec_le(x_2, x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Int_normalizationMonoid___lam__0___closed__2;
x_5 = lean_int_mul(x_1, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_Int_normalizationMonoid___lam__0___closed__1;
x_7 = lean_int_mul(x_1, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___at___00associatesIntEquivNat_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0;
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00associatesIntEquivNat_spec__2(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Associates_out___at___00associatesIntEquivNat_spec__0(x_1);
x_3 = lean_nat_abs(x_2);
lean_dec(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_associatesIntEquivNat() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_associatesIntEquivNat___lam__0), 1, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_associatesIntEquivNat___lam__1), 1, 0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instGCDMonoidNat___closed__0 = _init_lp_mathlib_instGCDMonoidNat___closed__0();
lean_mark_persistent(lp_mathlib_instGCDMonoidNat___closed__0);
lp_mathlib_instGCDMonoidNat___closed__1 = _init_lp_mathlib_instGCDMonoidNat___closed__1();
lean_mark_persistent(lp_mathlib_instGCDMonoidNat___closed__1);
lp_mathlib_instGCDMonoidNat___closed__2 = _init_lp_mathlib_instGCDMonoidNat___closed__2();
lean_mark_persistent(lp_mathlib_instGCDMonoidNat___closed__2);
lp_mathlib_instGCDMonoidNat = _init_lp_mathlib_instGCDMonoidNat();
lean_mark_persistent(lp_mathlib_instGCDMonoidNat);
lp_mathlib_instNormalizedGCDMonoidNat___closed__0 = _init_lp_mathlib_instNormalizedGCDMonoidNat___closed__0();
lean_mark_persistent(lp_mathlib_instNormalizedGCDMonoidNat___closed__0);
lp_mathlib_instNormalizedGCDMonoidNat___closed__1 = _init_lp_mathlib_instNormalizedGCDMonoidNat___closed__1();
lean_mark_persistent(lp_mathlib_instNormalizedGCDMonoidNat___closed__1);
lp_mathlib_instNormalizedGCDMonoidNat = _init_lp_mathlib_instNormalizedGCDMonoidNat();
lean_mark_persistent(lp_mathlib_instNormalizedGCDMonoidNat);
lp_mathlib_Int_normalizationMonoid___lam__0___closed__0 = _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Int_normalizationMonoid___lam__0___closed__0);
lp_mathlib_Int_normalizationMonoid___lam__0___closed__1 = _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_Int_normalizationMonoid___lam__0___closed__1);
lp_mathlib_Int_normalizationMonoid___lam__0___closed__2 = _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__2();
lean_mark_persistent(lp_mathlib_Int_normalizationMonoid___lam__0___closed__2);
lp_mathlib_Int_normalizationMonoid___lam__0___closed__3 = _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__3();
lean_mark_persistent(lp_mathlib_Int_normalizationMonoid___lam__0___closed__3);
lp_mathlib_Int_normalizationMonoid___lam__0___closed__4 = _init_lp_mathlib_Int_normalizationMonoid___lam__0___closed__4();
lean_mark_persistent(lp_mathlib_Int_normalizationMonoid___lam__0___closed__4);
lp_mathlib_Int_normalizationMonoid = _init_lp_mathlib_Int_normalizationMonoid();
lean_mark_persistent(lp_mathlib_Int_normalizationMonoid);
lp_mathlib_Int_instGCDMonoid = _init_lp_mathlib_Int_instGCDMonoid();
lean_mark_persistent(lp_mathlib_Int_instGCDMonoid);
lp_mathlib_Int_instNormalizedGCDMonoid___closed__0 = _init_lp_mathlib_Int_instNormalizedGCDMonoid___closed__0();
lean_mark_persistent(lp_mathlib_Int_instNormalizedGCDMonoid___closed__0);
lp_mathlib_Int_instNormalizedGCDMonoid___closed__1 = _init_lp_mathlib_Int_instNormalizedGCDMonoid___closed__1();
lean_mark_persistent(lp_mathlib_Int_instNormalizedGCDMonoid___closed__1);
lp_mathlib_Int_instNormalizedGCDMonoid = _init_lp_mathlib_Int_instNormalizedGCDMonoid();
lean_mark_persistent(lp_mathlib_Int_instNormalizedGCDMonoid);
lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0 = _init_lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0();
lean_mark_persistent(lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0);
lp_mathlib_associatesIntEquivNat = _init_lp_mathlib_associatesIntEquivNat();
lean_mark_persistent(lp_mathlib_associatesIntEquivNat);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
