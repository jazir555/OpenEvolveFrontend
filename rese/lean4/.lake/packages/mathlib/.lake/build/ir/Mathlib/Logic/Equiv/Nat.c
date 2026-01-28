// Lean compiler output
// Module: Mathlib.Logic.Equiv.Nat
// Imports: public import Init public import Mathlib.Data.Nat.Bits public import Mathlib.Data.Nat.Pairing
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
lean_object* lp_mathlib_Nat_bit(uint8_t, lean_object*);
lean_object* lp_mathlib_Equiv_boolProdEquivSum(lean_object*);
extern lean_object* lp_mathlib_Equiv_intEquivNatSumNat;
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_natSumNatEquivNat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEquivOfEquivNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEquivOfEquivNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__0(lean_object*);
static lean_object* lp_mathlib_Equiv_natSumNatEquivNat___closed__1;
lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_natSumNatEquivNat___closed__2;
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_natSumNatEquivNat;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_intEquivNat;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__0___boxed(lean_object*);
static lean_object* lp_mathlib_Equiv_intEquivNat___closed__0;
extern lean_object* lp_mathlib_Nat_pairEquiv;
lean_object* lp_mathlib_Nat_boddDiv2___boxed(lean_object*);
static lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___closed__0;
static lean_object* _init_lp_mathlib_Equiv_boolProdNatEquivNat___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_boddDiv2___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_unbox(x_2);
x_5 = lp_mathlib_Nat_bit(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_boolProdNatEquivNat___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Equiv_boolProdNatEquivNat() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Equiv_boolProdNatEquivNat___lam__0___boxed), 1, 0);
x_2 = lp_mathlib_Equiv_boolProdNatEquivNat___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_boolProdEquivSum(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Equiv_natSumNatEquivNat___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Equiv_boolProdNatEquivNat;
x_2 = lp_mathlib_Equiv_natSumNatEquivNat___closed__1;
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumNatEquivNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_natSumNatEquivNat___closed__2;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_intEquivNat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Equiv_natSumNatEquivNat;
x_2 = lp_mathlib_Equiv_intEquivNatSumNat;
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Equiv_intEquivNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_intEquivNat___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEquivOfEquivNat___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref_n(x_1, 2);
x_2 = lp_mathlib_Equiv_prodCongr___redArg(x_1, x_1);
x_3 = lp_mathlib_Nat_pairEquiv;
x_4 = lp_mathlib_Equiv_trans___redArg(x_2, x_3);
x_5 = lp_mathlib_Equiv_symm___redArg(x_1);
x_6 = lp_mathlib_Equiv_trans___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEquivOfEquivNat(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_prodEquivOfEquivNat___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Bits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Pairing(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Bits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Pairing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_boolProdNatEquivNat___closed__0 = _init_lp_mathlib_Equiv_boolProdNatEquivNat___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_boolProdNatEquivNat___closed__0);
lp_mathlib_Equiv_boolProdNatEquivNat = _init_lp_mathlib_Equiv_boolProdNatEquivNat();
lean_mark_persistent(lp_mathlib_Equiv_boolProdNatEquivNat);
lp_mathlib_Equiv_natSumNatEquivNat___closed__0 = _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_natSumNatEquivNat___closed__0);
lp_mathlib_Equiv_natSumNatEquivNat___closed__1 = _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_natSumNatEquivNat___closed__1);
lp_mathlib_Equiv_natSumNatEquivNat___closed__2 = _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__2();
lean_mark_persistent(lp_mathlib_Equiv_natSumNatEquivNat___closed__2);
lp_mathlib_Equiv_natSumNatEquivNat = _init_lp_mathlib_Equiv_natSumNatEquivNat();
lean_mark_persistent(lp_mathlib_Equiv_natSumNatEquivNat);
lp_mathlib_Equiv_intEquivNat___closed__0 = _init_lp_mathlib_Equiv_intEquivNat___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_intEquivNat___closed__0);
lp_mathlib_Equiv_intEquivNat = _init_lp_mathlib_Equiv_intEquivNat();
lean_mark_persistent(lp_mathlib_Equiv_intEquivNat);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
