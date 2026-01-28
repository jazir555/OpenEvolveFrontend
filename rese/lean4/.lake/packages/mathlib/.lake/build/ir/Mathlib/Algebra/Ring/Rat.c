// Lean compiler output
// Module: Mathlib.Algebra.Ring.Rat
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Units.Basic public import Mathlib.Algebra.Ring.Basic public import Mathlib.Algebra.Ring.Int.Defs public import Mathlib.Data.Rat.Defs public import Mathlib.Algebra.Group.Nat.Defs
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
lean_object* l_Rat_div___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_commRing___closed__2;
static lean_object* lp_mathlib_Rat_commRing___closed__3;
static lean_object* lp_mathlib_Rat_commGroupWithZero___closed__0;
lean_object* l_Rat_mul___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_commGroupWithZero___closed__4;
lean_object* l_Rat_instNatCast___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__0___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_commRing___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commSemiring;
lean_object* l_instNatCastInt___lam__0(lean_object*);
static lean_object* lp_mathlib_Rat_commGroupWithZero___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__2(lean_object*, lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
extern lean_object* lp_mathlib_Rat_commMonoid;
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* l_Rat_mul(lean_object*, lean_object*);
lean_object* l_Rat_sub(lean_object*, lean_object*);
lean_object* l_Rat_inv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero___lam__0(lean_object*, lean_object*);
lean_object* l_Rat_pow(lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_commGroupWithZero___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero;
static lean_object* lp_mathlib_Rat_commRing___closed__1;
static lean_object* lp_mathlib_Rat_commRing___closed__4;
lean_object* l_Rat_zpow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__1(lean_object*);
static lean_object* lp_mathlib_Rat_commGroupWithZero___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Rat_semiring;
lean_object* l_Rat_ofInt(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing;
extern lean_object* lp_mathlib_Rat_addCommGroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Rat_neg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_Rat_pow(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = l_instNatCastInt___lam__0(x_1);
x_3 = l_Rat_ofInt(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = l_Rat_ofInt(x_1);
x_4 = l_Rat_mul(x_3, x_2);
lean_dec_ref(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Rat_commRing___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_ofInt), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_commRing___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_commRing___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = l_Rat_instNatCast___lam__0(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_commRing___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_neg), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_commRing___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_sub), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Rat_commRing___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_commRing() {
_start:
{
lean_object* x_1; uint8_t x_2; 
x_1 = lp_mathlib_Rat_addCommGroup;
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 3);
lean_dec(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_dec(x_5);
x_6 = lean_ctor_get(x_1, 1);
lean_dec(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Rat_commRing___lam__0___boxed), 2, 0);
x_8 = lp_mathlib_Rat_commRing___closed__0;
x_9 = lean_alloc_closure((void*)(lp_mathlib_Rat_commRing___lam__1), 1, 0);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Rat_commRing___lam__2), 2, 0);
x_11 = lp_mathlib_Rat_commRing___closed__1;
x_12 = lp_mathlib_Rat_commRing___closed__2;
x_13 = lp_mathlib_Rat_commRing___closed__3;
x_14 = lp_mathlib_Rat_commRing___closed__4;
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_3);
lean_ctor_set(x_15, 1, x_11);
lean_ctor_set(x_1, 3, x_7);
lean_ctor_set(x_1, 2, x_9);
lean_ctor_set(x_1, 1, x_12);
lean_ctor_set(x_1, 0, x_15);
x_16 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_16, 0, x_1);
lean_ctor_set(x_16, 1, x_13);
lean_ctor_set(x_16, 2, x_14);
lean_ctor_set(x_16, 3, x_10);
lean_ctor_set(x_16, 4, x_8);
return x_16;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_17 = lean_ctor_get(x_1, 0);
lean_inc(x_17);
lean_dec(x_1);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Rat_commRing___lam__0___boxed), 2, 0);
x_19 = lp_mathlib_Rat_commRing___closed__0;
x_20 = lean_alloc_closure((void*)(lp_mathlib_Rat_commRing___lam__1), 1, 0);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Rat_commRing___lam__2), 2, 0);
x_22 = lp_mathlib_Rat_commRing___closed__1;
x_23 = lp_mathlib_Rat_commRing___closed__2;
x_24 = lp_mathlib_Rat_commRing___closed__3;
x_25 = lp_mathlib_Rat_commRing___closed__4;
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_17);
lean_ctor_set(x_26, 1, x_22);
x_27 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_23);
lean_ctor_set(x_27, 2, x_20);
lean_ctor_set(x_27, 3, x_18);
x_28 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set(x_28, 1, x_24);
lean_ctor_set(x_28, 2, x_25);
lean_ctor_set(x_28, 3, x_21);
lean_ctor_set(x_28, 4, x_19);
return x_28;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_Rat_zpow(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_commGroupWithZero___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_commRing;
x_2 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_commGroupWithZero___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_commGroupWithZero___closed__0;
x_2 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_commGroupWithZero___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_commGroupWithZero___closed__1;
x_2 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_commGroupWithZero___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_inv), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_commGroupWithZero___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_div___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Rat_commGroupWithZero___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_commGroupWithZero() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; 
x_1 = lp_mathlib_Rat_commMonoid;
x_2 = lp_mathlib_Rat_commGroupWithZero___closed__2;
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 0);
lean_dec(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Rat_commGroupWithZero___lam__0___boxed), 2, 0);
x_6 = lp_mathlib_Rat_commGroupWithZero___closed__3;
x_7 = lp_mathlib_Rat_commGroupWithZero___closed__4;
lean_ctor_set(x_2, 0, x_1);
x_8 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_8, 0, x_2);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
lean_ctor_set(x_8, 3, x_5);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
lean_dec(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Rat_commGroupWithZero___lam__0___boxed), 2, 0);
x_11 = lp_mathlib_Rat_commGroupWithZero___closed__3;
x_12 = lp_mathlib_Rat_commGroupWithZero___closed__4;
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_1);
lean_ctor_set(x_13, 1, x_9);
x_14 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_11);
lean_ctor_set(x_14, 2, x_12);
lean_ctor_set(x_14, 3, x_10);
return x_14;
}
}
}
static lean_object* _init_lp_mathlib_Rat_commSemiring() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_commRing;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_semiring() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_commSemiring;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_commRing___closed__0 = _init_lp_mathlib_Rat_commRing___closed__0();
lean_mark_persistent(lp_mathlib_Rat_commRing___closed__0);
lp_mathlib_Rat_commRing___closed__1 = _init_lp_mathlib_Rat_commRing___closed__1();
lean_mark_persistent(lp_mathlib_Rat_commRing___closed__1);
lp_mathlib_Rat_commRing___closed__2 = _init_lp_mathlib_Rat_commRing___closed__2();
lean_mark_persistent(lp_mathlib_Rat_commRing___closed__2);
lp_mathlib_Rat_commRing___closed__3 = _init_lp_mathlib_Rat_commRing___closed__3();
lean_mark_persistent(lp_mathlib_Rat_commRing___closed__3);
lp_mathlib_Rat_commRing___closed__4 = _init_lp_mathlib_Rat_commRing___closed__4();
lean_mark_persistent(lp_mathlib_Rat_commRing___closed__4);
lp_mathlib_Rat_commRing = _init_lp_mathlib_Rat_commRing();
lean_mark_persistent(lp_mathlib_Rat_commRing);
lp_mathlib_Rat_commGroupWithZero___closed__0 = _init_lp_mathlib_Rat_commGroupWithZero___closed__0();
lean_mark_persistent(lp_mathlib_Rat_commGroupWithZero___closed__0);
lp_mathlib_Rat_commGroupWithZero___closed__1 = _init_lp_mathlib_Rat_commGroupWithZero___closed__1();
lean_mark_persistent(lp_mathlib_Rat_commGroupWithZero___closed__1);
lp_mathlib_Rat_commGroupWithZero___closed__2 = _init_lp_mathlib_Rat_commGroupWithZero___closed__2();
lean_mark_persistent(lp_mathlib_Rat_commGroupWithZero___closed__2);
lp_mathlib_Rat_commGroupWithZero___closed__3 = _init_lp_mathlib_Rat_commGroupWithZero___closed__3();
lean_mark_persistent(lp_mathlib_Rat_commGroupWithZero___closed__3);
lp_mathlib_Rat_commGroupWithZero___closed__4 = _init_lp_mathlib_Rat_commGroupWithZero___closed__4();
lean_mark_persistent(lp_mathlib_Rat_commGroupWithZero___closed__4);
lp_mathlib_Rat_commGroupWithZero = _init_lp_mathlib_Rat_commGroupWithZero();
lean_mark_persistent(lp_mathlib_Rat_commGroupWithZero);
lp_mathlib_Rat_commSemiring = _init_lp_mathlib_Rat_commSemiring();
lean_mark_persistent(lp_mathlib_Rat_commSemiring);
lp_mathlib_Rat_semiring = _init_lp_mathlib_Rat_semiring();
lean_mark_persistent(lp_mathlib_Rat_semiring);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
