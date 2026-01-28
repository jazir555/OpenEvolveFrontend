// Lean compiler output
// Module: Mathlib.Algebra.Ring.Int.Defs
// Imports: public import Init public import Mathlib.Algebra.CharZero.Defs public import Mathlib.Algebra.Ring.Defs public import Mathlib.Algebra.Group.Int.Defs public import Mathlib.Data.Int.Basic public import Mathlib.Data.Int.Cast.Basic public import Mathlib.Algebra.Ring.GrindInstances
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
extern lean_object* lp_mathlib_Int_instAddCommGroup;
extern lean_object* lp_mathlib_Int_instCommMonoid;
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__0___boxed(lean_object*);
static lean_object* lp_mathlib_Int_instDistrib___closed__1;
static lean_object* lp_mathlib_Int_instDistrib___closed__2;
extern lean_object* lp_mathlib_Int_instCommSemigroup;
lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object*);
lean_object* l_Int_sub___boxed(lean_object*, lean_object*);
lean_object* l_instNatCastInt___lam__0(lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
static lean_object* lp_mathlib_Int_instCommRing___closed__4;
lean_object* l_Int_pow(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instDistrib;
static lean_object* lp_mathlib_Int_instDistrib___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing;
LEAN_EXPORT lean_object* lp_mathlib_Int_instRing;
static lean_object* lp_mathlib_Int_instCommRing___closed__3;
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instCancelCommMonoidWithZero;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__1___boxed(lean_object*, lean_object*);
lean_object* l_Int_neg___boxed(lean_object*);
lean_object* l_Int_mul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__1(lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_instCommRing___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommSemiring;
static lean_object* lp_mathlib_Int_instCommRing___closed__1;
static lean_object* lp_mathlib_Int_instCommRing___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Int_instSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_Int_pow(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Int_instCommRing___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instCommMonoid;
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instCommRing___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instCommRing___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instNatCastInt___lam__0), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instCommRing___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_neg___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instCommRing___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_sub___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Int_instCommRing___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Int_instCommRing___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Int_instCommRing() {
_start:
{
lean_object* x_1; uint8_t x_2; 
x_1 = lp_mathlib_Int_instAddCommGroup;
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 3);
lean_dec(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_dec(x_5);
x_6 = lean_ctor_get(x_1, 1);
lean_dec(x_6);
x_7 = lp_mathlib_Int_instCommRing___closed__0;
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 1);
lean_dec(x_10);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Int_instCommRing___lam__0___boxed), 1, 0);
x_12 = lp_mathlib_Int_instCommRing___closed__1;
x_13 = lean_alloc_closure((void*)(lp_mathlib_Int_instCommRing___lam__1___boxed), 2, 0);
x_14 = lp_mathlib_Int_instCommRing___closed__2;
x_15 = lp_mathlib_Int_instCommSemigroup;
x_16 = lp_mathlib_Int_instCommRing___closed__3;
x_17 = lp_mathlib_Int_instCommRing___closed__4;
lean_ctor_set(x_7, 1, x_15);
lean_ctor_set(x_7, 0, x_3);
lean_ctor_set(x_1, 3, x_13);
lean_ctor_set(x_1, 2, x_14);
lean_ctor_set(x_1, 1, x_9);
lean_ctor_set(x_1, 0, x_7);
x_18 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_18, 0, x_1);
lean_ctor_set(x_18, 1, x_16);
lean_ctor_set(x_18, 2, x_17);
lean_ctor_set(x_18, 3, x_12);
lean_ctor_set(x_18, 4, x_11);
return x_18;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_19 = lean_ctor_get(x_7, 0);
lean_inc(x_19);
lean_dec(x_7);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Int_instCommRing___lam__0___boxed), 1, 0);
x_21 = lp_mathlib_Int_instCommRing___closed__1;
x_22 = lean_alloc_closure((void*)(lp_mathlib_Int_instCommRing___lam__1___boxed), 2, 0);
x_23 = lp_mathlib_Int_instCommRing___closed__2;
x_24 = lp_mathlib_Int_instCommSemigroup;
x_25 = lp_mathlib_Int_instCommRing___closed__3;
x_26 = lp_mathlib_Int_instCommRing___closed__4;
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_3);
lean_ctor_set(x_27, 1, x_24);
lean_ctor_set(x_1, 3, x_22);
lean_ctor_set(x_1, 2, x_23);
lean_ctor_set(x_1, 1, x_19);
lean_ctor_set(x_1, 0, x_27);
x_28 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_28, 0, x_1);
lean_ctor_set(x_28, 1, x_25);
lean_ctor_set(x_28, 2, x_26);
lean_ctor_set(x_28, 3, x_21);
lean_ctor_set(x_28, 4, x_20);
return x_28;
}
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_29 = lean_ctor_get(x_1, 0);
lean_inc(x_29);
lean_dec(x_1);
x_30 = lp_mathlib_Int_instCommRing___closed__0;
x_31 = lean_ctor_get(x_30, 0);
lean_inc(x_31);
if (lean_is_exclusive(x_30)) {
 lean_ctor_release(x_30, 0);
 lean_ctor_release(x_30, 1);
 x_32 = x_30;
} else {
 lean_dec_ref(x_30);
 x_32 = lean_box(0);
}
x_33 = lean_alloc_closure((void*)(lp_mathlib_Int_instCommRing___lam__0___boxed), 1, 0);
x_34 = lp_mathlib_Int_instCommRing___closed__1;
x_35 = lean_alloc_closure((void*)(lp_mathlib_Int_instCommRing___lam__1___boxed), 2, 0);
x_36 = lp_mathlib_Int_instCommRing___closed__2;
x_37 = lp_mathlib_Int_instCommSemigroup;
x_38 = lp_mathlib_Int_instCommRing___closed__3;
x_39 = lp_mathlib_Int_instCommRing___closed__4;
if (lean_is_scalar(x_32)) {
 x_40 = lean_alloc_ctor(0, 2, 0);
} else {
 x_40 = x_32;
}
lean_ctor_set(x_40, 0, x_29);
lean_ctor_set(x_40, 1, x_37);
x_41 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_31);
lean_ctor_set(x_41, 2, x_36);
lean_ctor_set(x_41, 3, x_35);
x_42 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_42, 0, x_41);
lean_ctor_set(x_42, 1, x_38);
lean_ctor_set(x_42, 2, x_39);
lean_ctor_set(x_42, 3, x_34);
lean_ctor_set(x_42, 4, x_33);
return x_42;
}
}
}
static lean_object* _init_lp_mathlib_Int_instCancelCommMonoidWithZero() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Int_instCommRing;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Int_instCommSemiring() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instCommRing;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instSemiring() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instCommSemiring;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instRing() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instCommRing;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instDistrib___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instCommRing;
x_2 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instDistrib___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instDistrib___closed__0;
x_2 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instDistrib___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instDistrib___closed__1;
x_2 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instDistrib() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instDistrib___closed__2;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_instCommRing___closed__0 = _init_lp_mathlib_Int_instCommRing___closed__0();
lean_mark_persistent(lp_mathlib_Int_instCommRing___closed__0);
lp_mathlib_Int_instCommRing___closed__1 = _init_lp_mathlib_Int_instCommRing___closed__1();
lean_mark_persistent(lp_mathlib_Int_instCommRing___closed__1);
lp_mathlib_Int_instCommRing___closed__2 = _init_lp_mathlib_Int_instCommRing___closed__2();
lean_mark_persistent(lp_mathlib_Int_instCommRing___closed__2);
lp_mathlib_Int_instCommRing___closed__3 = _init_lp_mathlib_Int_instCommRing___closed__3();
lean_mark_persistent(lp_mathlib_Int_instCommRing___closed__3);
lp_mathlib_Int_instCommRing___closed__4 = _init_lp_mathlib_Int_instCommRing___closed__4();
lean_mark_persistent(lp_mathlib_Int_instCommRing___closed__4);
lp_mathlib_Int_instCommRing = _init_lp_mathlib_Int_instCommRing();
lean_mark_persistent(lp_mathlib_Int_instCommRing);
lp_mathlib_Int_instCancelCommMonoidWithZero = _init_lp_mathlib_Int_instCancelCommMonoidWithZero();
lean_mark_persistent(lp_mathlib_Int_instCancelCommMonoidWithZero);
lp_mathlib_Int_instCommSemiring = _init_lp_mathlib_Int_instCommSemiring();
lean_mark_persistent(lp_mathlib_Int_instCommSemiring);
lp_mathlib_Int_instSemiring = _init_lp_mathlib_Int_instSemiring();
lean_mark_persistent(lp_mathlib_Int_instSemiring);
lp_mathlib_Int_instRing = _init_lp_mathlib_Int_instRing();
lean_mark_persistent(lp_mathlib_Int_instRing);
lp_mathlib_Int_instDistrib___closed__0 = _init_lp_mathlib_Int_instDistrib___closed__0();
lean_mark_persistent(lp_mathlib_Int_instDistrib___closed__0);
lp_mathlib_Int_instDistrib___closed__1 = _init_lp_mathlib_Int_instDistrib___closed__1();
lean_mark_persistent(lp_mathlib_Int_instDistrib___closed__1);
lp_mathlib_Int_instDistrib___closed__2 = _init_lp_mathlib_Int_instDistrib___closed__2();
lean_mark_persistent(lp_mathlib_Int_instDistrib___closed__2);
lp_mathlib_Int_instDistrib = _init_lp_mathlib_Int_instDistrib();
lean_mark_persistent(lp_mathlib_Int_instDistrib);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
