// Lean compiler output
// Module: Mathlib.Algebra.Algebra.ZMod
// Imports: public import Init public import Mathlib.Algebra.Algebra.Defs public import Mathlib.Data.ZMod.Basic
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_cast___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_castHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_ZMod_cast___redArg(x_1, x_2, x_4);
x_7 = lean_apply_2(x_3, x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ZMod_algebra_x27___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
lean_inc_ref(x_2);
x_7 = lp_mathlib_Ring_toNonAssocRing___redArg(x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_8);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_9);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
lean_dec(x_13);
lean_inc_ref(x_2);
x_14 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_2);
lean_inc(x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ZMod_algebra_x27___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_15, 0, x_14);
lean_closure_set(x_15, 1, x_3);
lean_closure_set(x_15, 2, x_12);
x_16 = lp_mathlib_ZMod_castHom___redArg(x_3, x_2);
lean_ctor_set(x_10, 1, x_16);
lean_ctor_set(x_10, 0, x_15);
return x_10;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_17 = lean_ctor_get(x_10, 0);
lean_inc(x_17);
lean_dec(x_10);
lean_inc_ref(x_2);
x_18 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_2);
lean_inc(x_3);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ZMod_algebra_x27___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_19, 0, x_18);
lean_closure_set(x_19, 1, x_3);
lean_closure_set(x_19, 2, x_17);
x_20 = lp_mathlib_ZMod_castHom___redArg(x_3, x_2);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
lean_dec(x_9);
lean_inc_ref(x_1);
x_10 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
lean_inc(x_2);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ZMod_algebra_x27___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_11, 0, x_10);
lean_closure_set(x_11, 1, x_2);
lean_closure_set(x_11, 2, x_8);
x_12 = lp_mathlib_ZMod_castHom___redArg(x_2, x_1);
lean_ctor_set(x_6, 1, x_12);
lean_ctor_set(x_6, 0, x_11);
return x_6;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_13 = lean_ctor_get(x_6, 0);
lean_inc(x_13);
lean_dec(x_6);
lean_inc_ref(x_1);
x_14 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
lean_inc(x_2);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ZMod_algebra_x27___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_15, 0, x_14);
lean_closure_set(x_15, 1, x_2);
lean_closure_set(x_15, 2, x_13);
x_16 = lp_mathlib_ZMod_castHom___redArg(x_2, x_1);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ZMod_algebra_x27(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_ZMod_cast___redArg(x_1, x_2, x_4);
x_7 = lean_apply_2(x_3, x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ZMod_algebra___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
lean_inc_ref(x_2);
x_5 = lp_mathlib_Ring_toNonAssocRing___redArg(x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_6);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_7);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_8, 0);
x_11 = lean_ctor_get(x_8, 1);
lean_dec(x_11);
lean_inc_ref(x_2);
x_12 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_2);
lean_inc(x_3);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ZMod_algebra___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_13, 0, x_12);
lean_closure_set(x_13, 1, x_3);
lean_closure_set(x_13, 2, x_10);
x_14 = lp_mathlib_ZMod_castHom___redArg(x_3, x_2);
lean_ctor_set(x_8, 1, x_14);
lean_ctor_set(x_8, 0, x_13);
return x_8;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_8, 0);
lean_inc(x_15);
lean_dec(x_8);
lean_inc_ref(x_2);
x_16 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_2);
lean_inc(x_3);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ZMod_algebra___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_17, 0, x_16);
lean_closure_set(x_17, 1, x_3);
lean_closure_set(x_17, 2, x_15);
x_18 = lp_mathlib_ZMod_castHom___redArg(x_3, x_2);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebra___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
lean_dec(x_9);
lean_inc_ref(x_1);
x_10 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
lean_inc(x_2);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ZMod_algebra___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_11, 0, x_10);
lean_closure_set(x_11, 1, x_2);
lean_closure_set(x_11, 2, x_8);
x_12 = lp_mathlib_ZMod_castHom___redArg(x_2, x_1);
lean_ctor_set(x_6, 1, x_12);
lean_ctor_set(x_6, 0, x_11);
return x_6;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_13 = lean_ctor_get(x_6, 0);
lean_inc(x_13);
lean_dec(x_6);
lean_inc_ref(x_1);
x_14 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
lean_inc(x_2);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ZMod_algebra___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_15, 0, x_14);
lean_closure_set(x_15, 1, x_2);
lean_closure_set(x_15, 2, x_13);
x_16 = lp_mathlib_ZMod_castHom___redArg(x_2, x_1);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
x_5 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_6 = lean_ctor_get(x_5, 2);
lean_inc(x_6);
lean_dec_ref(x_5);
lean_inc(x_2);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ZMod_algebraOfModule___redArg___lam__0), 3, 2);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_6);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_2);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ZMod_algebraOfModule___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ZMod_algebraOfModule(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_algebraOfModule___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ZMod_algebraOfModule___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_ZMod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
