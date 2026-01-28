// Lean compiler output
// Module: Mathlib.Algebra.Ring.NegOnePow
// Imports: public import Init public import Mathlib.Algebra.Ring.Int.Parity public import Mathlib.Algebra.Ring.Int.Units public import Mathlib.Data.ZMod.IntUnitsPower
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
static lean_object* lp_mathlib_Int_negOnePow___closed__0;
static lean_object* lp_mathlib_Int_negOnePow___closed__3;
static lean_object* lp_mathlib_Int_negOnePow___closed__2;
static lean_object* lp_mathlib_Int_negOnePow___closed__5;
lean_object* lp_mathlib_Additive_toMul(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_negOnePow___boxed(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* l_Int_pow(lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_negOnePow___closed__1;
lean_object* lean_nat_abs(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_negOnePow___closed__4;
lean_object* lean_int_neg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_negOnePow(lean_object*);
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Additive_ofMul(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_negOnePow___closed__1;
x_2 = lean_int_neg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_negOnePow___closed__2;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Additive_toMul(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_negOnePow(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_2 = lp_mathlib_Int_negOnePow___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_unsigned_to_nat(1u);
x_5 = lp_mathlib_Int_negOnePow___closed__3;
x_6 = lp_mathlib_Int_negOnePow___closed__4;
x_11 = lean_apply_1(x_3, x_5);
x_12 = lp_mathlib_Int_negOnePow___closed__5;
x_13 = lean_int_dec_lt(x_1, x_12);
if (x_13 == 0)
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_11);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_11, 0);
x_16 = lean_ctor_get(x_11, 1);
x_17 = lean_nat_abs(x_1);
x_18 = l_Int_pow(x_15, x_17);
lean_dec(x_15);
x_19 = l_Int_pow(x_16, x_17);
lean_dec(x_17);
lean_dec(x_16);
lean_ctor_set(x_11, 1, x_19);
lean_ctor_set(x_11, 0, x_18);
x_7 = x_11;
goto block_10;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_20 = lean_ctor_get(x_11, 0);
x_21 = lean_ctor_get(x_11, 1);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_11);
x_22 = lean_nat_abs(x_1);
x_23 = l_Int_pow(x_20, x_22);
lean_dec(x_20);
x_24 = l_Int_pow(x_21, x_22);
lean_dec(x_22);
lean_dec(x_21);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_23);
lean_ctor_set(x_25, 1, x_24);
x_7 = x_25;
goto block_10;
}
}
else
{
uint8_t x_26; 
x_26 = !lean_is_exclusive(x_11);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_27 = lean_ctor_get(x_11, 0);
x_28 = lean_ctor_get(x_11, 1);
x_29 = lean_nat_abs(x_1);
x_30 = lean_nat_sub(x_29, x_4);
lean_dec(x_29);
x_31 = lean_nat_add(x_30, x_4);
lean_dec(x_30);
x_32 = l_Int_pow(x_27, x_31);
lean_dec(x_27);
x_33 = l_Int_pow(x_28, x_31);
lean_dec(x_31);
lean_dec(x_28);
lean_ctor_set(x_11, 1, x_32);
lean_ctor_set(x_11, 0, x_33);
x_7 = x_11;
goto block_10;
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_34 = lean_ctor_get(x_11, 0);
x_35 = lean_ctor_get(x_11, 1);
lean_inc(x_35);
lean_inc(x_34);
lean_dec(x_11);
x_36 = lean_nat_abs(x_1);
x_37 = lean_nat_sub(x_36, x_4);
lean_dec(x_36);
x_38 = lean_nat_add(x_37, x_4);
lean_dec(x_37);
x_39 = l_Int_pow(x_34, x_38);
lean_dec(x_34);
x_40 = l_Int_pow(x_35, x_38);
lean_dec(x_38);
lean_dec(x_35);
x_41 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_39);
x_7 = x_41;
goto block_10;
}
}
block_10:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
x_9 = lean_apply_1(x_8, x_7);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_negOnePow___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Int_negOnePow(x_1);
lean_dec(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_NegOnePow(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_negOnePow___closed__0 = _init_lp_mathlib_Int_negOnePow___closed__0();
lean_mark_persistent(lp_mathlib_Int_negOnePow___closed__0);
lp_mathlib_Int_negOnePow___closed__1 = _init_lp_mathlib_Int_negOnePow___closed__1();
lean_mark_persistent(lp_mathlib_Int_negOnePow___closed__1);
lp_mathlib_Int_negOnePow___closed__2 = _init_lp_mathlib_Int_negOnePow___closed__2();
lean_mark_persistent(lp_mathlib_Int_negOnePow___closed__2);
lp_mathlib_Int_negOnePow___closed__3 = _init_lp_mathlib_Int_negOnePow___closed__3();
lean_mark_persistent(lp_mathlib_Int_negOnePow___closed__3);
lp_mathlib_Int_negOnePow___closed__4 = _init_lp_mathlib_Int_negOnePow___closed__4();
lean_mark_persistent(lp_mathlib_Int_negOnePow___closed__4);
lp_mathlib_Int_negOnePow___closed__5 = _init_lp_mathlib_Int_negOnePow___closed__5();
lean_mark_persistent(lp_mathlib_Int_negOnePow___closed__5);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
