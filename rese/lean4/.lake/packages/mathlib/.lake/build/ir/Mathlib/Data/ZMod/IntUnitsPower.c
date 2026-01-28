// Lean compiler output
// Module: Mathlib.Data.ZMod.IntUnitsPower
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Divisibility public import Mathlib.Data.Int.Order.Units public import Mathlib.Data.ZMod.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt;
static lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1;
lean_object* lp_mathlib_Additive_toMul(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* l_Int_pow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt;
LEAN_EXPORT lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_val(lean_object*, lean_object*);
static lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0;
static lean_object* _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Additive_toMul(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Additive_ofMul(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_7 = lean_ctor_get(x_5, 0);
x_8 = lean_ctor_get(x_5, 1);
x_9 = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1;
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
x_11 = lean_unsigned_to_nat(2u);
x_12 = lp_mathlib_ZMod_val(x_11, x_1);
x_13 = l_Int_pow(x_7, x_12);
lean_dec(x_7);
x_14 = l_Int_pow(x_8, x_12);
lean_dec(x_12);
lean_dec(x_8);
lean_ctor_set(x_5, 1, x_14);
lean_ctor_set(x_5, 0, x_13);
x_15 = lean_apply_1(x_10, x_5);
return x_15;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_16 = lean_ctor_get(x_5, 0);
x_17 = lean_ctor_get(x_5, 1);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_5);
x_18 = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1;
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
x_20 = lean_unsigned_to_nat(2u);
x_21 = lp_mathlib_ZMod_val(x_20, x_1);
x_22 = l_Int_pow(x_16, x_21);
lean_dec(x_16);
x_23 = l_Int_pow(x_17, x_21);
lean_dec(x_21);
lean_dec(x_17);
x_24 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_23);
x_25 = lean_apply_1(x_19, x_24);
return x_25;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, x_3);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
x_10 = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1;
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lp_mathlib_ZMod_val(x_1, x_2);
x_13 = l_Int_pow(x_8, x_12);
lean_dec(x_8);
x_14 = l_Int_pow(x_9, x_12);
lean_dec(x_12);
lean_dec(x_9);
lean_ctor_set(x_6, 1, x_14);
lean_ctor_set(x_6, 0, x_13);
x_15 = lean_apply_1(x_11, x_6);
return x_15;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_16 = lean_ctor_get(x_6, 0);
x_17 = lean_ctor_get(x_6, 1);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_6);
x_18 = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1;
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
x_20 = lp_mathlib_ZMod_val(x_1, x_2);
x_21 = l_Int_pow(x_16, x_20);
lean_dec(x_16);
x_22 = l_Int_pow(x_17, x_20);
lean_dec(x_20);
lean_dec(x_17);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_apply_1(x_19, x_23);
return x_24;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_alloc_closure((void*)(lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0;
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_apply_1(x_5, x_2);
x_9 = lean_apply_2(x_1, x_3, x_8);
x_10 = lean_apply_1(x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Int_instUnitsPow___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Int_instUnitsPow___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Int_instUnitsPow(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Order_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Order_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0 = _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0);
lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1 = _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1);
lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt = _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt();
lean_mark_persistent(lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt);
lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt = _init_lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt();
lean_mark_persistent(lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
