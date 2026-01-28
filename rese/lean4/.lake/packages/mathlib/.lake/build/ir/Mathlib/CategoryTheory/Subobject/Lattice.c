// Lean compiler output
// Module: Mathlib.CategoryTheory.Subobject.Lattice
// Imports: public import Init public import Mathlib.CategoryTheory.Functor.Currying public import Mathlib.CategoryTheory.Subobject.FactorThru public import Mathlib.CategoryTheory.Subobject.WellPowered public import Mathlib.Data.Finset.Lattice.Fold
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_instTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_mapTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_instInhabited(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_instInhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_leTop(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_MonoOver_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_leTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_mapTop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_orderTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_instInhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_leTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_instTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_orderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_instTop___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
lean_inc(x_2);
x_4 = lean_apply_1(x_3, x_2);
x_5 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_instTop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_MonoOver_instTop___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_instInhabited___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
lean_inc(x_2);
x_4 = lean_apply_1(x_3, x_2);
x_5 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_instInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_MonoOver_instInhabited___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_leTop___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 2);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_leTop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_MonoOver_leTop___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_leTop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_MonoOver_leTop(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_mapTop___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc(x_4);
lean_inc(x_2);
lean_inc_ref(x_1);
x_5 = lp_mathlib_CategoryTheory_MonoOver_map___redArg(x_1, x_2, x_3, x_4);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_7 = lean_ctor_get(x_5, 0);
x_8 = lean_ctor_get(x_5, 1);
lean_dec(x_8);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_dec_ref(x_1);
lean_inc(x_9);
lean_inc(x_2);
x_10 = lean_apply_1(x_9, x_2);
lean_inc(x_2);
x_11 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_10);
x_12 = lean_apply_1(x_7, x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_4);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc(x_9);
x_16 = lean_apply_1(x_9, x_13);
x_17 = lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(x_16);
x_18 = lean_apply_1(x_9, x_15);
x_19 = lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(x_18);
lean_ctor_set(x_5, 1, x_19);
lean_ctor_set(x_5, 0, x_17);
return x_5;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_20 = lean_ctor_get(x_5, 0);
lean_inc(x_20);
lean_dec(x_5);
x_21 = lean_ctor_get(x_1, 1);
lean_inc(x_21);
lean_dec_ref(x_1);
lean_inc(x_21);
lean_inc(x_2);
x_22 = lean_apply_1(x_21, x_2);
lean_inc(x_2);
x_23 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_22);
x_24 = lean_apply_1(x_20, x_23);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_4);
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
lean_inc(x_21);
x_28 = lean_apply_1(x_21, x_25);
x_29 = lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(x_28);
x_30 = lean_apply_1(x_21, x_27);
x_31 = lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(x_30);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_29);
lean_ctor_set(x_32, 1, x_31);
return x_32;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MonoOver_mapTop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_MonoOver_mapTop___redArg(x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_orderTop___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
lean_inc(x_2);
x_4 = lean_apply_1(x_3, x_2);
x_5 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_orderTop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Subobject_orderTop___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_instInhabited___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
lean_inc(x_2);
x_4 = lean_apply_1(x_3, x_2);
x_5 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_instInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Subobject_instInhabited___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Functor_Currying(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Subobject_FactorThru(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Subobject_WellPowered(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Subobject_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Functor_Currying(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Subobject_FactorThru(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Subobject_WellPowered(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
