// Lean compiler output
// Module: Mathlib.Data.Finset.Density
// Imports: public import Init public import Mathlib.Algebra.Order.Field.Rat public import Mathlib.Data.Fintype.Card public import Mathlib.Data.NNRat.Order public import Mathlib.Data.Rat.Cast.CharZero public import Mathlib.Tactic.Positivity.Basic
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
lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
extern lean_object* lp_mathlib_NNRat_instSemifield;
LEAN_EXPORT lean_object* lp_mathlib_Finset_dens___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Finset_dens___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_dens(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_dens___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_dens___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* l_Rat_div(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Rat_instNNRatCast___lam__0(lean_object*);
static lean_object* _init_lp_mathlib_Finset_dens___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_NNRat_instSemifield;
x_2 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_dens___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_3 = lp_mathlib_Finset_dens___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
x_5 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_4);
lean_dec_ref(x_4);
x_6 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = l_List_lengthTR___redArg(x_2);
lean_inc(x_7);
x_9 = lean_apply_1(x_7, x_8);
x_10 = l_List_lengthTR___redArg(x_1);
x_11 = lean_apply_1(x_7, x_10);
x_12 = lp_mathlib_Rat_instNNRatCast___lam__0(x_9);
lean_dec(x_9);
x_13 = lp_mathlib_Rat_instNNRatCast___lam__0(x_11);
lean_dec(x_11);
x_14 = l_Rat_div(x_12, x_13);
lean_dec_ref(x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_dens(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_dens___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_dens___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_dens(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_dens___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_dens___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Field_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_NNRat_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Positivity_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Density(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Field_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_NNRat_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Positivity_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Finset_dens___redArg___closed__0 = _init_lp_mathlib_Finset_dens___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Finset_dens___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
