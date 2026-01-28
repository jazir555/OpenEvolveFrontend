// Lean compiler output
// Module: Mathlib.Analysis.Calculus.ContDiff.FaaDiBruno
// Imports: public import Init public import Mathlib.Data.Finite.Card public import Mathlib.Analysis.Analytic.Within public import Mathlib.Analysis.Calculus.FDeriv.Analytic public import Mathlib.Analysis.Calculus.ContDiff.FTaylorSeries
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
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_cases___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition_decEq___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_applyOrderedFinpartition___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_instUniqueZero;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__Fin_succ_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma___redArg(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition_decEq(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__OrderedFinpartition_extend_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_applyOrderedFinpartition___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__Fin_succ_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__OrderedFinpartition_extend_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__Fin_succ_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_applyOrderedFinpartition(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic___lam__0(lean_object*);
static lean_object* lp_mathlib_OrderedFinpartition_instUniqueOne___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extend___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extend(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_instUniqueOne;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__OrderedFinpartition_extend_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic___lam__0___boxed(lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_succ___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__1(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___boxed(lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_OrderedFinpartition_instUniqueZero___closed__0;
uint8_t lp_mathlib_Fintype_decidablePiFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_nat_dec_eq(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_6 = lean_apply_1(x_1, x_3);
x_7 = l_List_finRange(x_6);
x_8 = lp_mathlib_Fintype_decidablePiFintype___redArg(x_2, x_7, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__2(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_8);
lean_dec_ref(x_2);
x_9 = lean_nat_dec_eq(x_3, x_6);
lean_dec(x_6);
if (x_9 == 0)
{
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__0___boxed), 3, 0);
x_11 = l_List_finRange(x_3);
lean_inc_ref(x_4);
lean_inc(x_11);
lean_inc_ref(x_10);
x_12 = lp_mathlib_Fintype_decidablePiFintype___redArg(x_10, x_11, x_4, x_7);
if (x_12 == 0)
{
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_12;
}
else
{
lean_object* x_13; uint8_t x_14; 
x_13 = lean_alloc_closure((void*)(lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___lam__2___boxed), 5, 2);
lean_closure_set(x_13, 0, x_4);
lean_closure_set(x_13, 1, x_10);
x_14 = lp_mathlib_Fintype_decidablePiFintype___redArg(x_13, x_11, x_5, x_8);
return x_14;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition_decEq(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition_decEq___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_instDecidableEqOrderedFinpartition_decEq(x_1, x_2, x_3);
lean_dec(x_1);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqOrderedFinpartition___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lp_mathlib_instDecidableEqOrderedFinpartition_decEq___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_instDecidableEqOrderedFinpartition(x_1, x_2, x_3);
lean_dec(x_1);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqOrderedFinpartition___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_instDecidableEqOrderedFinpartition___redArg(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(1u);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_OrderedFinpartition_atomic___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderedFinpartition_atomic___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_atomic(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_atomic___lam__0___boxed), 1, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_atomic___lam__1___boxed), 2, 0);
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_instInhabited(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_OrderedFinpartition_atomic(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_embSigma___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_embSigma___redArg___lam__1), 2, 1);
lean_closure_set(x_6, 0, x_3);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_2);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderedFinpartition_embSigma___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_embSigma___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderedFinpartition_embSigma(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_OrderedFinpartition_instUniqueZero___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_mathlib_OrderedFinpartition_atomic(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_OrderedFinpartition_instUniqueZero() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_OrderedFinpartition_instUniqueZero___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_OrderedFinpartition_instUniqueOne___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lp_mathlib_OrderedFinpartition_atomic(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_OrderedFinpartition_instUniqueOne() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_OrderedFinpartition_instUniqueOne___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = l_Fin_cases___redArg(x_1, x_2, x_3);
x_6 = lean_apply_1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = l_Fin_cases___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = l_Fin_succ___redArg(x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_nat_mod(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderedFinpartition_extendLeft___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_OrderedFinpartition_extendLeft___lam__2(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderedFinpartition_extendLeft___lam__3(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
x_7 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendLeft___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lean_unsigned_to_nat(1u);
x_9 = lean_nat_add(x_1, x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendLeft___lam__1___boxed), 2, 1);
lean_closure_set(x_10, 0, x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendLeft___lam__2___boxed), 4, 2);
lean_closure_set(x_11, 0, x_10);
lean_closure_set(x_11, 1, x_7);
x_12 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendLeft___lam__3___boxed), 3, 2);
lean_closure_set(x_12, 0, x_8);
lean_closure_set(x_12, 1, x_5);
x_13 = lean_nat_add(x_4, x_8);
lean_dec(x_4);
lean_ctor_set(x_2, 2, x_11);
lean_ctor_set(x_2, 1, x_12);
lean_ctor_set(x_2, 0, x_13);
return x_2;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_14 = lean_ctor_get(x_2, 0);
x_15 = lean_ctor_get(x_2, 1);
x_16 = lean_ctor_get(x_2, 2);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_2);
x_17 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendLeft___lam__0), 3, 1);
lean_closure_set(x_17, 0, x_16);
x_18 = lean_unsigned_to_nat(1u);
x_19 = lean_nat_add(x_1, x_18);
x_20 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendLeft___lam__1___boxed), 2, 1);
lean_closure_set(x_20, 0, x_19);
x_21 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendLeft___lam__2___boxed), 4, 2);
lean_closure_set(x_21, 0, x_20);
lean_closure_set(x_21, 1, x_17);
x_22 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendLeft___lam__3___boxed), 3, 2);
lean_closure_set(x_22, 0, x_18);
lean_closure_set(x_22, 1, x_15);
x_23 = lean_nat_add(x_14, x_18);
lean_dec(x_14);
x_24 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_22);
lean_ctor_set(x_24, 2, x_21);
return x_24;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendLeft___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderedFinpartition_extendLeft(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__Fin_succ_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_2(x_4, x_3, lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__Fin_succ_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_2(x_2, x_1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__Fin_succ_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__Fin_succ_match__1_splitter(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lean_nat_dec_eq(x_4, x_2);
if (x_5 == 0)
{
lean_object* x_6; 
x_6 = lean_apply_1(x_1, x_4);
return x_6;
}
else
{
lean_dec(x_4);
lean_dec(x_1);
lean_inc(x_3);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = l_Fin_succ___redArg(x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; 
x_7 = lean_nat_dec_eq(x_5, x_1);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
lean_dec_ref(x_4);
x_8 = lean_apply_2(x_2, x_5, x_6);
x_9 = l_Fin_succ___redArg(x_8);
lean_dec(x_8);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_5);
lean_dec_ref(x_2);
x_10 = lean_unsigned_to_nat(0u);
x_11 = lean_nat_mod(x_10, x_3);
x_12 = l_Fin_cases___redArg(x_11, x_4, x_6);
lean_dec(x_6);
lean_dec(x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_OrderedFinpartition_extendMiddle___lam__1(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
lean_dec(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_OrderedFinpartition_extendMiddle___lam__2(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_add(x_1, x_7);
lean_inc(x_3);
lean_inc_ref(x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendMiddle___lam__0), 3, 2);
lean_closure_set(x_9, 0, x_6);
lean_closure_set(x_9, 1, x_3);
lean_inc(x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendMiddle___lam__1___boxed), 6, 4);
lean_closure_set(x_10, 0, x_3);
lean_closure_set(x_10, 1, x_6);
lean_closure_set(x_10, 2, x_8);
lean_closure_set(x_10, 3, x_9);
lean_inc_ref(x_5);
lean_inc(x_3);
x_11 = lean_apply_1(x_5, x_3);
x_12 = lean_nat_add(x_11, x_7);
lean_dec(x_11);
x_13 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendMiddle___lam__2___boxed), 4, 3);
lean_closure_set(x_13, 0, x_5);
lean_closure_set(x_13, 1, x_3);
lean_closure_set(x_13, 2, x_12);
lean_ctor_set(x_2, 2, x_10);
lean_ctor_set(x_2, 1, x_13);
return x_2;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_14 = lean_ctor_get(x_2, 0);
x_15 = lean_ctor_get(x_2, 1);
x_16 = lean_ctor_get(x_2, 2);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_2);
x_17 = lean_unsigned_to_nat(1u);
x_18 = lean_nat_add(x_1, x_17);
lean_inc(x_3);
lean_inc_ref(x_16);
x_19 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendMiddle___lam__0), 3, 2);
lean_closure_set(x_19, 0, x_16);
lean_closure_set(x_19, 1, x_3);
lean_inc(x_3);
x_20 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendMiddle___lam__1___boxed), 6, 4);
lean_closure_set(x_20, 0, x_3);
lean_closure_set(x_20, 1, x_16);
lean_closure_set(x_20, 2, x_18);
lean_closure_set(x_20, 3, x_19);
lean_inc_ref(x_15);
lean_inc(x_3);
x_21 = lean_apply_1(x_15, x_3);
x_22 = lean_nat_add(x_21, x_17);
lean_dec(x_21);
x_23 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_extendMiddle___lam__2___boxed), 4, 3);
lean_closure_set(x_23, 0, x_15);
lean_closure_set(x_23, 1, x_3);
lean_closure_set(x_23, 2, x_22);
x_24 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_24, 0, x_14);
lean_ctor_set(x_24, 1, x_23);
lean_ctor_set(x_24, 2, x_20);
return x_24;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Function_update___at___00OrderedFinpartition_extendMiddle_spec__0___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extendMiddle___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderedFinpartition_extendMiddle(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extend(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderedFinpartition_extendLeft(x_1, x_2);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lp_mathlib_OrderedFinpartition_extendMiddle(x_1, x_2, x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_extend___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderedFinpartition_extend(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = l_Fin_succ___redArg(x_2);
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = l_Fin_succ___redArg(x_3);
x_6 = lean_apply_2(x_1, x_5, x_4);
x_7 = lean_nat_sub(x_6, x_2);
lean_dec(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__1(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__1___boxed), 4, 2);
lean_closure_set(x_8, 0, x_5);
lean_closure_set(x_8, 1, x_7);
x_9 = lean_nat_sub(x_3, x_7);
lean_dec(x_3);
lean_ctor_set(x_1, 2, x_8);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_9);
return x_1;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_10 = lean_ctor_get(x_1, 0);
x_11 = lean_ctor_get(x_1, 1);
x_12 = lean_ctor_get(x_1, 2);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_13, 0, x_11);
x_14 = lean_unsigned_to_nat(1u);
x_15 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_eraseLeft___redArg___lam__1___boxed), 4, 2);
lean_closure_set(x_15, 0, x_12);
lean_closure_set(x_15, 1, x_14);
x_16 = lean_nat_sub(x_10, x_14);
lean_dec(x_10);
x_17 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_13);
lean_ctor_set(x_17, 2, x_15);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderedFinpartition_eraseLeft___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_eraseLeft___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderedFinpartition_eraseLeft(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__OrderedFinpartition_extend_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_3);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_2);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_1(x_3, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__OrderedFinpartition_extend_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__OrderedFinpartition_extend_match__1_splitter___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__OrderedFinpartition_extend_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno_0__OrderedFinpartition_extend_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_applyOrderedFinpartition___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
lean_inc(x_4);
x_6 = lean_apply_1(x_5, x_4);
x_7 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_3);
lean_closure_set(x_7, 4, x_6);
x_8 = lean_apply_2(x_2, x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_applyOrderedFinpartition(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_OrderedFinpartition_applyOrderedFinpartition___redArg(x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_applyOrderedFinpartition___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_OrderedFinpartition_applyOrderedFinpartition(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_applyOrderedFinpartition___boxed), 13, 12);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_1);
lean_closure_set(x_11, 2, lean_box(0));
lean_closure_set(x_11, 3, x_2);
lean_closure_set(x_11, 4, x_3);
lean_closure_set(x_11, 5, lean_box(0));
lean_closure_set(x_11, 6, x_4);
lean_closure_set(x_11, 7, x_5);
lean_closure_set(x_11, 8, x_6);
lean_closure_set(x_11, 9, x_7);
lean_closure_set(x_11, 10, x_8);
lean_closure_set(x_11, 11, x_10);
x_12 = lean_apply_1(x_9, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___redArg___lam__0), 10, 9);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_2);
lean_closure_set(x_10, 2, x_3);
lean_closure_set(x_10, 3, x_4);
lean_closure_set(x_10, 4, x_5);
lean_closure_set(x_10, 5, x_6);
lean_closure_set(x_10, 6, x_7);
lean_closure_set(x_10, 7, x_9);
lean_closure_set(x_10, 8, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___redArg(x_2, x_4, x_5, x_7, x_8, x_12, x_13, x_14, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_11);
lean_dec_ref(x_10);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_9, x_8, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097___redArg___lam__0), 10, 7);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_3);
lean_closure_set(x_8, 3, x_4);
lean_closure_set(x_8, 4, x_5);
lean_closure_set(x_8, 5, x_6);
lean_closure_set(x_8, 6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097___redArg(x_2, x_4, x_5, x_7, x_8, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition_u2097(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_11);
lean_dec_ref(x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_1, x_3);
x_6 = lean_apply_2(x_2, x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_9, 0);
x_11 = lean_ctor_get(x_9, 1);
lean_inc_ref(x_11);
x_12 = lean_alloc_closure((void*)(lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition___redArg___lam__0), 4, 2);
lean_closure_set(x_12, 0, x_11);
lean_closure_set(x_12, 1, x_8);
lean_inc(x_10);
x_13 = lean_apply_1(x_7, x_10);
x_14 = lean_alloc_closure((void*)(lp_mathlib_OrderedFinpartition_compAlongOrderedFinpartition___redArg___lam__0), 10, 9);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_3);
lean_closure_set(x_14, 3, x_4);
lean_closure_set(x_14, 4, x_5);
lean_closure_set(x_14, 5, x_6);
lean_closure_set(x_14, 6, x_9);
lean_closure_set(x_14, 7, x_12);
lean_closure_set(x_14, 8, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition___redArg(x_2, x_4, x_5, x_7, x_8, x_12, x_13, x_14, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_FormalMultilinearSeries_compAlongOrderedFinpartition(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_11);
lean_dec_ref(x_10);
return x_16;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finite_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Analytic_Within(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Calculus_FDeriv_Analytic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Calculus_ContDiff_FTaylorSeries(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Calculus_ContDiff_FaaDiBruno(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finite_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Analytic_Within(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Calculus_FDeriv_Analytic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Calculus_ContDiff_FTaylorSeries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_OrderedFinpartition_instUniqueZero___closed__0 = _init_lp_mathlib_OrderedFinpartition_instUniqueZero___closed__0();
lean_mark_persistent(lp_mathlib_OrderedFinpartition_instUniqueZero___closed__0);
lp_mathlib_OrderedFinpartition_instUniqueZero = _init_lp_mathlib_OrderedFinpartition_instUniqueZero();
lean_mark_persistent(lp_mathlib_OrderedFinpartition_instUniqueZero);
lp_mathlib_OrderedFinpartition_instUniqueOne___closed__0 = _init_lp_mathlib_OrderedFinpartition_instUniqueOne___closed__0();
lean_mark_persistent(lp_mathlib_OrderedFinpartition_instUniqueOne___closed__0);
lp_mathlib_OrderedFinpartition_instUniqueOne = _init_lp_mathlib_OrderedFinpartition_instUniqueOne();
lean_mark_persistent(lp_mathlib_OrderedFinpartition_instUniqueOne);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
