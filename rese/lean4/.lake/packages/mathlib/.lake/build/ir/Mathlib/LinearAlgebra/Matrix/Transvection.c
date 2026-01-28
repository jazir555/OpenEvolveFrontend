// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.Transvection
// Imports: public import Init public import Mathlib.Data.Matrix.Basis public import Mathlib.Data.Matrix.DMatrix public import Mathlib.LinearAlgebra.Matrix.Determinant.Basic public import Mathlib.LinearAlgebra.Matrix.Reindex public import Mathlib.Tactic.Field
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
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transvection___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_toMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_reindexEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecRow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_sumInl(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transvection___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transvection(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecCol___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_diagonal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_toMatrix___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecRow___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1___closed__0;
uint8_t l_instDecidableEqSum_decEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecCol(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_inv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_sumInl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_reindexEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transvection___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecRow___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_inv___redArg(lean_object*, lean_object*);
lean_object* l_instDecidableEqFin___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_single___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* l_instDecidableEqPUnit___boxed(lean_object*, lean_object*);
lean_object* l_List_ofFn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transvection___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transvection___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Matrix_transvection___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transvection___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
lean_inc_ref(x_2);
x_8 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_2);
x_9 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_8);
lean_inc_ref(x_9);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_9);
x_11 = lean_ctor_get(x_10, 1);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_9);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_2);
x_15 = lean_ctor_get(x_14, 1);
lean_inc_ref(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_15, 2);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_transvection___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_17, 0, x_16);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_13);
lean_inc_ref(x_1);
x_18 = lp_mathlib_Matrix_diagonal___redArg(x_1, x_13, x_17, x_6, x_7);
lean_inc_ref(x_1);
x_19 = lp_mathlib_Matrix_single___redArg(x_1, x_1, x_13, x_3, x_4, x_5, x_6, x_7);
x_20 = lean_apply_2(x_11, x_18, x_19);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transvection(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_transvection___redArg(x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_toMatrix___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_3, 2);
lean_inc(x_8);
lean_dec_ref(x_3);
x_9 = lp_mathlib_Matrix_transvection___redArg(x_1, x_2, x_6, x_7, x_8, x_4, x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_toMatrix(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_TransvectionStruct_toMatrix___redArg(x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_inv___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 2);
x_5 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_6 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_5);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_7, x_4);
lean_ctor_set(x_2, 2, x_8);
return x_2;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_9 = lean_ctor_get(x_2, 0);
x_10 = lean_ctor_get(x_2, 1);
x_11 = lean_ctor_get(x_2, 2);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_2);
x_12 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_13 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_12);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_13, 1);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lean_apply_1(x_14, x_11);
x_16 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_16, 0, x_9);
lean_ctor_set(x_16, 1, x_10);
lean_ctor_set(x_16, 2, x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_inv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Matrix_TransvectionStruct_inv___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_sumInl___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_5, 0, x_3);
x_6 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_7);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_8);
x_12 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
lean_ctor_set(x_12, 2, x_9);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_sumInl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Matrix_TransvectionStruct_sumInl___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_reindexEquiv___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
lean_inc(x_6);
x_7 = lean_apply_1(x_6, x_4);
x_8 = lean_apply_1(x_6, x_5);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_7);
return x_2;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = lean_ctor_get(x_2, 0);
x_10 = lean_ctor_get(x_2, 1);
x_11 = lean_ctor_get(x_2, 2);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_2);
x_12 = lean_ctor_get(x_1, 0);
lean_inc(x_12);
lean_dec_ref(x_1);
lean_inc(x_12);
x_13 = lean_apply_1(x_12, x_9);
x_14 = lean_apply_1(x_12, x_10);
x_15 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
lean_ctor_set(x_15, 2, x_11);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_TransvectionStruct_reindexEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_TransvectionStruct_reindexEquiv___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_closure((void*)(l_instDecidableEqPUnit___boxed), 2, 0);
x_6 = l_instDecidableEqSum_decEq___redArg(x_4, x_5, x_2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_6);
x_10 = lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1___closed__0;
lean_inc(x_1);
lean_inc_ref(x_9);
x_11 = lean_apply_2(x_1, x_9, x_10);
x_12 = lean_apply_1(x_2, x_11);
x_13 = lean_apply_2(x_1, x_10, x_10);
x_14 = lean_apply_2(x_3, x_12, x_13);
x_15 = lp_mathlib_Matrix_transvection___redArg(x_4, x_5, x_9, x_10, x_14, x_7, x_8);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecCol___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
x_5 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
lean_inc_ref(x_5);
x_6 = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(x_5);
x_7 = lean_ctor_get(x_6, 2);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_5);
x_9 = lp_mathlib_Ring_toAddCommGroup___redArg(x_8);
x_10 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_9);
lean_dec_ref(x_9);
x_11 = lean_ctor_get(x_10, 1);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc(x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_12, 0, x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1), 8, 5);
lean_closure_set(x_13, 0, x_3);
lean_closure_set(x_13, 1, x_11);
lean_closure_set(x_13, 2, x_7);
lean_closure_set(x_13, 3, x_12);
lean_closure_set(x_13, 4, x_4);
x_14 = l_List_ofFn___redArg(x_2, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecCol(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Matrix_Pivot_listTransvecCol___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecRow___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1___closed__0;
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_6);
lean_inc(x_1);
lean_inc_ref(x_10);
x_11 = lean_apply_2(x_1, x_9, x_10);
x_12 = lean_apply_1(x_2, x_11);
x_13 = lean_apply_2(x_1, x_9, x_9);
x_14 = lean_apply_2(x_3, x_12, x_13);
x_15 = lp_mathlib_Matrix_transvection___redArg(x_4, x_5, x_9, x_10, x_14, x_7, x_8);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecRow___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
x_5 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
lean_inc_ref(x_5);
x_6 = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(x_5);
x_7 = lean_ctor_get(x_6, 2);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_5);
x_9 = lp_mathlib_Ring_toAddCommGroup___redArg(x_8);
x_10 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_9);
lean_dec_ref(x_9);
x_11 = lean_ctor_get(x_10, 1);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc(x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_12, 0, x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_Pivot_listTransvecRow___redArg___lam__1), 8, 5);
lean_closure_set(x_13, 0, x_3);
lean_closure_set(x_13, 1, x_11);
lean_closure_set(x_13, 2, x_7);
lean_closure_set(x_13, 3, x_12);
lean_closure_set(x_13, 4, x_4);
x_14 = l_List_ofFn___redArg(x_2, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_Pivot_listTransvecRow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Matrix_Pivot_listTransvecRow___redArg(x_2, x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Basis(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_DMatrix(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Determinant_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Reindex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Field(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Transvection(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Basis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_DMatrix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Determinant_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Reindex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1___closed__0 = _init_lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1___closed__0();
lean_mark_persistent(lp_mathlib_Matrix_Pivot_listTransvecCol___redArg___lam__1___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
