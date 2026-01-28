// Lean compiler output
// Module: Mathlib.Order.RelClasses
// Imports: public import Init public import Mathlib.Logic.IsEmpty public import Mathlib.Order.Basic public import Mathlib.Tactic.MkIffOfInductiveProp
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
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfSTO___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsWellFounded_fix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg(lean_object*);
static lean_object* lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__0;
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfSTO___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
static lean_object* lp_mathlib_partialOrderOfSO___closed__0;
lean_object* lp_batteries_Lean_MVarId_assignIfDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_fix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_toWellFoundedRelation(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_partialOrderOfSO(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_GCongr_exactSubsetOfSSubset;
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_toWellFoundedRelation(lean_object*, lean_object*, lean_object*);
lean_object* l_WellFounded_fixC___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_fix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_fix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsWellOrder_toHasWellFounded(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsWellFounded_fix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__1(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__1;
lean_object* l_Lean_Name_mkStr1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_fix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsWellFounded_toWellFoundedRelation(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__2(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_partialOrderOfSO___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_partialOrderOfSO(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_partialOrderOfSO___closed__0;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_3);
lean_inc(x_2);
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
lean_dec(x_3);
return x_2;
}
else
{
lean_dec(x_2);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_3);
lean_inc(x_2);
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
lean_dec(x_2);
return x_3;
}
else
{
lean_dec(x_3);
return x_2;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfSTO___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = 1;
return x_6;
}
else
{
uint8_t x_7; 
x_7 = 0;
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_linearOrderOfSTO___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfSTO___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
lean_inc(x_3);
lean_inc(x_2);
lean_inc_ref(x_1);
x_4 = lp_mathlib_decidableLTOfDecidableLE___redArg(x_1, x_2, x_3);
if (x_4 == 0)
{
uint8_t x_5; 
x_5 = lp_mathlib_decidableEqOfDecidableLE___redArg(x_1, x_2, x_3);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = 2;
return x_6;
}
else
{
uint8_t x_7; 
x_7 = 1;
return x_7;
}
}
else
{
uint8_t x_8; 
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_8 = 0;
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_linearOrderOfSTO___redArg___lam__3(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_5, 0, x_4);
lean_inc_ref(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__1), 3, 1);
lean_closure_set(x_6, 0, x_5);
lean_inc_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__2), 3, 1);
lean_closure_set(x_7, 0, x_5);
lean_inc_ref(x_5);
x_8 = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__3___boxed), 3, 1);
lean_closure_set(x_8, 0, x_5);
x_9 = lp_mathlib_partialOrderOfSO___closed__0;
lean_inc_ref(x_5);
x_10 = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_9);
lean_closure_set(x_10, 2, x_5);
lean_inc_ref(x_5);
x_11 = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_9);
lean_closure_set(x_11, 2, x_5);
x_12 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_12, 0, x_9);
lean_ctor_set(x_12, 1, x_7);
lean_ctor_set(x_12, 2, x_6);
lean_ctor_set(x_12, 3, x_8);
lean_ctor_set(x_12, 4, x_5);
lean_ctor_set(x_12, 5, x_10);
lean_ctor_set(x_12, 6, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__1), 3, 1);
lean_closure_set(x_3, 0, x_2);
lean_inc_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__2), 3, 1);
lean_closure_set(x_4, 0, x_2);
lean_inc_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__3___boxed), 3, 1);
lean_closure_set(x_5, 0, x_2);
x_6 = lp_mathlib_partialOrderOfSO___closed__0;
lean_inc_ref(x_2);
x_7 = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_6);
lean_closure_set(x_7, 2, x_2);
lean_inc_ref(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_6);
lean_closure_set(x_8, 2, x_2);
x_9 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_4);
lean_ctor_set(x_9, 2, x_3);
lean_ctor_set(x_9, 3, x_5);
lean_ctor_set(x_9, 4, x_2);
lean_ctor_set(x_9, 5, x_7);
lean_ctor_set(x_9, 6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsWellFounded_fix(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = l_WellFounded_fixC___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsWellFounded_fix___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_WellFounded_fixC___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsWellFounded_toWellFoundedRelation(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_fix(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = l_WellFounded_fixC___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_fix___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_WellFounded_fixC___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_fix(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = l_WellFounded_fixC___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_fix___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_WellFounded_fixC___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_toWellFoundedRelation(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_toWellFoundedRelation(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsWellOrder_toHasWellFounded(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
static lean_object* _init_lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("subset_of_ssubset", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__1;
x_9 = lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__2;
x_10 = lean_array_push(x_9, x_1);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_11 = l_Lean_Meta_mkAppM(x_8, x_10, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lp_batteries_Lean_MVarId_assignIfDefEq(x_2, x_12, x_3, x_4, x_5, x_6);
return x_13;
}
else
{
uint8_t x_14; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_14 = !lean_is_exclusive(x_11);
if (x_14 == 0)
{
return x_11;
}
else
{
lean_object* x_15; lean_object* x_16; 
x_15 = lean_ctor_get(x_11, 0);
lean_inc(x_15);
lean_dec(x_11);
x_16 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_16, 0, x_15);
return x_16;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
static lean_object* _init_lp_mathlib_GCongr_exactSubsetOfSSubset() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___boxed), 7, 0);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_IsEmpty(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_IsEmpty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_partialOrderOfSO___closed__0 = _init_lp_mathlib_partialOrderOfSO___closed__0();
lean_mark_persistent(lp_mathlib_partialOrderOfSO___closed__0);
lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__0 = _init_lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__0);
lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__1 = _init_lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__1);
lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__2 = _init_lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__2();
lean_mark_persistent(lp_mathlib_GCongr_exactSubsetOfSSubset___lam__0___closed__2);
lp_mathlib_GCongr_exactSubsetOfSSubset = _init_lp_mathlib_GCongr_exactSubsetOfSSubset();
lean_mark_persistent(lp_mathlib_GCongr_exactSubsetOfSSubset);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
