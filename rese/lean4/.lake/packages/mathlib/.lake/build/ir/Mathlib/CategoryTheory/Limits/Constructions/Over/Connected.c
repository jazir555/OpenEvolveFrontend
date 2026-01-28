// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Constructions.Over.Connected
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Creates public import Mathlib.CategoryTheory.Comma.Over.Basic public import Mathlib.CategoryTheory.IsConnected public import Mathlib.CategoryTheory.Filtered.Final
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Over_forget___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePostIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_const___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_proj___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cone_category___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_functoriality___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Over_post___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePostIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCone___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
x_5 = lean_ctor_get(x_4, 2);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow___redArg(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_natTransInCostructuredArrow(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lp_mathlib_CategoryTheory_CostructuredArrow_proj___redArg(x_1, x_2, x_3, x_4);
x_9 = lp_mathlib_CategoryTheory_Functor_mapCone___redArg(x_8, x_5, x_7);
x_10 = lean_apply_1(x_6, x_9);
x_11 = lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___redArg___lam__0___boxed), 7, 6);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_3);
lean_closure_set(x_7, 3, x_4);
lean_closure_set(x_7, 4, x_5);
lean_closure_set(x_7, 5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___redArg(x_4, x_6, x_7, x_9, x_10, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_CostructuredArrow_CreatesConnected_isLimitRaiseCone(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_2);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__0(x_1, x_2, x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lean_apply_1(x_1, x_3);
x_5 = lp_mathlib_CategoryTheory_CostructuredArrow_homMk___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 1);
x_9 = lean_apply_1(x_5, x_7);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
lean_inc(x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__1), 2, 1);
lean_closure_set(x_11, 0, x_8);
lean_inc(x_2);
x_12 = lean_apply_1(x_10, x_2);
x_13 = lean_apply_1(x_8, x_2);
x_14 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_12, x_13);
lean_ctor_set(x_3, 1, x_11);
lean_ctor_set(x_3, 0, x_14);
return x_3;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_15 = lean_ctor_get(x_3, 0);
x_16 = lean_ctor_get(x_3, 1);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_3);
x_17 = lean_apply_1(x_5, x_15);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
lean_inc(x_16);
x_19 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__1), 2, 1);
lean_closure_set(x_19, 0, x_16);
lean_inc(x_2);
x_20 = lean_apply_1(x_18, x_2);
x_21 = lean_apply_1(x_16, x_2);
x_22 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_20, x_21);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_19);
return x_23;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__0___boxed), 3, 0);
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Over_conePost___redArg___lam__2), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Over_conePost___redArg(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePost___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Over_conePost(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePostIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_inc(x_4);
x_6 = lean_apply_1(x_5, x_4);
lean_inc(x_4);
x_7 = lp_mathlib_CategoryTheory_Over_post___redArg(x_1, x_4, x_3);
lean_inc_ref(x_2);
x_8 = lp_mathlib_CategoryTheory_Over_forget___redArg(x_2, x_6);
lean_inc_ref(x_2);
x_9 = lp_mathlib_CategoryTheory_Limits_Cone_category___redArg(x_2);
x_10 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_9);
x_11 = lp_mathlib_CategoryTheory_Over_conePost___redArg(x_2, x_4);
x_12 = lp_mathlib_CategoryTheory_Limits_Cones_functoriality___redArg(x_7, x_8);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_11, x_12);
x_14 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_10, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Over_conePostIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Over_conePostIso___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Creates(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Comma_Over_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_IsConnected(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Filtered_Final(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_Over_Connected(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Creates(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Comma_Over_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_IsConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Filtered_Final(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
