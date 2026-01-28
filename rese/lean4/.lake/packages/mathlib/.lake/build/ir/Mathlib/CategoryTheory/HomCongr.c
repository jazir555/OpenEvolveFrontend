// Lean compiler output
// Module: Mathlib.CategoryTheory.HomCongr
// Imports: public import Init public import Mathlib.CategoryTheory.Iso
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
lean_object* lp_mathlib_CategoryTheory_Iso_trans___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_homCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongrLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_homCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongrRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_homCongr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_homCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_homCongr___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_dec_ref(x_1);
x_10 = lean_ctor_get(x_2, 0);
lean_inc(x_10);
lean_dec_ref(x_2);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_dec_ref(x_3);
lean_inc(x_9);
lean_inc(x_6);
lean_inc(x_4);
x_12 = lean_apply_5(x_9, x_4, x_5, x_6, x_8, x_11);
x_13 = lean_apply_5(x_9, x_7, x_4, x_6, x_10, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_homCongr___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_dec_ref(x_1);
x_10 = lean_ctor_get(x_2, 1);
lean_inc(x_10);
lean_dec_ref(x_2);
x_11 = lean_ctor_get(x_3, 0);
lean_inc(x_11);
lean_dec_ref(x_3);
lean_inc(x_9);
lean_inc(x_6);
lean_inc(x_4);
x_12 = lean_apply_5(x_9, x_4, x_5, x_6, x_8, x_11);
x_13 = lean_apply_5(x_9, x_7, x_4, x_6, x_10, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_homCongr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc(x_2);
lean_inc(x_3);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Iso_homCongr___redArg___lam__0), 8, 7);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_6);
lean_closure_set(x_8, 2, x_7);
lean_closure_set(x_8, 3, x_4);
lean_closure_set(x_8, 4, x_5);
lean_closure_set(x_8, 5, x_3);
lean_closure_set(x_8, 6, x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Iso_homCongr___redArg___lam__1), 8, 7);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_6);
lean_closure_set(x_9, 2, x_7);
lean_closure_set(x_9, 3, x_2);
lean_closure_set(x_9, 4, x_3);
lean_closure_set(x_9, 5, x_5);
lean_closure_set(x_9, 6, x_4);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_homCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Iso_homCongr___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongr___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_1);
lean_inc(x_5);
lean_inc(x_3);
lean_inc_ref(x_2);
x_10 = lp_mathlib_CategoryTheory_Iso_trans___redArg(x_2, x_3, x_4, x_5, x_8, x_6);
x_11 = lp_mathlib_CategoryTheory_Iso_trans___redArg(x_2, x_7, x_3, x_5, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongr___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_1);
lean_inc(x_5);
lean_inc(x_3);
lean_inc_ref(x_2);
x_10 = lp_mathlib_CategoryTheory_Iso_trans___redArg(x_2, x_3, x_4, x_5, x_8, x_9);
x_11 = lp_mathlib_CategoryTheory_Iso_trans___redArg(x_2, x_6, x_3, x_5, x_7, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc(x_4);
lean_inc_ref(x_7);
lean_inc(x_5);
lean_inc(x_3);
lean_inc(x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Iso_isoCongr___redArg___lam__0), 8, 7);
lean_closure_set(x_8, 0, x_6);
lean_closure_set(x_8, 1, x_1);
lean_closure_set(x_8, 2, x_2);
lean_closure_set(x_8, 3, x_3);
lean_closure_set(x_8, 4, x_5);
lean_closure_set(x_8, 5, x_7);
lean_closure_set(x_8, 6, x_4);
x_9 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Iso_isoCongr___redArg___lam__1), 8, 7);
lean_closure_set(x_9, 0, x_7);
lean_closure_set(x_9, 1, x_1);
lean_closure_set(x_9, 2, x_4);
lean_closure_set(x_9, 3, x_5);
lean_closure_set(x_9, 4, x_3);
lean_closure_set(x_9, 5, x_2);
lean_closure_set(x_9, 6, x_6);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Iso_isoCongr___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongrLeft___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
lean_inc_ref(x_1);
x_6 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_1, x_4);
lean_inc(x_4);
x_7 = lp_mathlib_CategoryTheory_Iso_isoCongr___redArg(x_1, x_2, x_4, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongrLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Iso_isoCongrLeft___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongrRight___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
lean_inc(x_2);
lean_inc_ref(x_1);
x_6 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_1, x_2);
lean_inc(x_2);
x_7 = lp_mathlib_CategoryTheory_Iso_isoCongr___redArg(x_1, x_2, x_3, x_2, x_4, x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_isoCongrRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Iso_isoCongrRight___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Iso(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_HomCongr(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Iso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
