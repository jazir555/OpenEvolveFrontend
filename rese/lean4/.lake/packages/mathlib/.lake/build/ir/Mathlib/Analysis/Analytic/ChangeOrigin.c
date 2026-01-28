// Lean compiler output
// Module: Mathlib.Analysis.Analytic.ChangeOrigin
// Imports: public import Init public import Mathlib.Analysis.Analytic.Basic
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
lean_object* l_List_lengthTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv;
lean_object* lp_mathlib_finCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv___lam__0(lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv___lam__1(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_nat_add(x_3, x_5);
lean_dec(x_5);
lean_dec(x_3);
lean_ctor_set(x_2, 0, x_6);
return x_2;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_2);
x_9 = lean_nat_add(x_3, x_7);
lean_dec(x_7);
lean_dec(x_3);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_8);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv___lam__1(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = l_List_lengthTR___redArg(x_4);
x_6 = lean_nat_sub(x_3, x_5);
x_7 = lean_nat_add(x_6, x_5);
x_8 = lp_mathlib_finCongr(x_3, x_7, lean_box(0));
lean_dec(x_7);
lean_dec(x_3);
x_9 = lp_mathlib_Equiv_toEmbedding___redArg(x_8);
x_10 = lp_mathlib_Finset_map___redArg(x_9, x_4);
lean_ctor_set(x_1, 1, x_10);
lean_ctor_set(x_1, 0, x_5);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_6);
lean_ctor_set(x_11, 1, x_1);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_12 = lean_ctor_get(x_1, 0);
x_13 = lean_ctor_get(x_1, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_1);
x_14 = l_List_lengthTR___redArg(x_13);
x_15 = lean_nat_sub(x_12, x_14);
x_16 = lean_nat_add(x_15, x_14);
x_17 = lp_mathlib_finCongr(x_12, x_16, lean_box(0));
lean_dec(x_16);
lean_dec(x_12);
x_18 = lp_mathlib_Equiv_toEmbedding___redArg(x_17);
x_19 = lp_mathlib_Finset_map___redArg(x_18, x_13);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_14);
lean_ctor_set(x_20, 1, x_19);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_15);
lean_ctor_set(x_21, 1, x_20);
return x_21;
}
}
}
static lean_object* _init_lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv___lam__0), 1, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv___lam__1), 1, 0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Analytic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Analytic_ChangeOrigin(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Analytic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv = _init_lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv();
lean_mark_persistent(lp_mathlib_FormalMultilinearSeries_changeOriginIndexEquiv);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
