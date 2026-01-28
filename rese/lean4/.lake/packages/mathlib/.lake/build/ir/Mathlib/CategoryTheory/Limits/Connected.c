// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Connected
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Shapes.BinaryProducts public import Mathlib.CategoryTheory.Limits.Shapes.Equalizers public import Mathlib.CategoryTheory.Limits.Shapes.WidePullbacks public import Mathlib.CategoryTheory.IsConnected public import Mathlib.CategoryTheory.Limits.Preserves.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCocone___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_const___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCone___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_NatTrans_id___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCocone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_CategoryTheory_parallelPairInhabited;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCone___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_dec(x_6);
lean_inc(x_2);
x_7 = lean_apply_1(x_5, x_2);
x_8 = lp_mathlib_CategoryTheory_NatTrans_id___redArg(x_1, x_7);
lean_ctor_set(x_3, 1, x_8);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_3, 0);
lean_inc(x_9);
lean_dec(x_3);
lean_inc(x_2);
x_10 = lean_apply_1(x_9, x_2);
x_11 = lp_mathlib_CategoryTheory_NatTrans_id___redArg(x_1, x_10);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Limits_constCone___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Limits_constCone(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCocone___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_dec(x_6);
lean_inc(x_2);
x_7 = lean_apply_1(x_5, x_2);
x_8 = lp_mathlib_CategoryTheory_NatTrans_id___redArg(x_1, x_7);
lean_ctor_set(x_3, 1, x_8);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_3, 0);
lean_inc(x_9);
lean_dec(x_3);
lean_inc(x_2);
x_10 = lean_apply_1(x_9, x_2);
x_11 = lp_mathlib_CategoryTheory_NatTrans_id___redArg(x_1, x_10);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Limits_constCocone___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_constCocone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Limits_constCocone(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
static uint8_t _init_lp_mathlib_CategoryTheory_parallelPairInhabited() {
_start:
{
uint8_t x_1; 
x_1 = 1;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryProducts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Equalizers(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_WidePullbacks(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_IsConnected(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Connected(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Equalizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_WidePullbacks(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_IsConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_parallelPairInhabited = _init_lp_mathlib_CategoryTheory_parallelPairInhabited();
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
