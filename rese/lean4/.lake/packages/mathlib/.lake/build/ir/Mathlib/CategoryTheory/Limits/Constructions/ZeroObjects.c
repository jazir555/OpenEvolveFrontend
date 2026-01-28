// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Constructions.ZeroObjects
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Shapes.ZeroMorphisms public import Mathlib.CategoryTheory.Limits.Constructions.BinaryProducts
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroRightIsColimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroRightIsLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroLeftIsColimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroLeftIsColimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroLeftIsColimit___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroLeftIsLimit___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroLeftIsLimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroRightIsColimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroRightIsLimit___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroRightIsColimit___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroLeftIsLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroRightIsLimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroLeftIsLimit___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = 1;
x_4 = lean_box(x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroLeftIsLimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_binaryFanZeroLeftIsLimit___lam__0), 1, 0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroLeftIsLimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Limits_binaryFanZeroLeftIsLimit(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroRightIsLimit___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = 0;
x_4 = lean_box(x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroRightIsLimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_binaryFanZeroRightIsLimit___lam__0), 1, 0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryFanZeroRightIsLimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Limits_binaryFanZeroRightIsLimit(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroLeftIsColimit___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = 1;
x_4 = lean_box(x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroLeftIsColimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_binaryCofanZeroLeftIsColimit___lam__0), 1, 0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroLeftIsColimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Limits_binaryCofanZeroLeftIsColimit(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroRightIsColimit___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = 0;
x_4 = lean_box(x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroRightIsColimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_binaryCofanZeroRightIsColimit___lam__0), 1, 0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_binaryCofanZeroRightIsColimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Limits_binaryCofanZeroRightIsColimit(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_ZeroMorphisms(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_BinaryProducts(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_ZeroObjects(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_ZeroMorphisms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_BinaryProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
