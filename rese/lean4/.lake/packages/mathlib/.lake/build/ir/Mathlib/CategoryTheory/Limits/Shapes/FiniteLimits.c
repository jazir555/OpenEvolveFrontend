// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Shapes.FiniteLimits
// Imports: public import Init public import Mathlib.CategoryTheory.FinCategory.AsType public import Mathlib.CategoryTheory.Limits.Shapes.BinaryProducts public import Mathlib.CategoryTheory.Limits.Shapes.Equalizers public import Mathlib.CategoryTheory.Limits.Shapes.WidePullbacks public import Mathlib.CategoryTheory.Limits.Shapes.Pullback.HasPullback public import Mathlib.Data.Fintype.Option
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_instCategoryULiftHomULiftOfSmallCategory(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_instDecidableEqWalkingPair___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_WidePullbackShape_fintypeObj___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_uliftCategory___redArg(lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__0;
static lean_object* lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_instCategoryULiftHomULiftOfSmallCategory___redArg(lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair;
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__2;
lean_object* lp_mathlib_Finset_insertNone___lam__0(lean_object*);
lean_object* lp_mathlib_CategoryTheory_ULiftHom_category___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_instDecidableEqWalkingParallelPair___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_WidePushoutShape_fintypeObj___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_ndinsert___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_WidePullbackShape_fintypeObj(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_WidePushoutShape_fintypeObj(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_instCategoryULiftHomULiftOfSmallCategory___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_CategoryTheory_uliftCategory___redArg(x_1);
x_3 = lp_mathlib_CategoryTheory_ULiftHom_category___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_instCategoryULiftHomULiftOfSmallCategory(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Limits_instCategoryULiftHomULiftOfSmallCategory___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__0() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_box(0);
x_2 = 1;
x_3 = lean_box(x_2);
x_4 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__1() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__0;
x_2 = 0;
x_3 = lean_box(x_2);
x_4 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__1;
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_instDecidableEqWalkingParallelPair___boxed), 2, 0);
x_3 = lp_mathlib_List_dedup___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__2;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_WidePullbackShape_fintypeObj(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_insertNone___lam__0(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_WidePullbackShape_fintypeObj___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Finset_insertNone___lam__0(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_WidePushoutShape_fintypeObj(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_insertNone___lam__0(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_WidePushoutShape_fintypeObj___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Finset_insertNone___lam__0(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__0() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_box(0);
x_2 = 1;
x_3 = lean_box(x_2);
x_4 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__1() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__0;
x_2 = 0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_instDecidableEqWalkingPair___boxed), 2, 0);
x_4 = lean_box(x_2);
x_5 = lp_mathlib_Multiset_ndinsert___redArg(x_3, x_4, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__1;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_FinCategory_AsType(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryProducts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Equalizers(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_WidePullbacks(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_HasPullback(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_FiniteLimits(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_FinCategory_AsType(builtin);
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
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_HasPullback(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__0 = _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__0);
lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__1 = _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__1();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__1);
lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__2 = _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__2();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair___closed__2);
lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair = _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_fintypeWalkingParallelPair);
lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__0 = _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__0);
lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__1 = _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__1();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair___closed__1);
lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair = _init_lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_fintypeWalkingPair);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
