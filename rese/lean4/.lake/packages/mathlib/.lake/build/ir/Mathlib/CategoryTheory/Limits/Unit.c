// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Unit
// Imports: public import Init public import Mathlib.CategoryTheory.PUnit public import Mathlib.CategoryTheory.Limits.HasLimits
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCone(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCocone___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCone___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCoconeIsColimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_const___redArg(lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCoconeIsColimit(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_punitExt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCocone___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_discreteCategory(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCoconeIsColimit___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCocone(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_eqToHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCone___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_discreteCategory(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__0;
x_2 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCone___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_2 = lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__1;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_3, x_4);
x_6 = lp_mathlib_CategoryTheory_Functor_punitExt___redArg(x_5, x_1);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
lean_dec(x_9);
lean_ctor_set(x_6, 1, x_8);
lean_ctor_set(x_6, 0, x_4);
return x_6;
}
else
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_6, 0);
lean_inc(x_10);
lean_dec(x_6);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCone(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Limits_punitCone___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Limits_punitCone(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCocone___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_2 = lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__1;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_3, x_4);
x_6 = lp_mathlib_CategoryTheory_Functor_punitExt___redArg(x_1, x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
lean_dec(x_9);
lean_ctor_set(x_6, 1, x_8);
lean_ctor_set(x_6, 0, x_4);
return x_6;
}
else
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_6, 0);
lean_inc(x_10);
lean_dec(x_6);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCocone(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Limits_punitCocone___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCocone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Limits_punitCocone(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lp_mathlib_CategoryTheory_eqToHom___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___redArg___lam__0(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitConeIsLimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Limits_punitConeIsLimit(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCoconeIsColimit___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_CategoryTheory_eqToHom___redArg(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCoconeIsColimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__0;
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_punitCoconeIsColimit___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_punitCoconeIsColimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Limits_punitCoconeIsColimit(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_PUnit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_HasLimits(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Unit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_HasLimits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__0 = _init_lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__0);
lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__1 = _init_lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__1();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_punitCone___redArg___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
