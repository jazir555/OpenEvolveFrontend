// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.IsConnected
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Types.Colimits public import Mathlib.CategoryTheory.IsConnected public import Mathlib.CategoryTheory.Limits.Final public import Mathlib.CategoryTheory.HomCongr
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
static lean_object* lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_const___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone(lean_object*, lean_object*);
extern lean_object* lp_mathlib_CategoryTheory_types;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CategoryTheory_types;
x_2 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone___lam__0___boxed), 2, 0);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, lean_box(0));
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Limits_Types_pUnitCocone(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Colimits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_IsConnected(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Final(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_HomCongr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_IsConnected(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Colimits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_IsConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Final(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_HomCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor___closed__0 = _init_lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_Types_constPUnitFunctor___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
