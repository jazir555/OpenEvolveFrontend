// Lean compiler output
// Module: Mathlib.Algebra.PEmptyInstances
// Imports: public import Init public import Mathlib.Algebra.Group.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_AddSemigroupPEmpty;
LEAN_EXPORT lean_object* lp_mathlib_SemigroupPEmpty;
LEAN_EXPORT uint8_t lp_mathlib_SemigroupPEmpty___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemigroupPEmpty___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddSemigroupPEmpty___closed__0;
LEAN_EXPORT uint8_t lp_mathlib_SemigroupPEmpty___lam__0(uint8_t x_1, uint8_t x_2, lean_object* x_3) {
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemigroupPEmpty___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; uint8_t x_5; uint8_t x_6; lean_object* x_7; 
x_4 = lean_unbox(x_1);
x_5 = lean_unbox(x_2);
x_6 = lp_mathlib_SemigroupPEmpty___lam__0(x_4, x_5, x_3);
x_7 = lean_box(x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_SemigroupPEmpty() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_SemigroupPEmpty___lam__0___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddSemigroupPEmpty___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_SemigroupPEmpty___lam__0___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddSemigroupPEmpty() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddSemigroupPEmpty___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_PEmptyInstances(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_SemigroupPEmpty = _init_lp_mathlib_SemigroupPEmpty();
lean_mark_persistent(lp_mathlib_SemigroupPEmpty);
lp_mathlib_AddSemigroupPEmpty___closed__0 = _init_lp_mathlib_AddSemigroupPEmpty___closed__0();
lean_mark_persistent(lp_mathlib_AddSemigroupPEmpty___closed__0);
lp_mathlib_AddSemigroupPEmpty = _init_lp_mathlib_AddSemigroupPEmpty();
lean_mark_persistent(lp_mathlib_AddSemigroupPEmpty);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
