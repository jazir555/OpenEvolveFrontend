// Lean compiler output
// Module: Mathlib.Algebra.GCDMonoid.PUnit
// Imports: public import Init public import Mathlib.Algebra.GCDMonoid.Basic public import Mathlib.Algebra.Ring.PUnit
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
static lean_object* lp_mathlib_PUnit_normalizedGCDMonoid___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_normalizedGCDMonoid;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_normalizedGCDMonoid___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_normalizedGCDMonoid___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_normalizedGCDMonoid___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
static lean_object* _init_lp_mathlib_PUnit_normalizedGCDMonoid___lam__1___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_normalizedGCDMonoid___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_PUnit_normalizedGCDMonoid___lam__1___closed__0;
return x_2;
}
}
static lean_object* _init_lp_mathlib_PUnit_normalizedGCDMonoid() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PUnit_normalizedGCDMonoid___lam__0), 2, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_PUnit_normalizedGCDMonoid___lam__1), 1, 0);
lean_inc_ref(x_1);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_PUnit(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_PUnit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PUnit_normalizedGCDMonoid___lam__1___closed__0 = _init_lp_mathlib_PUnit_normalizedGCDMonoid___lam__1___closed__0();
lean_mark_persistent(lp_mathlib_PUnit_normalizedGCDMonoid___lam__1___closed__0);
lp_mathlib_PUnit_normalizedGCDMonoid = _init_lp_mathlib_PUnit_normalizedGCDMonoid();
lean_mark_persistent(lp_mathlib_PUnit_normalizedGCDMonoid);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
