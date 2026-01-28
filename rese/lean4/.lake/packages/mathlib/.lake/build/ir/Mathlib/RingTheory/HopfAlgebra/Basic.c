// Lean compiler output
// Module: Mathlib.RingTheory.HopfAlgebra.Basic
// Imports: public import Init public import Mathlib.RingTheory.Bialgebra.Basic
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
static lean_object* lp_mathlib_CommSemiring_toHopfAlgebra___redArg___closed__0;
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_CommSemiring_toBialgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toHopfAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toHopfAlgebra(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_CommSemiring_toHopfAlgebra___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toHopfAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_CommSemiring_toBialgebra___redArg(x_1);
x_3 = lp_mathlib_CommSemiring_toHopfAlgebra___redArg___closed__0;
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toHopfAlgebra(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommSemiring_toHopfAlgebra___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Bialgebra_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_HopfAlgebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Bialgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CommSemiring_toHopfAlgebra___redArg___closed__0 = _init_lp_mathlib_CommSemiring_toHopfAlgebra___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CommSemiring_toHopfAlgebra___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
