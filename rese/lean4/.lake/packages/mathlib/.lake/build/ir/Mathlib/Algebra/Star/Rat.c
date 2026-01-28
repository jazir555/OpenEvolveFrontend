// Lean compiler output
// Module: Mathlib.Algebra.Star.Rat
// Imports: public import Init public import Mathlib.Algebra.Field.Opposite public import Mathlib.Algebra.Star.Basic public import Mathlib.Data.NNRat.Defs public import Mathlib.Data.Rat.Cast.Defs
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
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instStarRing;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instStarRing;
static lean_object* lp_mathlib_Rat_instStarRing___closed__0;
static lean_object* _init_lp_mathlib_Rat_instStarRing___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_instStarRing() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_instStarRing___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_NNRat_instStarRing() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_instStarRing___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Star_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_NNRat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Star_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Star_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_NNRat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_instStarRing___closed__0 = _init_lp_mathlib_Rat_instStarRing___closed__0();
lean_mark_persistent(lp_mathlib_Rat_instStarRing___closed__0);
lp_mathlib_Rat_instStarRing = _init_lp_mathlib_Rat_instStarRing();
lean_mark_persistent(lp_mathlib_Rat_instStarRing);
lp_mathlib_NNRat_instStarRing = _init_lp_mathlib_NNRat_instStarRing();
lean_mark_persistent(lp_mathlib_NNRat_instStarRing);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
