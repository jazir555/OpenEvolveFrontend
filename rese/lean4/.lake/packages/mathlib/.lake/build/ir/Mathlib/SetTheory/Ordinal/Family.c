// Lean compiler output
// Module: Mathlib.SetTheory.Ordinal.Family
// Imports: public import Init public import Mathlib.SetTheory.Ordinal.Arithmetic
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
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Ordinal_typein(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Ordinal_typein(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lean_apply_2(x_1, x_5, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Ordinal_familyOfBFamily_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Ordinal_familyOfBFamily(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Arithmetic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Family(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Ordinal_Arithmetic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0 = _init_lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
