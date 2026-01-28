// Lean compiler output
// Module: Mathlib.Algebra.HierarchyDesign
// Imports: public import Init public import Mathlib.Init public import Mathlib.Tactic.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_lower_x20instance_x20priority;
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_the_x20algebraic_x20hierarchy;
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_reducible_x20non_x2dinstances;
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_instance_x20argument_x20order;
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_implicit_x20instance_x20arguments;
static lean_object* _init_lp_mathlib_LibraryNote_the_x20algebraic_x20hierarchy() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_reducible_x20non_x2dinstances() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_implicit_x20instance_x20arguments() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_lower_x20instance_x20priority() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_instance_x20argument_x20order() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_HierarchyDesign(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_the_x20algebraic_x20hierarchy = _init_lp_mathlib_LibraryNote_the_x20algebraic_x20hierarchy();
lean_mark_persistent(lp_mathlib_LibraryNote_the_x20algebraic_x20hierarchy);
lp_mathlib_LibraryNote_reducible_x20non_x2dinstances = _init_lp_mathlib_LibraryNote_reducible_x20non_x2dinstances();
lean_mark_persistent(lp_mathlib_LibraryNote_reducible_x20non_x2dinstances);
lp_mathlib_LibraryNote_implicit_x20instance_x20arguments = _init_lp_mathlib_LibraryNote_implicit_x20instance_x20arguments();
lean_mark_persistent(lp_mathlib_LibraryNote_implicit_x20instance_x20arguments);
lp_mathlib_LibraryNote_lower_x20instance_x20priority = _init_lp_mathlib_LibraryNote_lower_x20instance_x20priority();
lean_mark_persistent(lp_mathlib_LibraryNote_lower_x20instance_x20priority);
lp_mathlib_LibraryNote_instance_x20argument_x20order = _init_lp_mathlib_LibraryNote_instance_x20argument_x20order();
lean_mark_persistent(lp_mathlib_LibraryNote_instance_x20argument_x20order);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
