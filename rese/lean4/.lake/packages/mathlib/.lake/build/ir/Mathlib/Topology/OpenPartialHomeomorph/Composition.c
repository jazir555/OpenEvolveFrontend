// Lean compiler output
// Module: Mathlib.Topology.OpenPartialHomeomorph.Composition
// Imports: public import Init public import Mathlib.Topology.OpenPartialHomeomorph.IsImage
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
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_transOpenPartialHomeomorph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_transPartialHomeomorph___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_PartialEquiv_IsImage_restr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_transOpenPartialHomeomorph___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_transPartialHomeomorph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialEquiv_trans_x27___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_PartialEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_transPartialEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_trans_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_trans_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_trans_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_PartialEquiv_trans_x27___redArg(x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_trans_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PartialEquiv_trans_x27___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_trans___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_mathlib_PartialEquiv_symm___redArg(x_1);
x_4 = lp_mathlib_PartialEquiv_IsImage_restr___redArg(x_3);
x_5 = lp_mathlib_PartialEquiv_symm___redArg(x_4);
x_6 = lp_mathlib_PartialEquiv_IsImage_restr___redArg(x_2);
x_7 = lp_mathlib_PartialEquiv_trans_x27___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OpenPartialHomeomorph_trans(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_OpenPartialHomeomorph_trans___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_transOpenPartialHomeomorph(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Equiv_transPartialEquiv___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_transOpenPartialHomeomorph___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_transPartialEquiv___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_transPartialHomeomorph(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Equiv_transPartialEquiv___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_transPartialHomeomorph___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_transPartialEquiv___redArg(x_1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_OpenPartialHomeomorph_IsImage(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_OpenPartialHomeomorph_Composition(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_OpenPartialHomeomorph_IsImage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
