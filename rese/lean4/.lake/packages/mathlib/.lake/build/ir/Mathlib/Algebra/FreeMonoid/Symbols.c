// Lean compiler output
// Module: Mathlib.Algebra.FreeMonoid.Symbols
// Imports: public import Init public import Mathlib.Algebra.FreeMonoid.Basic public import Mathlib.Data.Finset.Lattice.Lemmas
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
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_symbols___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_symbols___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_symbols(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_symbols(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_symbols(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_List_dedup___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_symbols___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_List_dedup___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_symbols(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_List_dedup___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_symbols___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_List_dedup___redArg(x_1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_FreeMonoid_Symbols(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
