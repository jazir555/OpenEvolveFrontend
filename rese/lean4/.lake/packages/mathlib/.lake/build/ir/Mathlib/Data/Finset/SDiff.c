// Lean compiler output
// Module: Mathlib.Data.Finset.SDiff
// Imports: public import Init public import Mathlib.Data.Finset.Insert public import Mathlib.Data.Finset.Lattice.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_sub___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_sub___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Finset_instSDiff___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_instSDiff___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_Finset_instLattice___redArg(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Finset_instSDiff___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_box(0);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Insert(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
