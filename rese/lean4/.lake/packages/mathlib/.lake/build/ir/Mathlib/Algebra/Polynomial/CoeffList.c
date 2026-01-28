// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.CoeffList
// Imports: public import Init public import Mathlib.Algebra.Polynomial.Degree.Definitions public import Mathlib.Algebra.Polynomial.EraseLead public import Mathlib.Data.List.Range
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
lean_object* lp_mathlib_WithBot_succ___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeffList___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Polynomial_coeffList___redArg___closed__0;
lean_object* lp_mathlib_Nat_instSuccOrder___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeffList(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instPreorder;
lean_object* lp_mathlib_Polynomial_degree___redArg(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_mathlib_Polynomial_coeff___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Polynomial_coeffList___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_instSuccOrder___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeffList___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_3 = lp_mathlib_Nat_instPreorder;
x_4 = lean_unsigned_to_nat(0u);
x_5 = lp_mathlib_Polynomial_coeffList___redArg___closed__0;
lean_inc_ref(x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Polynomial_coeff___boxed), 4, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_2);
x_7 = lp_mathlib_Polynomial_degree___redArg(x_2);
x_8 = lp_mathlib_WithBot_succ___redArg(x_3, x_4, x_5, x_7);
x_9 = l_List_range(x_8);
x_10 = l_List_reverse___redArg(x_9);
x_11 = lean_box(0);
x_12 = l_List_mapTR_loop___redArg(x_6, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeffList(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Polynomial_coeffList___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Definitions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Range(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_CoeffList(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Definitions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Polynomial_coeffList___redArg___closed__0 = _init_lp_mathlib_Polynomial_coeffList___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Polynomial_coeffList___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
