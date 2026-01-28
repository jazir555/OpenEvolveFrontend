// Lean compiler output
// Module: Mathlib.CategoryTheory.Abelian.DiagramLemmas.Four
// Imports: public import Init public import Mathlib.Algebra.Homology.ExactSequence public import Mathlib.CategoryTheory.Abelian.Refinements
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_unsigned_to_nat(0u);
x_8 = lean_nat_dec_eq(x_1, x_7);
if (x_8 == 1)
{
uint8_t x_9; 
lean_dec(x_6);
x_9 = lean_nat_dec_eq(x_2, x_7);
if (x_9 == 1)
{
lean_object* x_10; 
lean_dec(x_5);
lean_dec(x_4);
x_10 = lean_apply_3(x_3, lean_box(0), lean_box(0), lean_box(0));
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; 
lean_dec(x_3);
x_11 = lean_unsigned_to_nat(1u);
x_12 = lean_nat_sub(x_2, x_11);
x_13 = lean_nat_dec_eq(x_12, x_7);
if (x_13 == 1)
{
lean_object* x_14; 
lean_dec(x_12);
lean_dec(x_5);
x_14 = lean_apply_3(x_4, lean_box(0), lean_box(0), lean_box(0));
return x_14;
}
else
{
lean_object* x_15; lean_object* x_16; 
lean_dec(x_4);
x_15 = lean_nat_sub(x_12, x_11);
lean_dec(x_12);
x_16 = lean_apply_4(x_5, lean_box(0), x_15, lean_box(0), lean_box(0));
return x_16;
}
}
}
else
{
uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
x_17 = lean_nat_dec_eq(x_2, x_7);
x_18 = lean_unsigned_to_nat(1u);
x_19 = lean_nat_sub(x_1, x_18);
x_20 = lean_nat_sub(x_2, x_18);
x_21 = lean_apply_5(x_6, x_19, lean_box(0), x_20, lean_box(0), lean_box(0));
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter___redArg(x_3, x_4, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four_0__CategoryTheory_ComposableArrows_Precomp_map_match__1_splitter___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_2);
lean_dec(x_1);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_ExactSequence(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_Refinements(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_DiagramLemmas_Four(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_ExactSequence(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Abelian_Refinements(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
