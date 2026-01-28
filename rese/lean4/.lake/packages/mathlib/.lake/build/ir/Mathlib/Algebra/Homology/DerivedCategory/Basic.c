// Lean compiler output
// Module: Mathlib.Algebra.Homology.DerivedCategory.Basic
// Imports: public import Init public import Mathlib.Algebra.Homology.HomotopyCategory.Acyclic public import Mathlib.Algebra.Homology.HomotopyCategory.SingleFunctors public import Mathlib.Algebra.Homology.HomotopyCategory.Triangulated
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
lean_object* lp_mathlib_HomologicalComplex_instCategory___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HasDerivedCategory_standard___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Q___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Qh___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_instCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_HomologicalComplexUpToQuasiIso_quotientCompQhIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Qh(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_MorphismProperty_HasLocalization_standard___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Q(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Q___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_instCategory___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_instCategory___redArg___boxed(lean_object*);
lean_object* lp_mathlib_HomologicalComplexUpToQuasiIso_Qh___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Q___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Qh___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HasDerivedCategory_standard(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_quotientCompQhIso(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_instCategory___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_quotientCompQhIso___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HasDerivedCategory_standard___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_2);
x_4 = lean_box(0);
x_5 = lp_mathlib_HomologicalComplex_instCategory___redArg(x_1, x_3, x_4);
x_6 = lp_mathlib_CategoryTheory_MorphismProperty_HasLocalization_standard___redArg(x_5);
lean_dec_ref(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HasDerivedCategory_standard(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_HasDerivedCategory_standard___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_instCategory(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_instCategory___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_instCategory___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DerivedCategory_instCategory(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_instCategory___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DerivedCategory_instCategory___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Q(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Q___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Q___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DerivedCategory_Q(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Q___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DerivedCategory_Q___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Qh(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_HomologicalComplexUpToQuasiIso_Qh___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Qh___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_HomologicalComplexUpToQuasiIso_Qh___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_Qh___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DerivedCategory_Qh(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_quotientCompQhIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_box(0);
x_5 = lp_mathlib_HomologicalComplexUpToQuasiIso_quotientCompQhIso___redArg(x_1, x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DerivedCategory_quotientCompQhIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DerivedCategory_quotientCompQhIso___redArg(x_2, x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Acyclic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_SingleFunctors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Triangulated(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Homology_DerivedCategory_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Acyclic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_SingleFunctors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Triangulated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
