// Lean compiler output
// Module: Mathlib.Data.Set.Finite.Lattice
// Imports: public import Init public import Mathlib.Data.Set.Finite.Powerset public import Mathlib.Data.Set.Finite.Range public import Mathlib.Data.Set.Lattice.Image
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
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_biUnion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypesUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypesUnion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_toFinset___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_Set_toFinset___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeiUnion___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lp_mathlib_Finset_biUnion___redArg(x_1, x_2, x_4);
x_6 = lp_mathlib_Fintype_subtype___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Set_fintypeiUnion___redArg(x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypesUnion___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_PLift_fintype___redArg(x_2);
x_5 = lp_mathlib_Set_fintypeiUnion___redArg(x_1, x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypesUnion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_fintypesUnion___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_2(x_1, x_2, lean_box(0));
x_4 = lp_mathlib_Set_toFinset___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeBiUnion___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lp_mathlib_Set_toFinset___redArg(x_2);
x_6 = lp_mathlib_Multiset_attach___redArg(x_5);
x_7 = lp_mathlib_Finset_biUnion___redArg(x_1, x_6, x_4);
x_8 = lp_mathlib_Fintype_subtype___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Set_fintypeBiUnion___redArg(x_2, x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_Set_toFinset___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeBiUnion_x27___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lp_mathlib_Set_toFinset___redArg(x_2);
x_6 = lp_mathlib_Finset_biUnion___redArg(x_1, x_5, x_4);
x_7 = lp_mathlib_Fintype_subtype___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Set_fintypeBiUnion_x27___redArg(x_2, x_5, x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
