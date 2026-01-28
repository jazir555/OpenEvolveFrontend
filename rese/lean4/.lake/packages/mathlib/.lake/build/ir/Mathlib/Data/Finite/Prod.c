// Lean compiler output
// Module: Mathlib.Data.Finite.Prod
// Imports: public import Init public import Mathlib.Data.Set.Finite.Basic public import Mathlib.Data.Fintype.Prod public import Mathlib.Data.Fintype.Pi public import Mathlib.Algebra.Order.Group.Multiset public import Mathlib.Data.Vector.Basic public import Mathlib.Tactic.ApplyFun public import Mathlib.Data.ULift public import Mathlib.Data.Set.NAry
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
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOffDiag___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_offDiag___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeProd___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Set_fintypeImage___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOffDiag(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_toFinset___redArg(lean_object*);
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeProd___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Set_toFinset___redArg(x_1);
x_4 = lp_mathlib_Set_toFinset___redArg(x_2);
x_5 = lp_mathlib_Multiset_product___redArg(x_3, x_4);
x_6 = lp_mathlib_Fintype_subtype___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeProd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Set_fintypeProd___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOffDiag___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Set_toFinset___redArg(x_2);
x_4 = lp_mathlib_Finset_offDiag___redArg(x_1, x_3);
x_5 = lp_mathlib_Fintype_subtype___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOffDiag(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_fintypeOffDiag___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_apply_2(x_1, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeImage2___redArg___lam__0), 2, 1);
lean_closure_set(x_5, 0, x_2);
x_6 = lp_mathlib_Set_fintypeProd___redArg(x_3, x_4);
x_7 = lp_mathlib_Set_fintypeImage___redArg(x_1, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Set_fintypeImage2___redArg(x_4, x_5, x_8, x_9);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Vector_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_NAry(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finite_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Vector_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
