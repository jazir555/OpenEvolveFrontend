// Lean compiler output
// Module: Mathlib.GroupTheory.Perm.List
// Imports: public import Init public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Data.List.Rotate public import Mathlib.GroupTheory.Perm.Support
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static lean_object* lp_mathlib_List_formPerm___redArg___closed__2;
lean_object* lp_batteries_List_prod___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_swap(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_List_formPerm___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_List_formPerm(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_formPerm___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_List_formPerm___redArg___closed__0;
lean_object* lp_mathlib_Equiv_Perm_instMul___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_zipWithTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_List_formPerm___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_instMul___lam__0), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_List_formPerm___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_List_formPerm___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_formPerm___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_List_formPerm___redArg___closed__0;
x_4 = lp_mathlib_List_formPerm___redArg___closed__1;
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_swap), 4, 2);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_1);
if (lean_obj_tag(x_2) == 0)
{
x_6 = x_2;
goto block_10;
}
else
{
lean_object* x_11; 
x_11 = lean_ctor_get(x_2, 1);
lean_inc(x_11);
x_6 = x_11;
goto block_10;
}
block_10:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_List_formPerm___redArg___closed__2;
x_8 = l___private_Init_Data_List_Impl_0__List_zipWithTR_go(lean_box(0), lean_box(0), lean_box(0), x_5, x_2, x_6, x_7);
x_9 = lp_batteries_List_prod___redArg(x_3, x_4, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_formPerm(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_List_formPerm___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Rotate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Perm_Support(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Perm_List(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Rotate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Perm_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_List_formPerm___redArg___closed__0 = _init_lp_mathlib_List_formPerm___redArg___closed__0();
lean_mark_persistent(lp_mathlib_List_formPerm___redArg___closed__0);
lp_mathlib_List_formPerm___redArg___closed__1 = _init_lp_mathlib_List_formPerm___redArg___closed__1();
lean_mark_persistent(lp_mathlib_List_formPerm___redArg___closed__1);
lp_mathlib_List_formPerm___redArg___closed__2 = _init_lp_mathlib_List_formPerm___redArg___closed__2();
lean_mark_persistent(lp_mathlib_List_formPerm___redArg___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
