// Lean compiler output
// Module: Mathlib.Data.Finset.Lattice.Fold
// Imports: public import Init public import Mathlib.Data.Finset.Fold public import Mathlib.Data.Finset.Sum public import Mathlib.Data.Multiset.Lattice public import Mathlib.Data.Set.BooleanAlgebra public import Mathlib.Order.Hom.BoundedLattice public import Mathlib.Order.Nat
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
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf_x27___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Finset_inf_x27___redArg___closed__0;
lean_object* lp_mathlib_WithBot_some(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_some(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Finset_sup_x27___redArg___closed__0;
lean_object* lp_mathlib_WithBot_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Finset_sup___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lp_mathlib_Finset_fold___redArg(x_5, x_2, x_4, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Finset_sup___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Finset_inf___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lp_mathlib_Finset_fold___redArg(x_5, x_2, x_4, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Finset_inf___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Finset_sup_x27___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_WithBot_some), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lp_mathlib_WithBot_semilatticeSup___redArg(x_1);
x_5 = lean_box(0);
x_6 = lp_mathlib_Finset_sup_x27___redArg___closed__0;
x_7 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_6);
lean_closure_set(x_7, 4, x_3);
x_8 = lp_mathlib_Finset_sup___redArg(x_4, x_5, x_2, x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Finset_sup_x27___redArg(x_3, x_4, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Finset_inf_x27___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_WithTop_some), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lp_mathlib_WithTop_semilatticeInf___redArg(x_1);
x_5 = lean_box(0);
x_6 = lp_mathlib_Finset_inf_x27___redArg___closed__0;
x_7 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_6);
lean_closure_set(x_7, 4, x_3);
x_8 = lp_mathlib_Finset_inf___redArg(x_4, x_5, x_2, x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Finset_inf_x27___redArg(x_3, x_4, x_6);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Finset_sup_x27___redArg___closed__0 = _init_lp_mathlib_Finset_sup_x27___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Finset_sup_x27___redArg___closed__0);
lp_mathlib_Finset_inf_x27___redArg___closed__0 = _init_lp_mathlib_Finset_inf_x27___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Finset_inf_x27___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
