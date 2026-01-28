// Lean compiler output
// Module: Mathlib.Data.Fintype.Order
// Imports: public import Init public import Mathlib.Data.Finset.Lattice.Fold public import Mathlib.Data.Finset.Order public import Mathlib.Data.Set.Finite.Basic public import Mathlib.Data.Set.Finite.Range public import Mathlib.Order.Atoms
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
lean_object* lp_mathlib_Finset_inf_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toBoundedOrder___redArg(lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toBoundedOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderTop___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderBot___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
static lean_object* lp_mathlib_Fintype_toOrderBot___closed__0;
static lean_object* _init_lp_mathlib_Fintype_toOrderBot___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderBot(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Fintype_toOrderBot___closed__0;
x_6 = lp_mathlib_Finset_inf_x27___redArg(x_4, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderBot___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Fintype_toOrderBot___closed__0;
x_4 = lp_mathlib_Finset_inf_x27___redArg(x_2, x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderTop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Fintype_toOrderBot___closed__0;
x_6 = lp_mathlib_Finset_sup_x27___redArg(x_4, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderTop___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Fintype_toOrderBot___closed__0;
x_4 = lp_mathlib_Finset_sup_x27___redArg(x_2, x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toBoundedOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc_ref(x_4);
x_5 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_4);
x_6 = !lean_is_exclusive(x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lean_ctor_get(x_4, 0);
x_8 = lean_ctor_get(x_4, 1);
lean_dec(x_8);
x_9 = lp_mathlib_Fintype_toOrderBot___closed__0;
lean_inc(x_2);
x_10 = lp_mathlib_Finset_inf_x27___redArg(x_5, x_2, x_9);
x_11 = lp_mathlib_Finset_sup_x27___redArg(x_7, x_2, x_9);
lean_ctor_set(x_4, 1, x_10);
lean_ctor_set(x_4, 0, x_11);
return x_4;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = lean_ctor_get(x_4, 0);
lean_inc(x_12);
lean_dec(x_4);
x_13 = lp_mathlib_Fintype_toOrderBot___closed__0;
lean_inc(x_2);
x_14 = lp_mathlib_Finset_inf_x27___redArg(x_5, x_2, x_13);
x_15 = lp_mathlib_Finset_sup_x27___redArg(x_12, x_2, x_13);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_14);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toBoundedOrder___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_2);
x_3 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_2);
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_2, 1);
lean_dec(x_6);
x_7 = lp_mathlib_Fintype_toOrderBot___closed__0;
lean_inc(x_1);
x_8 = lp_mathlib_Finset_inf_x27___redArg(x_3, x_1, x_7);
x_9 = lp_mathlib_Finset_sup_x27___redArg(x_5, x_1, x_7);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_9);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_2, 0);
lean_inc(x_10);
lean_dec(x_2);
x_11 = lp_mathlib_Fintype_toOrderBot___closed__0;
lean_inc(x_1);
x_12 = lp_mathlib_Finset_inf_x27___redArg(x_3, x_1, x_11);
x_13 = lp_mathlib_Finset_sup_x27___redArg(x_10, x_1, x_11);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_12);
return x_14;
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Atoms(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Atoms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Fintype_toOrderBot___closed__0 = _init_lp_mathlib_Fintype_toOrderBot___closed__0();
lean_mark_persistent(lp_mathlib_Fintype_toOrderBot___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
