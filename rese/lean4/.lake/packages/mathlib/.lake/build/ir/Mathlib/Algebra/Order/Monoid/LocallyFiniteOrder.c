// Lean compiler output
// Module: Mathlib.Algebra.Order.Monoid.LocallyFiniteOrder
// Imports: public import Init public import Mathlib.Algebra.Group.Subgroup.Ker public import Mathlib.Algebra.Order.Group.Units public import Mathlib.Algebra.Order.Hom.MonoidWithZero public import Mathlib.Algebra.Order.Hom.TypeTags public import Mathlib.Algebra.Order.Ring.Int public import Mathlib.Data.Nat.Cast.Order.Ring public import Mathlib.Tactic.Abel public import Mathlib.Algebra.Group.Embedding public import Mathlib.Order.Interval.Finset.Basic
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
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg(lean_object*, lean_object*);
lean_object* l_instNatCastInt___lam__0(lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_Ico___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc(x_4);
lean_inc(x_2);
lean_inc_ref(x_1);
x_5 = lp_mathlib_Finset_Ico___redArg(x_1, x_2, x_4);
x_6 = l_List_lengthTR___redArg(x_5);
lean_dec(x_5);
x_7 = l_instNatCastInt___lam__0(x_6);
x_8 = lean_apply_1(x_3, x_4);
x_9 = lp_mathlib_Finset_Ico___redArg(x_1, x_2, x_8);
x_10 = l_List_lengthTR___redArg(x_9);
lean_dec(x_9);
x_11 = l_instNatCastInt___lam__0(x_10);
x_12 = lean_int_sub(x_7, x_11);
lean_dec(x_11);
lean_dec(x_7);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg___lam__0), 4, 3);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_4);
lean_closure_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrder_addMonoidHom(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LocallyFiniteOrder_addMonoidHom___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LocallyFiniteOrder_orderAddMonoidHom___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_TypeTags(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Embedding(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_LocallyFiniteOrder(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
