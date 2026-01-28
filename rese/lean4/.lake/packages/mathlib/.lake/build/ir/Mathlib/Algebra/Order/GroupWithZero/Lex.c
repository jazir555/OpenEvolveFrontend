// Lean compiler output
// Module: Mathlib.Algebra.Order.GroupWithZero.Lex
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.ProdHom public import Mathlib.Algebra.Order.Group.Equiv public import Mathlib.Algebra.Order.Monoid.Lex public import Mathlib.Algebra.Order.Hom.MonoidWithZero public import Mathlib.Data.Prod.Lex
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
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZeroHom_inr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inl(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZeroHom_fst___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_map_x27___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Units_instMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZeroHom_inl___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_fst___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_fst(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instMulOneClass___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZeroHom_comp___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_fst___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inr___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_1, 5);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_3, x_2);
x_6 = lean_unbox(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(x_1);
x_4 = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_inc_ref(x_5);
x_6 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_5);
x_7 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lp_mathlib_Units_instMulOneClass___redArg(x_7);
lean_dec_ref(x_7);
x_9 = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(x_2);
x_10 = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
x_13 = lp_mathlib_Units_instMulOneClass___redArg(x_12);
lean_dec_ref(x_12);
x_14 = lp_mathlib_Prod_instMulOneClass___redArg(x_8, x_13);
x_15 = lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___closed__0;
x_16 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_16);
lean_dec_ref(x_1);
x_17 = lean_ctor_get(x_15, 0);
lean_inc(x_17);
x_18 = lean_ctor_get(x_16, 1);
lean_inc_ref(x_18);
lean_dec_ref(x_16);
x_19 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_6);
x_20 = lean_ctor_get(x_19, 1);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lp_mathlib_WithZero_map_x27___redArg(x_14, x_17);
x_22 = lean_alloc_closure((void*)(lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_22, 0, x_18);
lean_closure_set(x_22, 1, x_20);
x_23 = lp_mathlib_MonoidWithZeroHom_inl___redArg(x_4, x_10, x_22);
lean_dec_ref(x_10);
x_24 = lp_mathlib_MonoidWithZeroHom_comp___redArg(x_21, x_23);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inr___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
lean_inc_ref(x_2);
x_3 = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(x_2);
x_4 = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_inc_ref(x_5);
x_6 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_5);
x_7 = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(x_1);
x_8 = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_Units_instMulOneClass___redArg(x_10);
lean_dec_ref(x_10);
x_12 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_5);
x_13 = lp_mathlib_Units_instMulOneClass___redArg(x_12);
lean_dec_ref(x_12);
x_14 = lp_mathlib_Prod_instMulOneClass___redArg(x_11, x_13);
x_15 = lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___closed__0;
x_16 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_16);
lean_dec_ref(x_2);
x_17 = lean_ctor_get(x_15, 0);
lean_inc(x_17);
x_18 = lean_ctor_get(x_16, 1);
lean_inc_ref(x_18);
lean_dec_ref(x_16);
x_19 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_6);
x_20 = lean_ctor_get(x_19, 1);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lp_mathlib_WithZero_map_x27___redArg(x_14, x_17);
x_22 = lean_alloc_closure((void*)(lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_22, 0, x_18);
lean_closure_set(x_22, 1, x_20);
x_23 = lp_mathlib_MonoidWithZeroHom_inr___redArg(x_8, x_4, x_22);
lean_dec_ref(x_8);
x_24 = lp_mathlib_MonoidWithZeroHom_comp___redArg(x_21, x_23);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_inr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LinearOrderedCommGroupWithZero_inr___redArg(x_3, x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_LinearOrderedCommGroupWithZero_fst___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_fst___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_3 = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(x_1);
x_4 = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_Units_instMulOneClass___redArg(x_6);
lean_dec_ref(x_6);
x_8 = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(x_2);
x_9 = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
lean_dec_ref(x_10);
x_12 = lp_mathlib_Units_instMulOneClass___redArg(x_11);
lean_dec_ref(x_11);
x_13 = lp_mathlib_Prod_instMulOneClass___redArg(x_7, x_12);
x_14 = lp_mathlib_LinearOrderedCommGroupWithZero_fst___redArg___closed__0;
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
x_16 = lp_mathlib_MonoidWithZeroHom_fst___redArg(x_4);
x_17 = lp_mathlib_WithZero_map_x27___redArg(x_13, x_15);
x_18 = lp_mathlib_MonoidWithZeroHom_comp___redArg(x_16, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_fst(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LinearOrderedCommGroupWithZero_fst___redArg(x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_ProdHom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Lex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Prod_Lex(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Lex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_ProdHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Prod_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___closed__0 = _init_lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___closed__0();
lean_mark_persistent(lp_mathlib_LinearOrderedCommGroupWithZero_inl___redArg___closed__0);
lp_mathlib_LinearOrderedCommGroupWithZero_fst___redArg___closed__0 = _init_lp_mathlib_LinearOrderedCommGroupWithZero_fst___redArg___closed__0();
lean_mark_persistent(lp_mathlib_LinearOrderedCommGroupWithZero_fst___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
