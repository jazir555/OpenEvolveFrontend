// Lean compiler output
// Module: Mathlib.Algebra.Azumaya.Matrix
// Imports: public import Init public import Mathlib.Algebra.Azumaya.Defs public import Mathlib.LinearAlgebra.FreeModule.Finite.Basic
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
lean_object* lp_mathlib_TensorProduct_SMul_aux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddCon_lift___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_addCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_TensorProduct_tmul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_addMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instSMul___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc(x_12);
lean_dec_ref(x_10);
x_13 = lean_ctor_get(x_11, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_11, 1);
lean_inc(x_14);
lean_dec(x_11);
x_15 = lean_ctor_get(x_12, 0);
lean_inc(x_15);
x_16 = lean_ctor_get(x_12, 1);
lean_inc(x_16);
lean_dec(x_12);
lean_inc(x_3);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_2);
lean_inc_ref_n(x_1, 2);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_single), 11, 9);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, lean_box(0));
lean_closure_set(x_17, 2, lean_box(0));
lean_closure_set(x_17, 3, x_1);
lean_closure_set(x_17, 4, x_1);
lean_closure_set(x_17, 5, x_2);
lean_closure_set(x_17, 6, x_14);
lean_closure_set(x_17, 7, x_15);
lean_closure_set(x_17, 8, x_3);
lean_inc(x_16);
lean_inc(x_13);
x_18 = lean_apply_3(x_4, x_17, x_13, x_16);
lean_inc(x_3);
lean_inc(x_2);
lean_inc_ref_n(x_1, 2);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_single), 11, 9);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, lean_box(0));
lean_closure_set(x_19, 2, lean_box(0));
lean_closure_set(x_19, 3, x_1);
lean_closure_set(x_19, 4, x_1);
lean_closure_set(x_19, 5, x_2);
lean_closure_set(x_19, 6, x_13);
lean_closure_set(x_19, 7, x_14);
lean_closure_set(x_19, 8, x_3);
lean_inc_ref(x_1);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_single), 11, 9);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, lean_box(0));
lean_closure_set(x_20, 2, lean_box(0));
lean_closure_set(x_20, 3, x_1);
lean_closure_set(x_20, 4, x_1);
lean_closure_set(x_20, 5, x_2);
lean_closure_set(x_20, 6, x_15);
lean_closure_set(x_20, 7, x_16);
lean_closure_set(x_20, 8, x_3);
x_21 = lp_mathlib_TensorProduct_tmul___redArg(x_19, x_20);
lean_inc(x_8);
x_22 = lp_mathlib_TensorProduct_SMul_aux___redArg(x_5, x_6, x_7, x_8, x_9, x_8, x_18);
lean_dec(x_8);
x_23 = lp_mathlib_AddCon_lift___redArg(x_22);
x_24 = lean_apply_1(x_23, x_21);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_alloc_closure((void*)(lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__0___boxed), 10, 9);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_2);
lean_closure_set(x_12, 2, x_3);
lean_closure_set(x_12, 3, x_11);
lean_closure_set(x_12, 4, x_4);
lean_closure_set(x_12, 5, x_5);
lean_closure_set(x_12, 6, x_6);
lean_closure_set(x_12, 7, x_7);
lean_closure_set(x_12, 8, x_8);
lean_inc(x_9);
x_13 = lp_mathlib_Multiset_product___redArg(x_9, x_9);
lean_inc(x_13);
x_14 = lp_mathlib_Multiset_product___redArg(x_13, x_13);
x_15 = lp_mathlib_Finset_sum___redArg(x_10, x_14, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_6 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
x_9 = lp_mathlib_Matrix_addCommMonoid___redArg(x_8);
lean_inc_ref(x_3);
x_10 = lp_mathlib_Semiring_toModule___redArg(x_3);
x_11 = lp_mathlib_Matrix_module___redArg(x_10);
lean_inc_ref(x_9);
x_12 = lp_mathlib_MulOpposite_instAddCommMonoid___redArg(x_9);
lean_inc(x_11);
x_13 = lp_mathlib_MulOpposite_instSMul___redArg(x_11);
lean_inc(x_13);
lean_inc(x_11);
lean_inc_ref(x_12);
lean_inc_ref(x_9);
lean_inc_ref(x_3);
x_14 = lp_mathlib_TensorProduct_addMonoid___redArg(x_3, x_9, x_12, x_11, x_13);
x_15 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_7);
x_16 = lean_ctor_get(x_15, 1);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_6);
x_18 = lean_ctor_get(x_17, 2);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_alloc_closure((void*)(lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__1___boxed), 11, 10);
lean_closure_set(x_19, 0, x_5);
lean_closure_set(x_19, 1, x_16);
lean_closure_set(x_19, 2, x_18);
lean_closure_set(x_19, 3, x_3);
lean_closure_set(x_19, 4, x_9);
lean_closure_set(x_19, 5, x_12);
lean_closure_set(x_19, 6, x_11);
lean_closure_set(x_19, 7, x_13);
lean_closure_set(x_19, 8, x_4);
lean_closure_set(x_19, 9, x_14);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
x_7 = lp_mathlib_Matrix_addCommMonoid___redArg(x_6);
lean_inc_ref(x_1);
x_8 = lp_mathlib_Semiring_toModule___redArg(x_1);
x_9 = lp_mathlib_Matrix_module___redArg(x_8);
lean_inc_ref(x_7);
x_10 = lp_mathlib_MulOpposite_instAddCommMonoid___redArg(x_7);
lean_inc(x_9);
x_11 = lp_mathlib_MulOpposite_instSMul___redArg(x_9);
lean_inc(x_11);
lean_inc(x_9);
lean_inc_ref(x_10);
lean_inc_ref(x_7);
lean_inc_ref(x_1);
x_12 = lp_mathlib_TensorProduct_addMonoid___redArg(x_1, x_7, x_10, x_9, x_11);
x_13 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_5);
x_14 = lean_ctor_get(x_13, 1);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_16 = lean_ctor_get(x_15, 2);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_alloc_closure((void*)(lp_mathlib_AlgHom_mulLeftRightMatrix__inv___redArg___lam__1___boxed), 11, 10);
lean_closure_set(x_17, 0, x_3);
lean_closure_set(x_17, 1, x_14);
lean_closure_set(x_17, 2, x_16);
lean_closure_set(x_17, 3, x_1);
lean_closure_set(x_17, 4, x_7);
lean_closure_set(x_17, 5, x_10);
lean_closure_set(x_17, 6, x_9);
lean_closure_set(x_17, 7, x_11);
lean_closure_set(x_17, 8, x_2);
lean_closure_set(x_17, 9, x_12);
return x_17;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Azumaya_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_FreeModule_Finite_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Azumaya_Matrix(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Azumaya_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_FreeModule_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
