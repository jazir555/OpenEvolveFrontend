// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.ToLinearEquiv
// Imports: public import Init public import Mathlib.LinearAlgebra.Matrix.GeneralLinearGroup.Defs public import Mathlib.LinearAlgebra.Matrix.Nondegenerate public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse public import Mathlib.LinearAlgebra.Matrix.ToLin public import Mathlib.RingTheory.Localization.FractionRing public import Mathlib.RingTheory.Localization.Integer
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
lean_object* lp_mathlib_Pi_Function_module___redArg(lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquiv_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquiv_x27___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_GeneralLinearGroup_toLin___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquiv_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquiv_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquiv_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquiv_x27___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Matrix_toLinearEquiv_x27___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquiv_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_6 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_6);
lean_inc_ref(x_2);
x_7 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_2);
x_8 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toLinearEquiv_x27___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_10, 0, x_9);
x_11 = lp_mathlib_Pi_addCommMonoid___redArg(x_10);
x_12 = lp_mathlib_Matrix_GeneralLinearGroup_toLin___redArg(x_3, x_1, x_2);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc_ref(x_6);
x_14 = lp_mathlib_Semiring_toModule___redArg(x_6);
x_15 = lp_mathlib_Pi_Function_module___redArg(x_14);
x_16 = lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv___redArg(x_6, x_11, x_15);
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_ctor_get(x_16, 0);
x_19 = lean_ctor_get(x_16, 1);
lean_dec(x_19);
lean_ctor_set(x_16, 1, x_5);
lean_ctor_set(x_16, 0, x_4);
x_20 = lean_apply_1(x_13, x_16);
x_21 = lean_apply_1(x_18, x_20);
return x_21;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_22 = lean_ctor_get(x_16, 0);
lean_inc(x_22);
lean_dec(x_16);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_4);
lean_ctor_set(x_23, 1, x_5);
x_24 = lean_apply_1(x_13, x_23);
x_25 = lean_apply_1(x_22, x_24);
return x_25;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquiv_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_toLinearEquiv_x27___redArg(x_2, x_4, x_5, x_6, x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_GeneralLinearGroup_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Nondegenerate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_NonsingularInverse(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Integer(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLinearEquiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_GeneralLinearGroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Nondegenerate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_NonsingularInverse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_Integer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
