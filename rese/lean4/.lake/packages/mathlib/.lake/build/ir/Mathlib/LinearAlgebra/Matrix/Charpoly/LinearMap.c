// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.Charpoly.LinearMap
// Imports: public import Init public import Mathlib.LinearAlgebra.Matrix.Charpoly.Coeff public import Mathlib.LinearAlgebra.Matrix.ToLin
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
lean_object* lp_mathlib_Fintype_linearCombination___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_lcomp_u209b_u2097___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_Function_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_isRepresentation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_PiToModule_fromMatrix___redArg___closed__0;
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromMatrix___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromEnd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromMatrix___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_llcomp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_toLinearMap___redArg(lean_object*);
lean_object* lp_mathlib_Pi_addCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_toMatrixAlgEquiv_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_isRepresentation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromEnd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromMatrix___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromEnd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromMatrix___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromMatrix___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PiToModule_fromMatrix___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_PiToModule_fromMatrix___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromMatrix___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_7 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_7);
x_8 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_3);
x_9 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_7);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_14);
lean_dec_ref(x_2);
x_15 = lean_alloc_closure((void*)(lp_mathlib_PiToModule_fromMatrix___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_15, 0, x_10);
x_16 = lean_alloc_closure((void*)(lp_mathlib_PiToModule_fromMatrix___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_16, 0, x_13);
lean_inc_ref(x_7);
x_17 = lp_mathlib_Semiring_toModule___redArg(x_7);
x_18 = lp_mathlib_Pi_Function_module___redArg(x_17);
x_19 = lp_mathlib_Pi_addCommMonoid___redArg(x_15);
x_20 = lp_mathlib_Pi_addCommMonoid___redArg(x_16);
x_21 = lp_mathlib_PiToModule_fromMatrix___redArg___closed__0;
lean_inc(x_4);
lean_inc_ref(x_14);
lean_inc(x_1);
x_22 = lp_mathlib_Fintype_linearCombination___redArg(x_1, x_14, x_4, x_5);
lean_inc(x_18);
lean_inc_ref_n(x_7, 3);
x_23 = lp_mathlib_LinearMap_llcomp___redArg(x_7, x_7, x_7, x_19, x_20, x_14, x_18, x_18, x_4, x_21, x_21, x_21);
x_24 = lean_apply_1(x_23, x_22);
x_25 = lp_mathlib_LinearMap_toMatrixAlgEquiv_x27___redArg(x_7, x_6, x_1);
x_26 = lp_mathlib_Equiv_symm___redArg(x_25);
x_27 = lp_mathlib_AlgEquiv_toLinearMap___redArg(x_26);
x_28 = lp_mathlib_LinearMap_comp___redArg(x_24, x_27);
return x_28;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromMatrix(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_PiToModule_fromMatrix___redArg(x_2, x_4, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromEnd___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_2);
x_6 = lp_mathlib_Fintype_linearCombination___redArg(x_1, x_5, x_3, x_4);
x_7 = lp_mathlib_LinearMap_lcomp_u209b_u2097___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromEnd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_PiToModule_fromEnd___redArg(x_2, x_4, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiToModule_fromEnd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_PiToModule_fromEnd(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_isRepresentation(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_box(0);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_isRepresentation___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_isRepresentation(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec(x_2);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Charpoly_Coeff(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLin(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Charpoly_LinearMap(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Charpoly_Coeff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PiToModule_fromMatrix___redArg___closed__0 = _init_lp_mathlib_PiToModule_fromMatrix___redArg___closed__0();
lean_mark_persistent(lp_mathlib_PiToModule_fromMatrix___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
