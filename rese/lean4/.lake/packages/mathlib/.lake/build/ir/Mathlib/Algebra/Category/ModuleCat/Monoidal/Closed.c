// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Monoidal.Closed
// Imports: public import Init public import Mathlib.CategoryTheory.Monoidal.Closed.Basic public import Mathlib.CategoryTheory.Linear.Yoneda public import Mathlib.Algebra.Category.ModuleCat.Monoidal.Symmetric
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
lean_object* lp_mathlib_CategoryTheory_linearCoyoneda___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_monoidalClosedHomEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instMonoidalClosed___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_liftAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_comm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_MonoidalCategory_curriedTensor___redArg(lean_object*);
lean_object* lp_mathlib_ModuleCat_instLinear___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instMonoidalClosed___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_instAddCommGroupHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_moduleCategory(lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instMonoidalCategoryStruct___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_toModuleIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instMonoidalClosed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instMonoidalClosed___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_Hom_hom_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Adjunction_mkOfHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_ofHom_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_compr_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_11 = lp_mathlib_TensorProduct_comm___redArg(x_1, x_2, x_3, x_4, x_5);
x_12 = lp_mathlib_LinearEquiv_toModuleIso___redArg(x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_TensorProduct_mk(lean_box(0), x_1, lean_box(0), lean_box(0), x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_15 = lp_mathlib_LinearMap_comp___redArg(x_9, x_13);
x_16 = lp_mathlib_LinearMap_compr_u2082___redArg(x_14, x_15);
x_17 = lp_mathlib_ModuleCat_ofHom_u2082___redArg(x_6, x_7, x_8, x_16);
x_18 = lean_apply_1(x_17, x_10);
return x_18;
}
}
static lean_object* _init_lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc(x_4);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_12 = lp_mathlib_TensorProduct_comm___redArg(x_1, x_2, x_3, x_4, x_5);
x_13 = lp_mathlib_LinearEquiv_toModuleIso___redArg(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1___closed__0;
x_16 = lp_mathlib_ModuleCat_Hom_hom_u2082___redArg(x_6, x_7, x_8, x_11);
lean_inc_ref(x_1);
x_17 = lp_mathlib_TensorProduct_liftAux___redArg(x_1, x_1, x_15, x_2, x_9, x_4, x_10, x_16);
x_18 = lp_mathlib_LinearMap_comp___redArg(x_17, x_14);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_4, 0);
x_8 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_8);
x_9 = !lean_is_exclusive(x_3);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_10 = lean_ctor_get(x_3, 1);
x_11 = lean_ctor_get(x_3, 0);
lean_dec(x_11);
x_12 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_5);
x_13 = lean_ctor_get(x_2, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_14);
x_15 = lean_ctor_get(x_4, 1);
lean_inc(x_15);
x_16 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_16);
lean_inc_ref(x_4);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc(x_13);
lean_inc(x_10);
lean_inc_ref(x_14);
lean_inc_ref(x_12);
lean_inc_ref(x_8);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__0), 10, 8);
lean_closure_set(x_17, 0, x_8);
lean_closure_set(x_17, 1, x_12);
lean_closure_set(x_17, 2, x_14);
lean_closure_set(x_17, 3, x_10);
lean_closure_set(x_17, 4, x_13);
lean_closure_set(x_17, 5, x_1);
lean_closure_set(x_17, 6, x_2);
lean_closure_set(x_17, 7, x_4);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1), 11, 10);
lean_closure_set(x_18, 0, x_8);
lean_closure_set(x_18, 1, x_14);
lean_closure_set(x_18, 2, x_12);
lean_closure_set(x_18, 3, x_13);
lean_closure_set(x_18, 4, x_10);
lean_closure_set(x_18, 5, x_1);
lean_closure_set(x_18, 6, x_2);
lean_closure_set(x_18, 7, x_4);
lean_closure_set(x_18, 8, x_16);
lean_closure_set(x_18, 9, x_15);
lean_ctor_set(x_3, 1, x_18);
lean_ctor_set(x_3, 0, x_17);
return x_3;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_19 = lean_ctor_get(x_3, 1);
lean_inc(x_19);
lean_dec(x_3);
x_20 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_20);
lean_dec_ref(x_5);
x_21 = lean_ctor_get(x_2, 1);
lean_inc(x_21);
x_22 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_22);
x_23 = lean_ctor_get(x_4, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_24);
lean_inc_ref(x_4);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc(x_21);
lean_inc(x_19);
lean_inc_ref(x_22);
lean_inc_ref(x_20);
lean_inc_ref(x_8);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__0), 10, 8);
lean_closure_set(x_25, 0, x_8);
lean_closure_set(x_25, 1, x_20);
lean_closure_set(x_25, 2, x_22);
lean_closure_set(x_25, 3, x_19);
lean_closure_set(x_25, 4, x_21);
lean_closure_set(x_25, 5, x_1);
lean_closure_set(x_25, 6, x_2);
lean_closure_set(x_25, 7, x_4);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1), 11, 10);
lean_closure_set(x_26, 0, x_8);
lean_closure_set(x_26, 1, x_22);
lean_closure_set(x_26, 2, x_20);
lean_closure_set(x_26, 3, x_21);
lean_closure_set(x_26, 4, x_19);
lean_closure_set(x_26, 5, x_1);
lean_closure_set(x_26, 6, x_2);
lean_closure_set(x_26, 7, x_4);
lean_closure_set(x_26, 8, x_24);
lean_closure_set(x_26, 9, x_23);
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_25);
lean_ctor_set(x_27, 1, x_26);
return x_27;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_monoidalClosedHomEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instMonoidalClosed___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instMonoidalClosed___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
lean_inc_ref(x_1);
x_7 = lp_mathlib_CategoryTheory_linearCoyoneda___redArg(x_1, x_2, x_3);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lp_mathlib_CategoryTheory_MonoidalCategory_curriedTensor___redArg(x_4);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_ctor_get(x_9, 1);
lean_dec(x_12);
lean_inc_ref(x_6);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_instMonoidalClosed___redArg___lam__0), 4, 2);
lean_closure_set(x_13, 0, x_5);
lean_closure_set(x_13, 1, x_6);
lean_inc_ref(x_6);
x_14 = lean_apply_1(x_8, x_6);
x_15 = lean_apply_1(x_11, x_6);
lean_inc_ref(x_14);
lean_inc_ref(x_1);
x_16 = lp_mathlib_CategoryTheory_Adjunction_mkOfHomEquiv___redArg(x_1, x_1, x_15, x_14, x_13);
lean_ctor_set(x_9, 1, x_16);
lean_ctor_set(x_9, 0, x_14);
return x_9;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_17 = lean_ctor_get(x_9, 0);
lean_inc(x_17);
lean_dec(x_9);
lean_inc_ref(x_6);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_instMonoidalClosed___redArg___lam__0), 4, 2);
lean_closure_set(x_18, 0, x_5);
lean_closure_set(x_18, 1, x_6);
lean_inc_ref(x_6);
x_19 = lean_apply_1(x_8, x_6);
x_20 = lean_apply_1(x_17, x_6);
lean_inc_ref(x_19);
lean_inc_ref(x_1);
x_21 = lp_mathlib_CategoryTheory_Adjunction_mkOfHomEquiv___redArg(x_1, x_1, x_20, x_19, x_18);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_19);
lean_ctor_set(x_22, 1, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instMonoidalClosed___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_ModuleCat_moduleCategory(lean_box(0), x_1);
lean_inc_ref(x_1);
x_3 = lp_mathlib_ModuleCat_MonoidalCategory_instMonoidalCategoryStruct___redArg(x_1);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_instAddCommGroupHom___boxed), 4, 2);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_1);
x_5 = lp_mathlib_ModuleCat_instLinear___redArg(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_instMonoidalClosed___redArg___lam__1), 6, 5);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_4);
lean_closure_set(x_6, 2, x_5);
lean_closure_set(x_6, 3, x_3);
lean_closure_set(x_6, 4, x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instMonoidalClosed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ModuleCat_instMonoidalClosed___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Closed_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Linear_Yoneda(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Symmetric(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Closed(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Closed_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Linear_Yoneda(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Symmetric(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1___closed__0 = _init_lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1___closed__0();
lean_mark_persistent(lp_mathlib_ModuleCat_monoidalClosedHomEquiv___redArg___lam__1___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
