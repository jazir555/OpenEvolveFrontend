// Lean compiler output
// Module: Mathlib.Algebra.Homology.HomotopyCategory.DegreewiseSplit
// Imports: public import Init public import Mathlib.Algebra.Homology.HomotopyCategory.Pretriangulated
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
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___closed__0;
lean_object* lp_mathlib_CategoryTheory_shiftFunctor___redArg(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_mathlib_CochainComplex_HomComplex_Cochain_rightShift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CochainComplex_HomComplex_Cocycle_equivHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_homOfDegreewiseSplit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_triangleOfDegreewiseSplit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CochainComplex_instHasShiftInt___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_ShortComplex_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_triangleOfDegreewiseSplit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_HomologicalComplex_eval___redArg(lean_object*);
static lean_object* lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg___closed__0;
lean_object* lp_mathlib_CochainComplex_HomComplex_Cochain_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_10 = lean_ctor_get(x_1, 2);
lean_inc(x_10);
lean_dec_ref(x_1);
x_11 = lean_ctor_get(x_2, 0);
lean_inc(x_11);
lean_dec(x_2);
lean_inc(x_7);
x_12 = lp_mathlib_HomologicalComplex_eval___redArg(x_7);
x_13 = lp_mathlib_CategoryTheory_ShortComplex_map___redArg(x_3, x_12);
x_14 = lean_ctor_get(x_13, 1);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lean_ctor_get(x_4, 0);
lean_inc(x_15);
lean_dec(x_4);
lean_inc_ref(x_5);
lean_inc(x_7);
x_16 = lean_apply_1(x_5, x_7);
x_17 = lean_ctor_get(x_16, 1);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_6, 0);
lean_inc(x_18);
x_19 = lean_ctor_get(x_6, 1);
lean_inc(x_19);
lean_dec(x_6);
lean_inc(x_8);
x_20 = lean_apply_1(x_5, x_8);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_inc(x_7);
x_22 = lean_apply_1(x_11, x_7);
lean_inc(x_8);
x_23 = lean_apply_1(x_15, x_8);
lean_inc(x_8);
x_24 = lean_apply_1(x_18, x_8);
x_25 = lean_apply_2(x_19, x_7, x_8);
lean_inc(x_10);
lean_inc(x_23);
lean_inc(x_14);
x_26 = lean_apply_5(x_10, x_14, x_24, x_23, x_25, x_21);
x_27 = lean_apply_5(x_10, x_22, x_14, x_23, x_17, x_26);
return x_27;
}
}
static lean_object* _init_lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 2);
lean_inc(x_7);
lean_inc(x_5);
lean_inc(x_7);
lean_inc_ref(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___lam__0), 9, 6);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_7);
lean_closure_set(x_8, 2, x_3);
lean_closure_set(x_8, 3, x_5);
lean_closure_set(x_8, 4, x_4);
lean_closure_set(x_8, 5, x_6);
x_9 = lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___closed__0;
x_10 = lean_alloc_closure((void*)(lp_mathlib_CochainComplex_HomComplex_Cochain_mk___boxed), 8, 7);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_1);
lean_closure_set(x_10, 2, x_2);
lean_closure_set(x_10, 3, x_7);
lean_closure_set(x_10, 4, x_5);
lean_closure_set(x_10, 5, x_9);
lean_closure_set(x_10, 6, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 2);
lean_inc(x_6);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_7 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_1, x_2);
x_8 = lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___closed__0;
x_9 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_7, x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
lean_inc(x_5);
x_11 = lean_apply_1(x_10, x_5);
lean_inc(x_6);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_12 = lp_mathlib_CochainComplex_HomComplex_Cocycle_equivHom___redArg(x_1, x_2, x_6, x_11);
x_13 = lp_mathlib_Equiv_symm___redArg(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg___closed__0;
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_16 = lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg(x_1, x_2, x_3, x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_CochainComplex_HomComplex_Cochain_rightShift___boxed), 11, 10);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_1);
lean_closure_set(x_17, 2, x_2);
lean_closure_set(x_17, 3, x_6);
lean_closure_set(x_17, 4, x_5);
lean_closure_set(x_17, 5, x_8);
lean_closure_set(x_17, 6, x_16);
lean_closure_set(x_17, 7, x_8);
lean_closure_set(x_17, 8, x_15);
lean_closure_set(x_17, 9, lean_box(0));
x_18 = lean_apply_1(x_14, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_homOfDegreewiseSplit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_triangleOfDegreewiseSplit___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 2);
lean_inc(x_7);
x_8 = lean_ctor_get(x_3, 3);
lean_inc(x_8);
x_9 = lean_ctor_get(x_3, 4);
lean_inc(x_9);
x_10 = lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg(x_1, x_2, x_3, x_4);
x_11 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_11, 0, x_5);
lean_ctor_set(x_11, 1, x_6);
lean_ctor_set(x_11, 2, x_7);
lean_ctor_set(x_11, 3, x_8);
lean_ctor_set(x_11, 4, x_9);
lean_ctor_set(x_11, 5, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_triangleOfDegreewiseSplit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CochainComplex_triangleOfDegreewiseSplit___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Pretriangulated(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_DegreewiseSplit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Pretriangulated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___closed__0 = _init_lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CochainComplex_cocycleOfDegreewiseSplit___redArg___closed__0);
lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg___closed__0 = _init_lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CochainComplex_homOfDegreewiseSplit___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
