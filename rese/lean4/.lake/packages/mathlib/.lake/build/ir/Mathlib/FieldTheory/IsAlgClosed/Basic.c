// Lean compiler output
// Module: Mathlib.FieldTheory.IsAlgClosed.Basic
// Imports: public import Init public import Mathlib.FieldTheory.Extension public import Mathlib.FieldTheory.Normal.Defs public import Mathlib.FieldTheory.Perfect public import Mathlib.RingTheory.Localization.Integral
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
lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubfieldClass_toField___redArg(lean_object*);
lean_object* lp_mathlib_AlgHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subalgebra_algebra_x27___redArg(lean_object*);
lean_object* lp_mathlib_AlgHom_codRestrict___redArg(lean_object*);
lean_object* lp_mathlib_Subalgebra_val___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
static lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__1;
static lean_object* _init_lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_AlgHom_codRestrict___redArg), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Subalgebra_val___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_6 = lp_mathlib_Field_toSemifield___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lp_mathlib_Field_toSemifield___redArg(x_2);
x_9 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
lean_inc_ref(x_3);
x_11 = lp_mathlib_SubfieldClass_toField___redArg(x_3);
x_12 = lp_mathlib_Field_toSemifield___redArg(x_11);
lean_dec_ref(x_11);
x_13 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc_ref(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib_Field_toSemifield___redArg(x_3);
lean_dec_ref(x_3);
x_16 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_15);
x_17 = lean_ctor_get(x_16, 0);
lean_inc_ref(x_17);
lean_dec_ref(x_16);
x_18 = lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__0;
lean_inc_ref(x_5);
x_19 = lp_mathlib_Subalgebra_algebra_x27___redArg(x_5);
x_20 = lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__1;
x_21 = lean_alloc_closure((void*)(lp_mathlib_AlgHom_comp___boxed), 13, 12);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, lean_box(0));
lean_closure_set(x_21, 2, lean_box(0));
lean_closure_set(x_21, 3, lean_box(0));
lean_closure_set(x_21, 4, x_7);
lean_closure_set(x_21, 5, x_10);
lean_closure_set(x_21, 6, x_14);
lean_closure_set(x_21, 7, x_17);
lean_closure_set(x_21, 8, x_4);
lean_closure_set(x_21, 9, x_19);
lean_closure_set(x_21, 10, x_5);
lean_closure_set(x_21, 11, x_20);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_21);
lean_ctor_set(x_22, 1, x_18);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg(x_4, x_5, x_6, x_7, x_8);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_FieldTheory_Extension(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_FieldTheory_Normal_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_FieldTheory_Perfect(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Integral(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_FieldTheory_IsAlgClosed_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_FieldTheory_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_FieldTheory_Normal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_FieldTheory_Perfect(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_Integral(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__0 = _init_lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__0();
lean_mark_persistent(lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__0);
lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__1 = _init_lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__1();
lean_mark_persistent(lp_mathlib_IntermediateField_algHomEquivAlgHomOfSplits___redArg___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
