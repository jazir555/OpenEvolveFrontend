// Lean compiler output
// Module: Mathlib.Algebra.Homology.HomotopyCategory.ShiftSequence
// Imports: public import Init public import Mathlib.CategoryTheory.Shift.InducedShiftSequence public import Mathlib.CategoryTheory.Shift.Localization public import Mathlib.Algebra.Homology.HomotopyCategory.Shift public import Mathlib.Algebra.Homology.ShortComplex.HomologicalComplex public import Mathlib.Algebra.Homology.QuasiIso
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
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_shiftShortComplexFunctor_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_ShortComplex_isoMk___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_shiftFunctor___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_shiftShortComplexFunctor_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_shiftShortComplexFunctor_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CochainComplex_shiftEval___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(lean_object*);
lean_object* lp_mathlib_CochainComplex_instHasShiftInt___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_app___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_HomologicalComplex_eval___redArg(lean_object*);
lean_object* lp_mathlib_Int_negOnePow(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_shiftShortComplexFunctor_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; 
lean_inc(x_1);
x_12 = lp_mathlib_HomologicalComplex_eval___redArg(x_1);
lean_inc_ref(x_2);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_2, x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc(x_3);
x_15 = lp_mathlib_HomologicalComplex_eval___redArg(x_3);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc_ref(x_11);
x_17 = lean_apply_1(x_14, x_11);
lean_inc_ref(x_11);
x_18 = lean_apply_1(x_16, x_11);
lean_inc_ref(x_4);
lean_inc(x_18);
lean_inc(x_17);
x_19 = lean_apply_2(x_4, x_17, x_18);
x_20 = lean_ctor_get(x_19, 3);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lp_mathlib_Int_negOnePow(x_5);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
x_23 = lean_ctor_get(x_21, 1);
lean_inc(x_23);
lean_dec_ref(x_21);
lean_inc(x_5);
lean_inc_ref(x_6);
x_24 = lp_mathlib_CochainComplex_shiftEval___redArg(x_6, x_5, x_1, x_3);
lean_inc_ref(x_11);
x_25 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_24, x_11);
x_26 = !lean_is_exclusive(x_25);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; uint8_t x_42; 
x_27 = lean_ctor_get(x_25, 0);
x_28 = lean_ctor_get(x_25, 1);
lean_inc_ref(x_4);
x_29 = lean_apply_2(x_4, x_18, x_17);
x_30 = lean_ctor_get(x_29, 3);
lean_inc(x_30);
lean_dec_ref(x_29);
lean_inc(x_7);
x_31 = lp_mathlib_HomologicalComplex_eval___redArg(x_7);
x_32 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_2, x_31);
x_33 = lean_ctor_get(x_32, 0);
lean_inc(x_33);
lean_dec_ref(x_32);
lean_inc(x_8);
x_34 = lp_mathlib_HomologicalComplex_eval___redArg(x_8);
x_35 = lean_ctor_get(x_34, 0);
lean_inc(x_35);
lean_dec_ref(x_34);
lean_inc_ref(x_11);
x_36 = lean_apply_1(x_33, x_11);
lean_inc_ref(x_11);
x_37 = lean_apply_1(x_35, x_11);
lean_inc_ref(x_4);
lean_inc(x_37);
lean_inc(x_36);
x_38 = lean_apply_2(x_4, x_36, x_37);
x_39 = lean_ctor_get(x_38, 3);
lean_inc(x_39);
lean_dec_ref(x_38);
lean_inc(x_5);
lean_inc_ref(x_6);
x_40 = lp_mathlib_CochainComplex_shiftEval___redArg(x_6, x_5, x_7, x_8);
lean_inc_ref(x_11);
x_41 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_40, x_11);
x_42 = !lean_is_exclusive(x_41);
if (x_42 == 0)
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_43 = lean_ctor_get(x_41, 0);
x_44 = lean_ctor_get(x_41, 1);
x_45 = lean_apply_2(x_4, x_37, x_36);
x_46 = lean_ctor_get(x_45, 3);
lean_inc(x_46);
lean_dec_ref(x_45);
lean_inc(x_22);
x_47 = lean_apply_2(x_20, x_22, x_27);
lean_inc(x_23);
x_48 = lean_apply_2(x_30, x_23, x_28);
lean_ctor_set(x_41, 1, x_48);
lean_ctor_set(x_41, 0, x_47);
x_49 = lp_mathlib_CochainComplex_shiftEval___redArg(x_6, x_5, x_9, x_10);
x_50 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_49, x_11);
x_51 = lean_apply_2(x_39, x_22, x_43);
x_52 = lean_apply_2(x_46, x_23, x_44);
lean_ctor_set(x_25, 1, x_52);
lean_ctor_set(x_25, 0, x_51);
x_53 = lp_mathlib_CategoryTheory_ShortComplex_isoMk___redArg(x_41, x_50, x_25);
lean_dec_ref(x_50);
lean_dec_ref(x_41);
return x_53;
}
else
{
lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; 
x_54 = lean_ctor_get(x_41, 0);
x_55 = lean_ctor_get(x_41, 1);
lean_inc(x_55);
lean_inc(x_54);
lean_dec(x_41);
x_56 = lean_apply_2(x_4, x_37, x_36);
x_57 = lean_ctor_get(x_56, 3);
lean_inc(x_57);
lean_dec_ref(x_56);
lean_inc(x_22);
x_58 = lean_apply_2(x_20, x_22, x_27);
lean_inc(x_23);
x_59 = lean_apply_2(x_30, x_23, x_28);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_58);
lean_ctor_set(x_60, 1, x_59);
x_61 = lp_mathlib_CochainComplex_shiftEval___redArg(x_6, x_5, x_9, x_10);
x_62 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_61, x_11);
x_63 = lean_apply_2(x_39, x_22, x_54);
x_64 = lean_apply_2(x_57, x_23, x_55);
lean_ctor_set(x_25, 1, x_64);
lean_ctor_set(x_25, 0, x_63);
x_65 = lp_mathlib_CategoryTheory_ShortComplex_isoMk___redArg(x_60, x_62, x_25);
lean_dec_ref(x_62);
lean_dec_ref(x_60);
return x_65;
}
}
else
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; 
x_66 = lean_ctor_get(x_25, 0);
x_67 = lean_ctor_get(x_25, 1);
lean_inc(x_67);
lean_inc(x_66);
lean_dec(x_25);
lean_inc_ref(x_4);
x_68 = lean_apply_2(x_4, x_18, x_17);
x_69 = lean_ctor_get(x_68, 3);
lean_inc(x_69);
lean_dec_ref(x_68);
lean_inc(x_7);
x_70 = lp_mathlib_HomologicalComplex_eval___redArg(x_7);
x_71 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_2, x_70);
x_72 = lean_ctor_get(x_71, 0);
lean_inc(x_72);
lean_dec_ref(x_71);
lean_inc(x_8);
x_73 = lp_mathlib_HomologicalComplex_eval___redArg(x_8);
x_74 = lean_ctor_get(x_73, 0);
lean_inc(x_74);
lean_dec_ref(x_73);
lean_inc_ref(x_11);
x_75 = lean_apply_1(x_72, x_11);
lean_inc_ref(x_11);
x_76 = lean_apply_1(x_74, x_11);
lean_inc_ref(x_4);
lean_inc(x_76);
lean_inc(x_75);
x_77 = lean_apply_2(x_4, x_75, x_76);
x_78 = lean_ctor_get(x_77, 3);
lean_inc(x_78);
lean_dec_ref(x_77);
lean_inc(x_5);
lean_inc_ref(x_6);
x_79 = lp_mathlib_CochainComplex_shiftEval___redArg(x_6, x_5, x_7, x_8);
lean_inc_ref(x_11);
x_80 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_79, x_11);
x_81 = lean_ctor_get(x_80, 0);
lean_inc(x_81);
x_82 = lean_ctor_get(x_80, 1);
lean_inc(x_82);
if (lean_is_exclusive(x_80)) {
 lean_ctor_release(x_80, 0);
 lean_ctor_release(x_80, 1);
 x_83 = x_80;
} else {
 lean_dec_ref(x_80);
 x_83 = lean_box(0);
}
x_84 = lean_apply_2(x_4, x_76, x_75);
x_85 = lean_ctor_get(x_84, 3);
lean_inc(x_85);
lean_dec_ref(x_84);
lean_inc(x_22);
x_86 = lean_apply_2(x_20, x_22, x_66);
lean_inc(x_23);
x_87 = lean_apply_2(x_69, x_23, x_67);
if (lean_is_scalar(x_83)) {
 x_88 = lean_alloc_ctor(0, 2, 0);
} else {
 x_88 = x_83;
}
lean_ctor_set(x_88, 0, x_86);
lean_ctor_set(x_88, 1, x_87);
x_89 = lp_mathlib_CochainComplex_shiftEval___redArg(x_6, x_5, x_9, x_10);
x_90 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_89, x_11);
x_91 = lean_apply_2(x_78, x_22, x_81);
x_92 = lean_apply_2(x_85, x_23, x_82);
x_93 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_93, 0, x_91);
lean_ctor_set(x_93, 1, x_92);
x_94 = lp_mathlib_CategoryTheory_ShortComplex_isoMk___redArg(x_88, x_90, x_93);
lean_dec_ref(x_90);
lean_dec_ref(x_88);
return x_94;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_shiftShortComplexFunctor_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_10 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_1, x_2);
lean_inc(x_3);
x_11 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_10, x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_CochainComplex_shiftShortComplexFunctor_x27___redArg___lam__0), 11, 10);
lean_closure_set(x_12, 0, x_4);
lean_closure_set(x_12, 1, x_11);
lean_closure_set(x_12, 2, x_7);
lean_closure_set(x_12, 3, x_2);
lean_closure_set(x_12, 4, x_3);
lean_closure_set(x_12, 5, x_1);
lean_closure_set(x_12, 6, x_6);
lean_closure_set(x_12, 7, x_9);
lean_closure_set(x_12, 8, x_5);
lean_closure_set(x_12, 9, x_8);
x_13 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_shiftShortComplexFunctor_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CochainComplex_shiftShortComplexFunctor_x27___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_14;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Shift_InducedShiftSequence(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Shift_Localization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Shift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_HomologicalComplex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_QuasiIso(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_ShiftSequence(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Shift_InducedShiftSequence(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Shift_Localization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Shift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_HomologicalComplex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_QuasiIso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
