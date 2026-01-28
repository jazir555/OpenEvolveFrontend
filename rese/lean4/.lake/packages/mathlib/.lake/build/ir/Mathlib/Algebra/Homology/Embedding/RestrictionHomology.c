// Lean compiler output
// Module: Mathlib.Algebra.Homology.Embedding.RestrictionHomology
// Imports: public import Init public import Mathlib.Algebra.Homology.Embedding.Restriction public import Mathlib.Algebra.Homology.ShortComplex.HomologicalComplex
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
lean_object* lp_mathlib_CategoryTheory_ShortComplex_isoMk___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_HomologicalComplex_restrictionXIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_restriction_sc_x27Iso___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_restriction_sc_x27Iso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_restriction_sc_x27Iso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_restriction_sc_x27Iso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_10 = lp_mathlib_HomologicalComplex_restrictionXIso___redArg(x_1, x_2, x_3, x_4, x_7);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_11 = lp_mathlib_HomologicalComplex_restrictionXIso___redArg(x_1, x_2, x_3, x_5, x_8);
x_12 = lp_mathlib_HomologicalComplex_restrictionXIso___redArg(x_1, x_2, x_3, x_6, x_9);
x_13 = lp_mathlib_CategoryTheory_ShortComplex_isoMk___redArg(x_10, x_11, x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_restriction_sc_x27Iso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21) {
_start:
{
lean_object* x_22; 
x_22 = lp_mathlib_HomologicalComplex_restriction_sc_x27Iso___redArg(x_6, x_8, x_9, x_11, x_12, x_13, x_14, x_15, x_16);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_restriction_sc_x27Iso___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
lean_object* x_21 = _args[20];
_start:
{
lean_object* x_22; 
x_22 = lp_mathlib_HomologicalComplex_restriction_sc_x27Iso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21);
lean_dec(x_7);
return x_22;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_Embedding_Restriction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_HomologicalComplex(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Homology_Embedding_RestrictionHomology(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_Embedding_Restriction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_HomologicalComplex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
