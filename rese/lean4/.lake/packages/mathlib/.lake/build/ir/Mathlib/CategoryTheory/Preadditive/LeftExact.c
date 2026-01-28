// Lean compiler output
// Module: Mathlib.CategoryTheory.Preadditive.LeftExact
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Constructions.LimitsOfProductsAndEqualizers public import Mathlib.CategoryTheory.Limits.Preserves.Shapes.Kernels public import Mathlib.CategoryTheory.Preadditive.AdditiveFunctor
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_isColimitMapCoconeBinaryCofanOfPreservesCokernels___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_isLimitMapConeBinaryFanOfPreservesKernels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_isBinaryBilimitOfTotal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapBinaryBicone___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_BinaryFan_mk___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_BinaryBicone_ofLimitCone___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_BinaryCofan_mk___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_BinaryBicone_ofColimitCocone___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeBinaryCofanEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_isColimitMapCoconeBinaryCofanOfPreservesCokernels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_isLimitMapConeBinaryFanOfPreservesKernels___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_isLimitMapConeBinaryFanOfPreservesKernels___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_8);
lean_inc_ref(x_5);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_12 = lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg(x_1, x_3, x_5, x_8, x_6, x_7, x_9, x_10);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_5, 0);
x_15 = lp_mathlib_CategoryTheory_Limits_BinaryFan_mk___redArg(x_8, x_9, x_10);
lean_inc(x_7);
lean_inc(x_6);
x_16 = lp_mathlib_CategoryTheory_Limits_BinaryBicone_ofLimitCone___redArg(x_1, x_2, x_6, x_7, x_15, x_11);
lean_inc(x_14);
lean_inc(x_6);
x_17 = lean_apply_1(x_14, x_6);
lean_inc(x_14);
lean_inc(x_7);
x_18 = lean_apply_1(x_14, x_7);
x_19 = lp_mathlib_CategoryTheory_Functor_mapBinaryBicone___redArg(x_5, x_6, x_7, x_16);
x_20 = lp_mathlib_CategoryTheory_Limits_isBinaryBilimitOfTotal___redArg(x_3, x_4, x_17, x_18, x_19);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
x_22 = lean_apply_1(x_13, x_21);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_isLimitMapConeBinaryFanOfPreservesKernels(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Functor_isLimitMapConeBinaryFanOfPreservesKernels___redArg(x_2, x_3, x_5, x_6, x_7, x_9, x_10, x_11, x_12, x_13, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_isColimitMapCoconeBinaryCofanOfPreservesCokernels___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_8);
lean_inc_ref(x_5);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_12 = lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeBinaryCofanEquiv___redArg(x_1, x_3, x_5, x_8, x_6, x_7, x_9, x_10);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_5, 0);
x_15 = lp_mathlib_CategoryTheory_Limits_BinaryCofan_mk___redArg(x_8, x_9, x_10);
lean_inc(x_7);
lean_inc(x_6);
x_16 = lp_mathlib_CategoryTheory_Limits_BinaryBicone_ofColimitCocone___redArg(x_1, x_2, x_6, x_7, x_15, x_11);
lean_inc(x_14);
lean_inc(x_6);
x_17 = lean_apply_1(x_14, x_6);
lean_inc(x_14);
lean_inc(x_7);
x_18 = lean_apply_1(x_14, x_7);
x_19 = lp_mathlib_CategoryTheory_Functor_mapBinaryBicone___redArg(x_5, x_6, x_7, x_16);
x_20 = lp_mathlib_CategoryTheory_Limits_isBinaryBilimitOfTotal___redArg(x_3, x_4, x_17, x_18, x_19);
x_21 = lean_ctor_get(x_20, 1);
lean_inc(x_21);
lean_dec_ref(x_20);
x_22 = lean_apply_1(x_13, x_21);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_isColimitMapCoconeBinaryCofanOfPreservesCokernels(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Functor_isColimitMapCoconeBinaryCofanOfPreservesCokernels___redArg(x_2, x_3, x_5, x_6, x_7, x_9, x_10, x_11, x_12, x_13, x_15);
return x_16;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_LimitsOfProductsAndEqualizers(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_Kernels(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_LeftExact(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_LimitsOfProductsAndEqualizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_Kernels(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
