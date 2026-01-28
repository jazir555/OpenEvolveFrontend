// Lean compiler output
// Module: Mathlib.CategoryTheory.Extensive
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Shapes.Pullback.CommSq public import Mathlib.CategoryTheory.Limits.Shapes.StrictInitial public import Mathlib.CategoryTheory.Limits.Types.Coproducts public import Mathlib.CategoryTheory.Limits.Types.Products public import Mathlib.CategoryTheory.Limits.Types.Pullbacks public import Mathlib.Topology.Category.TopCat.Limits.Pullbacks public import Mathlib.CategoryTheory.Limits.FunctorCategory.Basic public import Mathlib.CategoryTheory.Limits.Constructions.FiniteProductsOfBinaryProducts public import Mathlib.CategoryTheory.Limits.VanKampen
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_CommSq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_StrictInitial(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Coproducts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Products(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Pullbacks(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Category_TopCat_Limits_Pullbacks(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_FunctorCategory_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_FiniteProductsOfBinaryProducts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_VanKampen(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Extensive(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_CommSq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_StrictInitial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Coproducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Products(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Pullbacks(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Category_TopCat_Limits_Pullbacks(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_FunctorCategory_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_FiniteProductsOfBinaryProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_VanKampen(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
