// Lean compiler output
// Module: Mathlib.Algebra.BigOperators.Group.List.Lemmas
// Imports: public import Init public import Mathlib.Algebra.BigOperators.Group.List.Basic public import Mathlib.Algebra.Divisibility.Basic public import Mathlib.Algebra.Group.Int.Units public import Mathlib.Data.List.Dedup public import Mathlib.Data.List.Flatten public import Mathlib.Data.List.Pairwise public import Mathlib.Data.List.Perm.Basic public import Mathlib.Data.List.Range public import Mathlib.Data.List.Rotate public import Mathlib.Data.List.ProdSigma public import Mathlib.Algebra.Group.Opposite
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter___redArg(uint8_t x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (x_1 == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_2);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_3, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_3);
x_6 = lean_box(0);
x_7 = lean_apply_1(x_2, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter(x_1, x_5, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lean_unbox(x_1);
x_5 = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_List_Lemmas_0__List_filter_match__1_splitter___redArg(x_4, x_2, x_3);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Dedup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Flatten(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Pairwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Rotate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_ProdSigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Dedup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Flatten(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Rotate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_ProdSigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
