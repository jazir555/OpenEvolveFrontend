// Lean compiler output
// Module: Mathlib.Algebra.Group.Units.Basic
// Imports: public import Init public import Mathlib.Algebra.Group.Basic public import Mathlib.Algebra.Group.Commute.Defs public import Mathlib.Algebra.Group.Units.Defs public import Mathlib.Logic.Unique public import Mathlib.Tactic.Nontriviality public import Mathlib.Tactic.Lift public import Mathlib.Tactic.Subsingleton
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
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAddUnitsOfSubsingleton(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddUnits_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueUnitsOfSubsingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAddUnitsOfSubsingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueUnitsOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAddUnitsOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueUnitsOfSubsingleton___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueUnitsOfSubsingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAddUnitsOfSubsingleton___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Units_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueUnitsOfSubsingleton(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Units_instInhabited___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueUnitsOfSubsingleton___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Units_instInhabited___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueUnitsOfSubsingleton___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instUniqueUnitsOfSubsingleton(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueUnitsOfSubsingleton___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_instUniqueUnitsOfSubsingleton___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAddUnitsOfSubsingleton(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddUnits_instInhabited___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAddUnitsOfSubsingleton___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddUnits_instInhabited___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAddUnitsOfSubsingleton___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instUniqueAddUnitsOfSubsingleton(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAddUnitsOfSubsingleton___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_instUniqueAddUnitsOfSubsingleton___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Unique(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Nontriviality(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Subsingleton(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Nontriviality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
