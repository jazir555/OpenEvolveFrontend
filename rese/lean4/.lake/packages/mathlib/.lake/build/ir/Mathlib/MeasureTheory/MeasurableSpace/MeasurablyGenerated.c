// Lean compiler output
// Module: Mathlib.MeasureTheory.MeasurableSpace.MeasurablyGenerated
// Imports: public import Init public import Mathlib.MeasureTheory.MeasurableSpace.Constructions public import Mathlib.Order.Filter.AtTopBot.CompleteLattice public import Mathlib.Order.Filter.AtTopBot.CountablyGenerated public import Mathlib.Order.Filter.SmallSets public import Mathlib.Order.LiminfLimsup public import Mathlib.Tactic.FinCases
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
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instInsert(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instInter(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instHasCompl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instMembership(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instEmptyCollection(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instSingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instUnion(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instSup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instSDiff(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instMembership(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instEmptyCollection(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instInsert(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instSingleton(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instHasCompl(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instUnion(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instSup(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instInter(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instInf(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instSDiff(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instBot(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSet_Subtype_instTop(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Constructions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_AtTopBot_CompleteLattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_AtTopBot_CountablyGenerated(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_SmallSets(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_LiminfLimsup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_MeasurablyGenerated(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Constructions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_AtTopBot_CompleteLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_AtTopBot_CountablyGenerated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_SmallSets(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_LiminfLimsup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
