// Lean compiler output
// Module: Mathlib.Topology.Constructions
// Imports: public import Init public import Mathlib.Algebra.Group.TypeTags.Basic public import Mathlib.Data.Fin.VecNotation public import Mathlib.Data.Finset.Piecewise public import Mathlib.Data.SetLike.Basic public import Mathlib.Order.Filter.Cofinite public import Mathlib.Order.Filter.Curry public import Mathlib.Topology.Constructions.SumProd public import Mathlib.Topology.NhdsSet
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
LEAN_EXPORT lean_object* lp_mathlib_ULift_topologicalSpace(lean_object*, lean_object*);
static lean_object* lp_mathlib_instTopologicalSpaceSigma___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Pi_topologicalSpace(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_piCurry(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CofiniteTopology_of___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceAdditive(lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_piCurry___closed__1;
lean_object* lp_mathlib_TopologicalSpace_instCompleteLattice(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceAdditive___redArg(lean_object*);
static lean_object* lp_mathlib_Pi_topologicalSpace___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CofiniteTopology_instInhabited___redArg(lean_object*);
static lean_object* lp_mathlib_Homeomorph_piCurry___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceSigma(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CofiniteTopology_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceMultiplicative(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceSigma___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_curry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopologicalSpace(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CofiniteTopology_of(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceQuot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_topologicalSpace___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_ofEqSubtypes___closed__0;
static lean_object* lp_mathlib_instTopologicalSpaceSigma___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_ofEqSubtypes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopologicalSpace___redArg(lean_object*);
static lean_object* lp_mathlib_Homeomorph_piCurry___closed__0;
lean_object* l_Function_uncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Pi_topologicalSpace___closed__1;
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CofiniteTopology_instTopologicalSpace(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceQuot(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceQuotient(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
static lean_object* _init_lp_mathlib_instTopologicalSpaceSigma___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_TopologicalSpace_instCompleteLattice(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_instTopologicalSpaceSigma___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_instTopologicalSpaceSigma___closed__0;
x_2 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceSigma(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_instTopologicalSpaceSigma___closed__1;
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceSigma___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instTopologicalSpaceSigma(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Pi_topologicalSpace___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_TopologicalSpace_instCompleteLattice(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Pi_topologicalSpace___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Pi_topologicalSpace___closed__0;
x_2 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_topologicalSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_Pi_topologicalSpace___closed__1;
x_5 = lean_ctor_get(x_4, 2);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_topologicalSpace___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_topologicalSpace(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_topologicalSpace(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceAdditive(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceAdditive___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceMultiplicative(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceMultiplicative___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopologicalSpace(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopologicalSpace___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
static lean_object* _init_lp_mathlib_CofiniteTopology_of___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CofiniteTopology_of(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CofiniteTopology_of___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CofiniteTopology_instInhabited___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_CofiniteTopology_of___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CofiniteTopology_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CofiniteTopology_instInhabited___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CofiniteTopology_instTopologicalSpace(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_ofEqSubtypes___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_ofEqSubtypes(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Homeomorph_ofEqSubtypes___closed__0;
return x_6;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_piCurry___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Function_curry), 6, 3);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
lean_closure_set(x_1, 2, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_piCurry___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Function_uncurry), 5, 3);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
lean_closure_set(x_1, 2, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_piCurry___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Homeomorph_piCurry___closed__1;
x_2 = lp_mathlib_Homeomorph_piCurry___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_piCurry(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Homeomorph_piCurry___closed__2;
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Piecewise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Cofinite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Curry(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Constructions_SumProd(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_NhdsSet(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Constructions(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Cofinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Curry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Constructions_SumProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_NhdsSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instTopologicalSpaceSigma___closed__0 = _init_lp_mathlib_instTopologicalSpaceSigma___closed__0();
lean_mark_persistent(lp_mathlib_instTopologicalSpaceSigma___closed__0);
lp_mathlib_instTopologicalSpaceSigma___closed__1 = _init_lp_mathlib_instTopologicalSpaceSigma___closed__1();
lean_mark_persistent(lp_mathlib_instTopologicalSpaceSigma___closed__1);
lp_mathlib_Pi_topologicalSpace___closed__0 = _init_lp_mathlib_Pi_topologicalSpace___closed__0();
lean_mark_persistent(lp_mathlib_Pi_topologicalSpace___closed__0);
lp_mathlib_Pi_topologicalSpace___closed__1 = _init_lp_mathlib_Pi_topologicalSpace___closed__1();
lean_mark_persistent(lp_mathlib_Pi_topologicalSpace___closed__1);
lp_mathlib_CofiniteTopology_of___closed__0 = _init_lp_mathlib_CofiniteTopology_of___closed__0();
lean_mark_persistent(lp_mathlib_CofiniteTopology_of___closed__0);
lp_mathlib_Homeomorph_ofEqSubtypes___closed__0 = _init_lp_mathlib_Homeomorph_ofEqSubtypes___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_ofEqSubtypes___closed__0);
lp_mathlib_Homeomorph_piCurry___closed__0 = _init_lp_mathlib_Homeomorph_piCurry___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_piCurry___closed__0);
lp_mathlib_Homeomorph_piCurry___closed__1 = _init_lp_mathlib_Homeomorph_piCurry___closed__1();
lean_mark_persistent(lp_mathlib_Homeomorph_piCurry___closed__1);
lp_mathlib_Homeomorph_piCurry___closed__2 = _init_lp_mathlib_Homeomorph_piCurry___closed__2();
lean_mark_persistent(lp_mathlib_Homeomorph_piCurry___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
