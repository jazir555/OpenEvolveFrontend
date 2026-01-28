// Lean compiler output
// Module: Mathlib.MeasureTheory.MeasurableSpace.Constructions
// Imports: public import Init public import Mathlib.Data.Finset.Update public import Mathlib.Data.Prod.TProd public import Mathlib.Data.Set.UnionLift public import Mathlib.GroupTheory.Coset.Defs public import Mathlib.MeasureTheory.MeasurableSpace.Basic public import Mathlib.MeasureTheory.MeasurableSpace.Instances public import Mathlib.Order.Disjointed
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
LEAN_EXPORT lean_object* lp_mathlib_Set_instMeasurableSpace___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_prod(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMeasurableSpace(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TProd_instMeasurableSpace___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TProd_instMeasurableSpace___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instMeasurableSpace(lean_object*);
static lean_object* lp_mathlib_Sigma_instMeasurableSpace___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMeasurableSpace(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Sigma_instMeasurableSpace___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_measurableSpace(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MeasurableSpace_pi___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_TProd_instMeasurableSpace(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quot_instMeasurableSpace(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_pi(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MeasurableSpace_instCompleteLattice(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_measurableSpace(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_pi___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instMeasurableSpace(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instMeasurableSpace___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MeasurableSpace_pi___closed__0;
lean_object* lp_mathlib_MeasurableSpace_giGenerateFrom___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Sum_instMeasurableSpace___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_measurableSpace___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instMeasurableSpace(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instMeasurableSpace___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TProd_instMeasurableSpace___redArg___boxed(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instMeasurableSpace(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_measurableSpace___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMeasurableSpace(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMeasurableSpace(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quot_instMeasurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instMeasurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_measurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_measurableSpace___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_QuotientGroup_measurableSpace(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_measurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_measurableSpace___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_QuotientAddGroup_measurableSpace(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instMeasurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_prod(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMeasurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
static lean_object* _init_lp_mathlib_MeasurableSpace_pi___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_MeasurableSpace_instCompleteLattice(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_MeasurableSpace_pi___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_MeasurableSpace_pi___closed__0;
x_2 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_pi(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_MeasurableSpace_pi___closed__1;
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasurableSpace_pi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MeasurableSpace_pi(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TProd_instMeasurableSpace___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TProd_instMeasurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_TProd_instMeasurableSpace___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TProd_instMeasurableSpace___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_TProd_instMeasurableSpace(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TProd_instMeasurableSpace___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_TProd_instMeasurableSpace___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Sum_instMeasurableSpace___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_MeasurableSpace_giGenerateFrom___lam__0(lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMeasurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Sum_instMeasurableSpace___closed__0;
return x_5;
}
}
static lean_object* _init_lp_mathlib_Sigma_instMeasurableSpace___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_MeasurableSpace_instCompleteLattice(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Sigma_instMeasurableSpace___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Sigma_instMeasurableSpace___closed__0;
x_2 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instMeasurableSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_Sigma_instMeasurableSpace___closed__1;
x_5 = lean_ctor_get(x_4, 2);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instMeasurableSpace___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Sigma_instMeasurableSpace(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instMeasurableSpace___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instMeasurableSpace___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_instMeasurableSpace___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instMeasurableSpace(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_box(0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Set_instMeasurableSpace___lam__0___boxed), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_MeasurableSpace_pi(lean_box(0), lean_box(0), x_3);
lean_dec_ref(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Update(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Prod_TProd(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_UnionLift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Instances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Disjointed(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Constructions(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Update(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Prod_TProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_UnionLift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Disjointed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MeasurableSpace_pi___closed__0 = _init_lp_mathlib_MeasurableSpace_pi___closed__0();
lean_mark_persistent(lp_mathlib_MeasurableSpace_pi___closed__0);
lp_mathlib_MeasurableSpace_pi___closed__1 = _init_lp_mathlib_MeasurableSpace_pi___closed__1();
lean_mark_persistent(lp_mathlib_MeasurableSpace_pi___closed__1);
lp_mathlib_Sum_instMeasurableSpace___closed__0 = _init_lp_mathlib_Sum_instMeasurableSpace___closed__0();
lean_mark_persistent(lp_mathlib_Sum_instMeasurableSpace___closed__0);
lp_mathlib_Sigma_instMeasurableSpace___closed__0 = _init_lp_mathlib_Sigma_instMeasurableSpace___closed__0();
lean_mark_persistent(lp_mathlib_Sigma_instMeasurableSpace___closed__0);
lp_mathlib_Sigma_instMeasurableSpace___closed__1 = _init_lp_mathlib_Sigma_instMeasurableSpace___closed__1();
lean_mark_persistent(lp_mathlib_Sigma_instMeasurableSpace___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
