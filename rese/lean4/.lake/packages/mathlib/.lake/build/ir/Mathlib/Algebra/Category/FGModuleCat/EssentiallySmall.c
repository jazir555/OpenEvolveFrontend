// Lean compiler output
// Module: Mathlib.Algebra.Category.FGModuleCat.EssentiallySmall
// Imports: public import Init public import Mathlib.Algebra.Category.FGModuleCat.Basic public import Mathlib.RingTheory.Finiteness.Cardinality
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
lean_object* lp_mathlib_CategoryTheory_InducedCategory_instCategory___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instSmallCategory___redArg(lean_object*);
lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_Function_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCategory___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_FGModuleCat_ulift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCategory___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_embed___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCategory___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instModuleRepr___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_moduleCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCoeSortType___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCoeSortType(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instSmallCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instModuleRepr(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_inducedFunctor___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instModuleRepr___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_embed(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCoeSortType(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCoeSortType___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_FGModuleRepr_instCoeSortType(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_Pi_addCommGroup___redArg(x_3);
x_5 = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instAddCommGroupRepr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FGModuleRepr_instAddCommGroupRepr(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instModuleRepr___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lp_mathlib_Semiring_toModule___redArg(x_2);
x_4 = lp_mathlib_Pi_Function_module___redArg(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instModuleRepr(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FGModuleRepr_instModuleRepr___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instModuleRepr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FGModuleRepr_instModuleRepr(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCategory___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_FGModuleRepr_instAddCommGroupRepr___redArg(x_1);
x_4 = lp_mathlib_FGModuleRepr_instModuleRepr___redArg(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCategory___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_FGModuleRepr_instCategory___redArg___lam__0(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCategory___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_FGModuleRepr_instCategory___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_ModuleCat_moduleCategory(lean_box(0), x_1);
lean_dec_ref(x_1);
x_4 = lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(x_3);
x_5 = lp_mathlib_CategoryTheory_InducedCategory_instCategory___redArg(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instCategory(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_FGModuleRepr_instCategory___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instSmallCategory(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_FGModuleRepr_instCategory___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_instSmallCategory___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_FGModuleRepr_instCategory___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_embed___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_FGModuleRepr_instCategory___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_CategoryTheory_inducedFunctor___redArg(x_2);
x_4 = lp_mathlib_FGModuleCat_ulift___redArg(x_1);
x_5 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleRepr_embed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_FGModuleRepr_embed___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_FGModuleCat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Cardinality(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_FGModuleCat_EssentiallySmall(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_FGModuleCat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Finiteness_Cardinality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
