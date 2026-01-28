// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Sheaf
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.Presheaf public import Mathlib.Algebra.Category.ModuleCat.Limits public import Mathlib.CategoryTheory.Sites.LocallyBijective public import Mathlib.CategoryTheory.Sites.Whiskering
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
lean_object* lp_mathlib_CategoryTheory_Functor_FullyFaithful_homEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_evaluation___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instAddCommGroupHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsFunctor___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_fullyFaithfulForget(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_addCommGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_evaluation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instPreadditive(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_evaluation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instAddCommGroupHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_evaluation___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instAddCommGroupHom___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_SheafOfModules_fullyFaithfulForget___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PresheafOfModules_instAddCommGroupHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_fullyFaithfulForget___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsFunctor___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsFunctor(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instPreadditive___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PresheafOfModules_sectionsMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instAddCommGroupHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PresheafOfModules_evaluation___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc(x_6);
x_8 = lean_apply_1(x_4, x_6);
x_9 = lean_apply_1(x_5, x_6);
x_10 = lp_mathlib_LinearMap_comp___redArg(x_9, x_8);
x_11 = lean_apply_1(x_10, x_7);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_SheafOfModules_instCategory___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_SheafOfModules_instCategory___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_SheafOfModules_instCategory___lam__0___boxed), 3, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_SheafOfModules_instCategory___lam__1___boxed), 7, 0);
x_7 = lean_box(0);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_5);
lean_ctor_set(x_8, 2, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instCategory___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SheafOfModules_instCategory(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___lam__0(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_apply_2(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SheafOfModules_forget___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_SheafOfModules_forget___lam__1(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_SheafOfModules_forget___lam__0___boxed), 1, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_SheafOfModules_forget___lam__1___boxed), 5, 0);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_forget___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SheafOfModules_forget(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_5;
}
}
static lean_object* _init_lp_mathlib_SheafOfModules_fullyFaithfulForget___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_SheafOfModules_forget___lam__1___boxed), 5, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_fullyFaithfulForget(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SheafOfModules_fullyFaithfulForget___closed__0;
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_fullyFaithfulForget___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SheafOfModules_fullyFaithfulForget(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_evaluation___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_mathlib_SheafOfModules_forget(lean_box(0), x_1, x_2, x_3);
x_6 = lp_mathlib_PresheafOfModules_evaluation___redArg(x_4);
x_7 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_evaluation(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_SheafOfModules_evaluation___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_evaluation___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_SheafOfModules_evaluation(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_evaluation___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SheafOfModules_evaluation___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instAddCommGroupHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lp_mathlib_SheafOfModules_forget(lean_box(0), x_1, x_2, x_3);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lp_mathlib_SheafOfModules_fullyFaithfulForget___closed__0;
lean_inc_ref(x_5);
x_9 = lp_mathlib_CategoryTheory_Functor_FullyFaithful_homEquiv___redArg(x_6, x_8, x_4, x_5);
x_10 = lean_apply_1(x_7, x_5);
x_11 = lp_mathlib_PresheafOfModules_instAddCommGroupHom___redArg(x_10);
x_12 = lp_mathlib_Equiv_addCommGroup___redArg(x_9, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instAddCommGroupHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_SheafOfModules_instAddCommGroupHom___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instAddCommGroupHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_SheafOfModules_instAddCommGroupHom(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instAddCommGroupHom___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_SheafOfModules_instAddCommGroupHom___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instPreadditive(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_SheafOfModules_instAddCommGroupHom___boxed), 6, 4);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_2);
lean_closure_set(x_5, 2, x_3);
lean_closure_set(x_5, 3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_instPreadditive___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_SheafOfModules_instAddCommGroupHom___boxed), 6, 4);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_2);
lean_closure_set(x_4, 3, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_PresheafOfModules_sectionsMap___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsMap___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PresheafOfModules_sectionsMap___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_SheafOfModules_sectionsMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsFunctor___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_PresheafOfModules_sectionsMap___redArg(x_3, x_4);
x_7 = lean_apply_1(x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsFunctor___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_SheafOfModules_sectionsFunctor___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsFunctor(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_SheafOfModules_sectionsFunctor___lam__0___boxed), 5, 0);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, lean_box(0));
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SheafOfModules_sectionsFunctor___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SheafOfModules_sectionsFunctor(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_LocallyBijective(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_Whiskering(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Sheaf(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_LocallyBijective(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_Whiskering(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_SheafOfModules_fullyFaithfulForget___closed__0 = _init_lp_mathlib_SheafOfModules_fullyFaithfulForget___closed__0();
lean_mark_persistent(lp_mathlib_SheafOfModules_fullyFaithfulForget___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
