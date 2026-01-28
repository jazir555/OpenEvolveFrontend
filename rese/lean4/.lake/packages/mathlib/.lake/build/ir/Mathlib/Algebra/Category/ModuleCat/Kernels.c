// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Kernels
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.EpiMono public import Mathlib.CategoryTheory.ConcreteCategory.Elementwise
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
lean_object* lp_mathlib_SMulMemClass_subtype___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cofork_of_u03c0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Fork_00_u03b9___redArg(lean_object*);
lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelIsLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ModuleCat_kernelCone___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelCocone___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Fork_of_u03b9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelIsLimit___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_Quotient_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelIsColimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_moduleCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelIsLimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelIsColimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelCocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_liftQ___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelIsColimit___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_codRestrict___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cofork_00_u03c0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ModuleCat_kernelCone___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_SMulMemClass_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ModuleCat_kernelCone___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_5 = lp_mathlib_ModuleCat_moduleCategory(lean_box(0), x_1);
x_6 = lean_ctor_get(x_3, 0);
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 1);
x_9 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
x_10 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_7);
x_11 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_9);
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_13 = lean_ctor_get(x_11, 0);
x_14 = lean_ctor_get(x_11, 1);
lean_dec(x_14);
lean_inc(x_8);
x_15 = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_15, 0, x_8);
lean_ctor_set(x_11, 1, x_15);
lean_ctor_set(x_11, 0, x_10);
x_16 = lp_mathlib_ModuleCat_kernelCone___redArg___closed__0;
x_17 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_kernelCone___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_17, 0, x_13);
x_18 = lp_mathlib_CategoryTheory_Limits_Fork_of_u03b9___redArg(x_5, x_2, x_3, x_4, x_17, x_11, x_16);
return x_18;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_19 = lean_ctor_get(x_11, 0);
lean_inc(x_19);
lean_dec(x_11);
lean_inc(x_8);
x_20 = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_20, 0, x_8);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_10);
lean_ctor_set(x_21, 1, x_20);
x_22 = lp_mathlib_ModuleCat_kernelCone___redArg___closed__0;
x_23 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_kernelCone___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_23, 0, x_19);
x_24 = lp_mathlib_CategoryTheory_Limits_Fork_of_u03b9___redArg(x_5, x_2, x_3, x_4, x_23, x_21, x_22);
return x_24;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_kernelCone___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_kernelCone(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelCone___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_kernelCone___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelIsLimit___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_CategoryTheory_Limits_Fork_00_u03b9___redArg(x_1);
x_4 = lp_mathlib_LinearMap_codRestrict___redArg(x_3);
x_5 = lean_apply_1(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelIsLimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_kernelIsLimit___lam__0), 2, 0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_kernelIsLimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_kernelIsLimit(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelCocone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_5 = lp_mathlib_ModuleCat_moduleCategory(lean_box(0), x_1);
x_6 = lean_ctor_get(x_3, 0);
x_7 = lean_ctor_get(x_3, 1);
x_8 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_6);
x_9 = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(x_6);
x_10 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_8);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
lean_dec(x_13);
x_14 = lean_box(0);
lean_inc(x_7);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(x_15, 0, x_7);
lean_ctor_set(x_10, 1, x_15);
lean_ctor_set(x_10, 0, x_9);
lean_inc(x_7);
lean_inc_ref(x_6);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, lean_box(0));
lean_closure_set(x_16, 2, x_1);
lean_closure_set(x_16, 3, x_6);
lean_closure_set(x_16, 4, x_7);
lean_closure_set(x_16, 5, x_14);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_kernelCone___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_17, 0, x_12);
x_18 = lp_mathlib_CategoryTheory_Limits_Cofork_of_u03c0___redArg(x_5, x_2, x_3, x_4, x_17, x_10, x_16);
return x_18;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_19 = lean_ctor_get(x_10, 0);
lean_inc(x_19);
lean_dec(x_10);
x_20 = lean_box(0);
lean_inc(x_7);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(x_21, 0, x_7);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_9);
lean_ctor_set(x_22, 1, x_21);
lean_inc(x_7);
lean_inc_ref(x_6);
x_23 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(x_23, 0, lean_box(0));
lean_closure_set(x_23, 1, lean_box(0));
lean_closure_set(x_23, 2, x_1);
lean_closure_set(x_23, 3, x_6);
lean_closure_set(x_23, 4, x_7);
lean_closure_set(x_23, 5, x_20);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_kernelCone___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_24, 0, x_19);
x_25 = lp_mathlib_CategoryTheory_Limits_Cofork_of_u03c0___redArg(x_5, x_2, x_3, x_4, x_24, x_22, x_23);
return x_25;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelCocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_cokernelCocone___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelIsColimit___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_CategoryTheory_Limits_Cofork_00_u03c0___redArg(x_1);
x_4 = lp_mathlib_Submodule_liftQ___redArg(x_3);
x_5 = lean_apply_1(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelIsColimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_cokernelIsColimit___lam__0), 2, 0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_cokernelIsColimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_cokernelIsColimit(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_EpiMono(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_ConcreteCategory_Elementwise(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Kernels(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_EpiMono(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_ConcreteCategory_Elementwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ModuleCat_kernelCone___redArg___closed__0 = _init_lp_mathlib_ModuleCat_kernelCone___redArg___closed__0();
lean_mark_persistent(lp_mathlib_ModuleCat_kernelCone___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
