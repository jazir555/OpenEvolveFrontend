// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Limits
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.Basic public import Mathlib.Algebra.Category.Grp.Limits public import Mathlib.Algebra.Colimit.Module public import Mathlib.Algebra.Module.Shrink
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
lean_object* lp_mathlib_Pi_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_addCommGroupObj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_sectionsSubmodule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_sectionsSubmodule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_addCommGroupObj___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_addCommGroupObj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections___redArg(lean_object*);
lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_moduleObj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_moduleObj___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_moduleObj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_addCommGroupObj___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_addCommGroupObj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ModuleCat_addCommGroupObj___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_addCommGroupObj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ModuleCat_addCommGroupObj(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_moduleObj___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_moduleObj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ModuleCat_moduleObj___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_moduleObj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ModuleCat_moduleObj(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_sectionsSubmodule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_sectionsSubmodule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_sectionsSubmodule(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addCommMonoid___redArg(x_2);
x_4 = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_instAddCommMonoidElemForallObjCompForgetSections(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_5, x_2);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_2(x_7, x_3, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_module___redArg(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_instModuleElemForallObjCompForgetSections(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_apply_4(x_1, x_2, x_3, lean_box(0), x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_directLimitDiagram___redArg___lam__0), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_directLimitDiagram___redArg___lam__1), 5, 1);
lean_closure_set(x_5, 0, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ModuleCat_directLimitDiagram___redArg(x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_directLimitDiagram___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ModuleCat_directLimitDiagram(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Colimit_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Shrink(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Limits(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Grp_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Colimit_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Shrink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
