// Lean compiler output
// Module: Mathlib.Algebra.Category.Grp.Limits
// Imports: public import Init public import Mathlib.Algebra.Category.Grp.ForgetCorepresentable public import Mathlib.Algebra.Category.Grp.Preadditive public import Mathlib.Algebra.Category.MonCat.ForgetCorepresentable public import Mathlib.Algebra.Category.MonCat.Limits public import Mathlib.Algebra.Group.Subgroup.Ker public import Mathlib.CategoryTheory.ConcreteCategory.ReflectsIso public import Mathlib.CategoryTheory.Limits.ConcreteCategory.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_addGroupObj___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_groupObj___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_Pi_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_commGroupObj(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sections_u03c0MonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_groupObj___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_commGroupObj___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sections_u03c0MonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsGroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_addCommGroupObj___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_addGroupObj___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sections_u03c0MonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddSubgroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsSubgroup(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_group___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_addCommGroupObj___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_addCommGroupObj(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsSubgroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddSubgroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_commGroupObj___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddGroup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_addGroupObj(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sections_u03c0AddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsGroup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sections_u03c0MonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddGroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sections_u03c0AddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_groupObj(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sections_u03c0AddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_groupObj___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_groupObj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_GrpCat_groupObj___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_groupObj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_GrpCat_groupObj(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_addGroupObj___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_addGroupObj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddGrpCat_addGroupObj___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_addGroupObj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddGrpCat_addGroupObj(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsSubgroup(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsSubgroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_GrpCat_sectionsSubgroup(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddSubgroup(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddSubgroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddGrpCat_sectionsAddSubgroup(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsGroup___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_GrpCat_sectionsGroup___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_group___redArg(x_2);
x_4 = lp_mathlib_SubgroupClass_toGroup___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_GrpCat_sectionsGroup___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sectionsGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_GrpCat_sectionsGroup(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddGroup___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_AddGrpCat_sectionsAddGroup___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addGroup___redArg(x_2);
x_4 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddGrpCat_sectionsAddGroup___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sectionsAddGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddGrpCat_sectionsAddGroup(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sections_u03c0MonoidHom___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sections_u03c0MonoidHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_GrpCat_sections_u03c0MonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sections_u03c0MonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_GrpCat_sections_u03c0MonoidHom___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_sections_u03c0MonoidHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_GrpCat_sections_u03c0MonoidHom(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sections_u03c0AddMonoidHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_GrpCat_sections_u03c0MonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sections_u03c0AddMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddGrpCat_sections_u03c0AddMonoidHom___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_sections_u03c0AddMonoidHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddGrpCat_sections_u03c0AddMonoidHom(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_commGroupObj___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_commGroupObj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CommGrpCat_commGroupObj___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_commGroupObj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CommGrpCat_commGroupObj(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_addCommGroupObj___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_addCommGroupObj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddCommGrpCat_addCommGroupObj___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_addCommGroupObj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddCommGrpCat_addCommGroupObj(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_ForgetCorepresentable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_Preadditive(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_MonCat_ForgetCorepresentable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_MonCat_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_ConcreteCategory_ReflectsIso(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_ConcreteCategory_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_Limits(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Grp_ForgetCorepresentable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Grp_Preadditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_MonCat_ForgetCorepresentable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_MonCat_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_ConcreteCategory_ReflectsIso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_ConcreteCategory_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
