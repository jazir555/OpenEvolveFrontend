// Lean compiler output
// Module: Mathlib.GroupTheory.Subgroup.Centralizer
// Imports: public import Init public import Mathlib.Algebra.Group.Action.End public import Mathlib.GroupTheory.Subgroup.Center public import Mathlib.GroupTheory.Submonoid.Centralizer
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
lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centralizer___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centralizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizerMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closureCommGroupOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closureAddCommGroupOfComm___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizerMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_centralizer(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closureAddCommGroupOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closureCommGroupOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centralizer(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centralizer___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subgroup_centralizer(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_centralizer(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_centralizer___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddSubgroup_centralizer(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closureCommGroupOfComm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SubgroupClass_toGroup___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closureCommGroupOfComm___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SubgroupClass_toGroup___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closureAddCommGroupOfComm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closureAddCommGroupOfComm___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
lean_inc(x_2);
lean_inc(x_3);
x_7 = lean_apply_2(x_2, x_3, x_4);
x_8 = lean_apply_1(x_6, x_3);
x_9 = lean_apply_2(x_2, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lp_mathlib_Monoid_toMulOneClass___redArg(x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizerMonoidHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_SubgroupClass_toGroup___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizer___redArg(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMulEquiv___boxed), 6, 5);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_2);
lean_closure_set(x_5, 3, x_3);
lean_closure_set(x_5, 4, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizerMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subgroup_normalizerMonoidHom___redArg(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subgroup_Center(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subgroup_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
