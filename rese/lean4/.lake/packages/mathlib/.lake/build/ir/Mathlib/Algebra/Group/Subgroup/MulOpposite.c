// Lean compiler output
// Module: Mathlib.Algebra.Group.Subgroup.MulOpposite
// Imports: public import Init public import Mathlib.Algebra.Group.Subgroup.Defs public import Mathlib.Algebra.Group.Submonoid.MulOpposite
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
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_op___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddSubgroup_equivOp___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_unop(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_opEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivOp(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_equivOp___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_opEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_opEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_unop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_op(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivOp___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_op(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_opEquiv(lean_object*, lean_object*);
static lean_object* lp_mathlib_Subgroup_equivOp___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_unop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_equivOp(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Subgroup_equivOp___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_opEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_op___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_unop(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddSubgroup_equivOp___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_op(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_op___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subgroup_op(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_op(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_op___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddSubgroup_op(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_unop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_unop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subgroup_unop(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_unop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_unop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddSubgroup_unop(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_opEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_op___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_unop___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_opEquiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subgroup_opEquiv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_opEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_op___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_unop___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_opEquiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddSubgroup_opEquiv___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Subgroup_equivOp___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Subgroup_equivOp___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Subgroup_equivOp___closed__0;
x_2 = lp_mathlib_Equiv_subtypeEquiv___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivOp(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subgroup_equivOp___closed__1;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivOp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subgroup_equivOp(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_AddSubgroup_equivOp___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddOpposite_opEquiv(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddSubgroup_equivOp___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_AddSubgroup_equivOp___closed__0;
x_2 = lp_mathlib_Equiv_subtypeEquiv___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_equivOp(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddSubgroup_equivOp___closed__1;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_equivOp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddSubgroup_equivOp(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOpposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Subgroup_equivOp___closed__0 = _init_lp_mathlib_Subgroup_equivOp___closed__0();
lean_mark_persistent(lp_mathlib_Subgroup_equivOp___closed__0);
lp_mathlib_Subgroup_equivOp___closed__1 = _init_lp_mathlib_Subgroup_equivOp___closed__1();
lean_mark_persistent(lp_mathlib_Subgroup_equivOp___closed__1);
lp_mathlib_AddSubgroup_equivOp___closed__0 = _init_lp_mathlib_AddSubgroup_equivOp___closed__0();
lean_mark_persistent(lp_mathlib_AddSubgroup_equivOp___closed__0);
lp_mathlib_AddSubgroup_equivOp___closed__1 = _init_lp_mathlib_AddSubgroup_equivOp___closed__1();
lean_mark_persistent(lp_mathlib_AddSubgroup_equivOp___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
