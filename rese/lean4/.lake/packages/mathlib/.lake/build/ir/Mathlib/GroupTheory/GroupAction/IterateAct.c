// Lean compiler output
// Module: Mathlib.GroupTheory.GroupAction.IterateAct
// Imports: public import Init public import Mathlib.Algebra.Group.Action.Defs public import Mathlib.Data.Countable.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instMulAction___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IterateAddAct_instAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IterateAddAct_instAddCommMonoid___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_IterateMulAct_instCommMonoid___closed__0;
lean_object* l_Nat_mul___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_IterateAddAct_instAddCommMonoid___closed__0;
static lean_object* lp_mathlib_IterateMulAct_instCommMonoid___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_IterateAddAct_instAddAction___redArg(lean_object*);
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IterateAddAct_instAddAction(lean_object*, lean_object*);
static lean_object* lp_mathlib_IterateMulAct_instCommMonoid___closed__2;
lean_object* lp_mathlib_Nat_iterate___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instMulAction(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_IterateMulAct_instCommMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_add___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_IterateMulAct_instCommMonoid___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_IterateMulAct_instCommMonoid___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_IterateMulAct_instCommMonoid___closed__1;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_mathlib_IterateMulAct_instCommMonoid___closed__0;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_IterateMulAct_instCommMonoid___closed__2;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_IterateMulAct_instCommMonoid(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_IterateAddAct_instAddCommMonoid___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_IterateMulAct_instCommMonoid___closed__1;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_mathlib_IterateMulAct_instCommMonoid___closed__0;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IterateAddAct_instAddCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_IterateAddAct_instAddCommMonoid___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IterateAddAct_instAddCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_IterateAddAct_instAddCommMonoid(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instMulAction___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Nat_iterate___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instMulAction___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_IterateMulAct_instMulAction___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IterateMulAct_instMulAction(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_IterateMulAct_instMulAction___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IterateAddAct_instAddAction___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_IterateMulAct_instMulAction___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IterateAddAct_instAddAction(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_IterateAddAct_instAddAction___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Countable_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_IterateAct(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Countable_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_IterateMulAct_instCommMonoid___closed__0 = _init_lp_mathlib_IterateMulAct_instCommMonoid___closed__0();
lean_mark_persistent(lp_mathlib_IterateMulAct_instCommMonoid___closed__0);
lp_mathlib_IterateMulAct_instCommMonoid___closed__1 = _init_lp_mathlib_IterateMulAct_instCommMonoid___closed__1();
lean_mark_persistent(lp_mathlib_IterateMulAct_instCommMonoid___closed__1);
lp_mathlib_IterateMulAct_instCommMonoid___closed__2 = _init_lp_mathlib_IterateMulAct_instCommMonoid___closed__2();
lean_mark_persistent(lp_mathlib_IterateMulAct_instCommMonoid___closed__2);
lp_mathlib_IterateAddAct_instAddCommMonoid___closed__0 = _init_lp_mathlib_IterateAddAct_instAddCommMonoid___closed__0();
lean_mark_persistent(lp_mathlib_IterateAddAct_instAddCommMonoid___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
