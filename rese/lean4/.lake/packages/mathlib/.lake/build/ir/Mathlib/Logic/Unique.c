// Lean compiler output
// Module: Mathlib.Logic.Unique
// Imports: public import Init public import Mathlib.Logic.IsEmpty public import Mathlib.Tactic.Inhabit public import Mathlib.Tactic.Push.Attr
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
LEAN_EXPORT lean_object* lp_mathlib_Option_instUniqueOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_unique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instUnique;
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instUnique;
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_Fin_instUnique___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___boxed(lean_object*, lean_object*);
lean_object* l_Fin_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueTrue;
lean_object* l_Pi_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_unique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueProp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_uniqueOfSubsingleton(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_uniqueOfSubsingleton___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_PUnit_instUnique() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueProp(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
static lean_object* _init_lp_mathlib_instUniqueTrue() {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Unique_instInhabited(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unique_instInhabited___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Unique_mk_x27(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unique_mk_x27___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_unique___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = l_Pi_instInhabited___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_unique___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty___lam__0(lean_object* x_1) {
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Pi_uniqueOfIsEmpty___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Pi_uniqueOfIsEmpty___lam__0___boxed), 1, 0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_unique(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_apply_1(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_unique___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Function_Injective_unique(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Function_Injective_unique___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_uniqueElim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_uniqueElim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instUniqueOfIsEmpty(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Unique_subtypeEq(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unique_subtypeEq___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Unique_subtypeEq_x27(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unique_subtypeEq_x27___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Fin_instUnique___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = l_Fin_instInhabited___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Fin_instUnique() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Fin_instUnique___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_IsEmpty(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push_Attr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Unique(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_IsEmpty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PUnit_instUnique = _init_lp_mathlib_PUnit_instUnique();
lean_mark_persistent(lp_mathlib_PUnit_instUnique);
lp_mathlib_instUniqueTrue = _init_lp_mathlib_instUniqueTrue();
lp_mathlib_Fin_instUnique___closed__0 = _init_lp_mathlib_Fin_instUnique___closed__0();
lean_mark_persistent(lp_mathlib_Fin_instUnique___closed__0);
lp_mathlib_Fin_instUnique = _init_lp_mathlib_Fin_instUnique();
lean_mark_persistent(lp_mathlib_Fin_instUnique);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
