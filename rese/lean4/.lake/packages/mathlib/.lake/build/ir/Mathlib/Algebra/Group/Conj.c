// Lean compiler output
// Module: Mathlib.Algebra.Group.Conj
// Imports: public import Init public import Mathlib.Algebra.Group.End public import Mathlib.Algebra.Group.Semiconj.Units
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
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mkEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mkEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_slow_x2dfailing_x20instance_x20priority;
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsConj_setoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk(lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0;
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsConj_setoid___boxed(lean_object*, lean_object*);
lean_object* l_Quotient_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsConj_setoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsConj_setoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_IsConj_setoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ConjClasses_mk(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ConjClasses_mk___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ConjClasses_instInhabited___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ConjClasses_instInhabited(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ConjClasses_instInhabited___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ConjClasses_instOne___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ConjClasses_instOne(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ConjClasses_instOne___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_apply_1(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ConjClasses_map(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_slow_x2dfailing_x20instance_x20priority() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_apply_2(x_3, x_4, x_5);
x_7 = lean_unbox(x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0;
x_2 = lean_box(0);
x_3 = lean_alloc_closure((void*)(l_Quotient_lift), 6, 5);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, lean_box(0));
lean_closure_set(x_3, 2, x_2);
lean_closure_set(x_3, 3, x_1);
lean_closure_set(x_3, 4, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mkEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ConjClasses_mk___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
x_3 = lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1;
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mkEquiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ConjClasses_mkEquiv___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Conj(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_slow_x2dfailing_x20instance_x20priority = _init_lp_mathlib_LibraryNote_slow_x2dfailing_x20instance_x20priority();
lean_mark_persistent(lp_mathlib_LibraryNote_slow_x2dfailing_x20instance_x20priority);
lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0 = _init_lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0();
lean_mark_persistent(lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0);
lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1 = _init_lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1();
lean_mark_persistent(lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
