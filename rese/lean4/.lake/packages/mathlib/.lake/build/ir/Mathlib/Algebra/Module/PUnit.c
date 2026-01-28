// Lean compiler output
// Module: Mathlib.Algebra.Module.PUnit
// Imports: public import Init public import Mathlib.Algebra.Module.Defs public import Mathlib.Algebra.Ring.Action.Basic public import Mathlib.Algebra.Ring.PUnit
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
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMul__mathlib___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulDistribMulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instDistribMulAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulDistribMulAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smul___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smulWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMul__mathlib___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instVAdd__mathlib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulSemiringAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_distribMulAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smulWithZero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_module(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulActionWithZero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smul___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulActionWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulSemiringAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_vadd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instDistribMulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instMulAction(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMulZeroClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smul(lean_object*);
static lean_object* lp_mathlib_PUnit_vadd___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_distribMulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMul__mathlib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_module___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_PUnit_instVAdd__mathlib___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smul___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smul___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_smul___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smul(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_PUnit_smul___lam__0___boxed), 2, 0);
return x_2;
}
}
static lean_object* _init_lp_mathlib_PUnit_vadd___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PUnit_smul___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_vadd(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_PUnit_vadd___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smulWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_vadd___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_smulWithZero___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_smulWithZero(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulAction(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_vadd___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulAction___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_mulAction(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_distribMulAction(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_vadd___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_distribMulAction___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_distribMulAction(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulDistribMulAction(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_vadd___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulDistribMulAction___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_mulDistribMulAction(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulSemiringAction(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_vadd___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulSemiringAction___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_mulSemiringAction(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulActionWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_vadd___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_mulActionWithZero___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_mulActionWithZero(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_module(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_vadd___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_module___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_module(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMul__mathlib___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMul__mathlib___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_instSMul__mathlib___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMul__mathlib(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_PUnit_instSMul__mathlib___lam__0___boxed), 2, 0);
return x_2;
}
}
static lean_object* _init_lp_mathlib_PUnit_instVAdd__mathlib___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PUnit_instSMul__mathlib___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instVAdd__mathlib(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_PUnit_instVAdd__mathlib___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instMulAction(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_PUnit_instVAdd__mathlib___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMulZeroClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_instVAdd__mathlib___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instSMulZeroClass___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_instSMulZeroClass(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instDistribMulAction(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_instVAdd__mathlib___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instDistribMulAction___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PUnit_instDistribMulAction(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_PUnit(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_PUnit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PUnit_vadd___closed__0 = _init_lp_mathlib_PUnit_vadd___closed__0();
lean_mark_persistent(lp_mathlib_PUnit_vadd___closed__0);
lp_mathlib_PUnit_instVAdd__mathlib___closed__0 = _init_lp_mathlib_PUnit_instVAdd__mathlib___closed__0();
lean_mark_persistent(lp_mathlib_PUnit_instVAdd__mathlib___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
