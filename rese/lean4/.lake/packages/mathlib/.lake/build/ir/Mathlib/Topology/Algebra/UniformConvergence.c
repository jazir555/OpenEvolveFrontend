// Lean compiler output
// Module: Mathlib.Topology.Algebra.UniformConvergence
// Imports: public import Init public import Mathlib.Topology.Algebra.UniformMulAction public import Mathlib.Algebra.Module.Pi public import Mathlib.Topology.UniformSpace.UniformConvergenceTopology
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
lean_object* lp_mathlib_Pi_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformOnFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addGroup___redArg(lean_object*);
lean_object* lp_mathlib_Pi_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNegUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformFun___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformFun___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformOnFun___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformOnFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformFun___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommGroupUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddUniformFun___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instAdd___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformFun___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNegUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommGroupUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommGroupUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformFun___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformFun___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformFun___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformFun___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommMonoidUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDivUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNegUniformOnFun___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_group___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformFun___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulUniformFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommGroupUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommMonoidUniformFun(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSubUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommMonoidUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommMonoidUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDivUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDivUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSubUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSubUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformFun___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_commGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDivUniformFun___redArg(lean_object*);
lean_object* lp_mathlib_Pi_distribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instDiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformFun___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_commMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNegUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformFun___redArg(lean_object*);
lean_object* lp_mathlib_Pi_mulAction___redArg(lean_object*);
lean_object* lp_mathlib_Pi_monoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformFun___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSubUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformFun___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformOnFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupUniformOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformFun___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformFun(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformFun___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformFun___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instOneUniformFun___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instOneUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instOne___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instOneUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instOneUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instZeroUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instOneUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instOne___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instOneUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instOneUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instZeroUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformFun___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_2(x_1, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformFun___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instMulUniformFun___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instMul___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instMulUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instAdd___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instAddUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instMul___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instMulUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instAdd___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instAddUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformFun___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformFun___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instInvUniformFun___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instInvUniformFun___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instInv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instInvUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNegUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instInvUniformFun___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instNeg___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNegUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instNegUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instInvUniformFun___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instInv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInvUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instInvUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNegUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instInvUniformFun___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instNeg___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNegUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instNegUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDivUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instDiv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDivUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instDivUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSubUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instSub___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSubUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instSubUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDivUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instDiv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDivUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instDivUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSubUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instSub___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSubUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instSubUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformFun___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformFun___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instMonoidUniformFun___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMonoidUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_monoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instMonoidUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformFun___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformFun___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instAddMonoidUniformFun___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instAddMonoidUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instAddMonoidUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMonoidUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_monoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMonoidUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instMonoidUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instAddMonoidUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instAddMonoidUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMonoidUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_commMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instCommMonoidUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommMonoidUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instAddMonoidUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addCommMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommMonoidUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instAddCommMonoidUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMonoidUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_commMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instCommMonoidUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommMonoidUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instAddMonoidUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addCommMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommMonoidUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instAddCommMonoidUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformFun___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformFun___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instGroupUniformFun___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instGroupUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_group___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instGroupUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformFun___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformFun___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instAddGroupUniformFun___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instAddGroupUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instAddGroupUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instGroupUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_group___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGroupUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instGroupUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instAddGroupUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddGroupUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instAddGroupUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommGroupUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instGroupUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_commGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommGroupUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instCommGroupUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instAddGroupUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addCommGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instAddCommGroupUniformFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommGroupUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instGroupUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_commGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommGroupUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instCommGroupUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instAddGroupUniformFun___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_addCommGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instAddCommGroupUniformOnFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instSMul___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instSMulUniformFun___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_instSMul___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instSMulUniformOnFun___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_mulAction___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instMulActionUniformFun___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformFun___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instMulActionUniformFun(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_mulAction___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_instMulActionUniformOnFun___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionUniformOnFun___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_instMulActionUniformOnFun(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_distribMulAction___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_instDistribMulActionUniformFun___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformFun___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_instDistribMulActionUniformFun(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_distribMulAction___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_instDistribMulActionUniformOnFun___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribMulActionUniformOnFun___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_instDistribMulActionUniformOnFun(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_module___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_instModuleUniformFun___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformFun___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_instModuleUniformFun(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformOnFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_instMulUniformFun___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Pi_module___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformOnFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_instModuleUniformOnFun___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleUniformOnFun___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_instModuleUniformOnFun(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_UniformMulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_UniformConvergenceTopology(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Algebra_UniformConvergence(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_UniformMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_UniformConvergenceTopology(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
