// Lean compiler output
// Module: Mathlib.Algebra.Group.Basic
// Imports: public import Init public import Aesop public import Mathlib.Algebra.Group.Defs public import Mathlib.Data.Int.Init public import Mathlib.Logic.Function.Iterate public import Mathlib.Tactic.SimpRw
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
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toGrindIntModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toGrindIntModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DivisionMonoid_toDivInvOneMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_2);
lean_inc(x_3);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_2);
lean_inc(x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommMonoid_toGrindNatModule___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommMonoid_toGrindNatModule(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCommMonoid_toGrindNatModule___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toGrindIntModule___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 1);
x_9 = lean_ctor_get(x_2, 2);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_7);
lean_ctor_set(x_2, 2, x_4);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_10);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_2);
lean_ctor_set(x_11, 1, x_9);
lean_ctor_set(x_11, 2, x_5);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
x_14 = lean_ctor_get(x_2, 2);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_12);
x_16 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_3);
lean_ctor_set(x_16, 2, x_4);
x_17 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_14);
lean_ctor_set(x_17, 2, x_5);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toGrindIntModule(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommGroup_toGrindIntModule___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
