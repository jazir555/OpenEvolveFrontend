// Lean compiler output
// Module: Mathlib.Algebra.NonAssoc.LieAdmissible.Defs
// Imports: public import Init public import Mathlib.Algebra.Lie.Basic public import Mathlib.Algebra.NonAssoc.PreLie.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instLieAdmissibleRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLieAdmissibleRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLieAdmissibleRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instLieAdmissibleRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLieAdmissibleRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instLieAdmissibleRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleRing_instLieRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLieAdmissibleRing___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Ring_instBracket___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instLieAdmissibleRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleRing_instLieRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleRing_instLieRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_Ring_instBracket___redArg(x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleRing_instLieRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LieAdmissibleRing_instLieRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_LieAdmissibleAlgebra_instLieAlgebra___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instLieAdmissibleRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instLieAdmissibleRing___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instLieAdmissibleRing___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LeftPreLieRing_instLieAdmissibleRing(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instLieAdmissibleRing___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_LeftPreLieRing_instLieAdmissibleRing___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_LeftPreLieAlgebra_instLieAdmissibleAlgebra___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLieAdmissibleRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLieAdmissibleRing___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLieAdmissibleRing___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RightPreLieRing_instLieAdmissibleRing(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLieAdmissibleRing___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RightPreLieRing_instLieAdmissibleRing___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RightPreLieAlgebra_instLieAdmissibleAlgebra___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_NonAssoc_PreLie_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_NonAssoc_LieAdmissible_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_NonAssoc_PreLie_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
