// Lean compiler output
// Module: Mathlib.Analysis.Normed.Field.Lemmas
// Imports: public import Init public import Mathlib.Analysis.Normed.Field.Basic public import Mathlib.Analysis.Normed.Group.Rat public import Mathlib.Analysis.Normed.Ring.Lemmas public import Mathlib.Topology.MetricSpace.DilationEquiv
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
lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulRight___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_mulLeft_u2080___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instDenselyNormedField;
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulLeft(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_instNormedField___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instNormedField;
extern lean_object* lp_mathlib_Rat_instField;
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulRight___redArg(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Rat_instNormedAddCommGroup;
lean_object* lp_mathlib_Equiv_mulRight_u2080___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_instNormedAddCommGroup___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulLeft___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_3);
x_5 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_4);
x_6 = lp_mathlib_Equiv_mulLeft_u2080___redArg(x_5, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DilationEquiv_mulLeft___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulLeft___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DilationEquiv_mulLeft(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulLeft___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DilationEquiv_mulLeft___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulRight___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_3);
x_5 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_4);
x_6 = lp_mathlib_Equiv_mulRight_u2080___redArg(x_5, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DilationEquiv_mulRight___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulRight___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DilationEquiv_mulRight(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DilationEquiv_mulRight___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DilationEquiv_mulRight___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_instNormedField___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Rat_instNormedAddCommGroup___lam__0), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_instNormedField() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; 
x_1 = lp_mathlib_Rat_instField;
x_2 = lp_mathlib_Rat_instNormedAddCommGroup;
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_2, 1);
lean_dec(x_4);
x_5 = lean_ctor_get(x_2, 0);
lean_dec(x_5);
x_6 = lp_mathlib_Rat_instNormedField___closed__0;
lean_ctor_set(x_2, 1, x_1);
lean_ctor_set(x_2, 0, x_6);
return x_2;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_2, 2);
lean_inc(x_7);
lean_dec(x_2);
x_8 = lp_mathlib_Rat_instNormedField___closed__0;
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_1);
lean_ctor_set(x_9, 2, x_7);
return x_9;
}
}
}
static lean_object* _init_lp_mathlib_Rat_instDenselyNormedField() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_instNormedField;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Field_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Ring_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_DilationEquiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Field_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Group_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Ring_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_DilationEquiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_instNormedField___closed__0 = _init_lp_mathlib_Rat_instNormedField___closed__0();
lean_mark_persistent(lp_mathlib_Rat_instNormedField___closed__0);
lp_mathlib_Rat_instNormedField = _init_lp_mathlib_Rat_instNormedField();
lean_mark_persistent(lp_mathlib_Rat_instNormedField);
lp_mathlib_Rat_instDenselyNormedField = _init_lp_mathlib_Rat_instDenselyNormedField();
lean_mark_persistent(lp_mathlib_Rat_instDenselyNormedField);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
