// Lean compiler output
// Module: Mathlib.FieldTheory.IntermediateField.Adjoin.Algebra
// Imports: public import Init public import Mathlib.FieldTheory.Finiteness public import Mathlib.FieldTheory.IntermediateField.Adjoin.Defs public import Mathlib.FieldTheory.IntermediateField.Algebraic
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg(lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_2(x_7, x_2, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___lam__0(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Set_inclusion___boxed), 5, 4);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
lean_closure_set(x_1, 2, lean_box(0));
lean_closure_set(x_1, 3, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_Field_toSemifield___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___closed__0;
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_FieldTheory_Finiteness(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_FieldTheory_IntermediateField_Adjoin_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_FieldTheory_IntermediateField_Algebraic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_FieldTheory_IntermediateField_Adjoin_Algebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_FieldTheory_Finiteness(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_FieldTheory_IntermediateField_Adjoin_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_FieldTheory_IntermediateField_Algebraic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___closed__0 = _init_lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___closed__0();
lean_mark_persistent(lp_mathlib_IntermediateField_algebraAdjoinAdjoin_instAlgebraSubtypeMemSubalgebraAdjoinAdjoin___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
