// Lean compiler output
// Module: Mathlib.Algebra.Order.AddGroupWithTop
// Imports: public import Init public import Mathlib.Algebra.CharZero.Defs public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Algebra.Order.Monoid.Canonical.Defs public import Mathlib.Algebra.Order.Monoid.WithTop import Mathlib.Tactic.TermCongr
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
lean_object* lp_mathlib_WithTop_linearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg(lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop(lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg(lean_object*);
lean_object* lp_mathlib_zsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_1, 2);
x_8 = lean_ctor_get(x_1, 3);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_1);
x_9 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_5);
x_10 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_6);
lean_ctor_set(x_10, 2, x_7);
lean_ctor_set(x_10, 3, x_8);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_dec(x_2);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_4);
lean_ctor_set(x_7, 1, x_5);
lean_ctor_set(x_7, 2, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_WithTop_addMonoid___redArg(x_1);
x_4 = lp_mathlib_WithTop_linearOrder___redArg(x_2);
x_5 = lean_box(0);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_dec(x_3);
lean_dec(x_2);
lean_inc(x_1);
return x_1;
}
else
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec_ref(x_4);
lean_dec(x_2);
lean_inc(x_1);
return x_1;
}
else
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_apply_2(x_2, x_7, x_5);
lean_ctor_set(x_3, 0, x_8);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_3, 0);
lean_inc(x_9);
lean_dec(x_3);
x_10 = lean_apply_2(x_2, x_9, x_5);
x_11 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 2);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lean_box(0);
x_4 = lean_alloc_closure((void*)(lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_4, 0, x_3);
lean_closure_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop___redArg(x_3, x_2);
x_5 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
x_8 = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(x_1);
x_9 = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg(x_1);
x_10 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_10, 0, x_6);
x_11 = lp_mathlib_WithTop_add___redArg(x_7);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
x_12 = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_10);
lean_closure_set(x_12, 2, x_11);
lean_inc_ref(x_8);
x_13 = lean_alloc_closure((void*)(lp_mathlib_zsmulRec___boxed), 7, 5);
lean_closure_set(x_13, 0, lean_box(0));
lean_closure_set(x_13, 1, x_10);
lean_closure_set(x_13, 2, x_11);
lean_closure_set(x_13, 3, x_8);
lean_closure_set(x_13, 4, x_12);
x_14 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_14, 0, x_4);
lean_ctor_set(x_14, 1, x_8);
lean_ctor_set(x_14, 2, x_9);
lean_ctor_set(x_14, 3, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid___redArg(x_2, x_3);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
