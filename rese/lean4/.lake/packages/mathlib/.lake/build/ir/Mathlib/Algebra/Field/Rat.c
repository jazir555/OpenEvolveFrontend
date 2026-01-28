// Lean compiler output
// Module: Mathlib.Algebra.Field.Rat
// Imports: public import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Data.NNRat.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instInv;
static lean_object* lp_mathlib_Rat_instField___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield;
static lean_object* lp_mathlib_NNRat_instSemifield___closed__2;
static lean_object* lp_mathlib_NNRat_instSemifield___closed__0;
lean_object* lp_mathlib_Rat_instNNRatCast___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instDiv___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instDiv;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__0(lean_object*, lean_object*);
lean_object* lp_batteries_instRatCastRat___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instDivisionRing;
static lean_object* lp_mathlib_NNRat_instSemifield___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instDiv___lam__0(lean_object*, lean_object*);
lean_object* l_Rat_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instInv___lam__0(lean_object*);
extern lean_object* lp_mathlib_instCommSemiringNNRat;
lean_object* lp_mathlib_NNRat_instNNRatCast___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__1___boxed(lean_object*, lean_object*);
lean_object* l_Rat_inv(lean_object*);
static lean_object* lp_mathlib_Rat_instField___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__0___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_NNRat_instSemifield___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instZPow___lam__0(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Rat_commGroupWithZero;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instZPow;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instZPow___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Rat_zpow(lean_object*, lean_object*);
lean_object* l_Rat_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__0(lean_object*, lean_object*);
lean_object* lp_batteries_instRatCastRat___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
extern lean_object* lp_mathlib_Rat_commRing;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instInv___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Rat_instNNRatCast___lam__0(lean_object*);
static lean_object* lp_mathlib_Rat_instDivisionRing___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_Rat_zpow(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Rat_instNNRatCast___lam__0(x_1);
x_4 = l_Rat_mul(x_3, x_2);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_batteries_instRatCastRat___lam__0(x_1);
x_4 = l_Rat_mul(x_3, x_2);
lean_dec_ref(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Rat_instField___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Rat_instNNRatCast___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_instField___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_batteries_instRatCastRat___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Rat_instField___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Rat_instField___lam__1(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Rat_instField___lam__2(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_instField() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_1 = lp_mathlib_Rat_commRing;
x_2 = lp_mathlib_Rat_commGroupWithZero;
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 2);
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Rat_instField___lam__0___boxed), 2, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Rat_instField___lam__1___boxed), 2, 0);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Rat_instField___lam__2___boxed), 2, 0);
x_8 = lp_mathlib_Rat_instField___closed__0;
x_9 = lp_mathlib_Rat_instField___closed__1;
x_10 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_10, 0, x_1);
lean_ctor_set(x_10, 1, x_3);
lean_ctor_set(x_10, 2, x_4);
lean_ctor_set(x_10, 3, x_5);
lean_ctor_set(x_10, 4, x_8);
lean_ctor_set(x_10, 5, x_9);
lean_ctor_set(x_10, 6, x_6);
lean_ctor_set(x_10, 7, x_7);
return x_10;
}
}
static lean_object* _init_lp_mathlib_Rat_instDivisionRing___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_instField;
x_2 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_instDivisionRing() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_instDivisionRing___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instInv___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Rat_instNNRatCast___lam__0(x_1);
x_3 = l_Rat_inv(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instInv___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NNRat_instInv___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_NNRat_instInv() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NNRat_instInv___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instDiv___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Rat_instNNRatCast___lam__0(x_1);
x_4 = lp_mathlib_Rat_instNNRatCast___lam__0(x_2);
x_5 = l_Rat_div(x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instDiv___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NNRat_instDiv___lam__0(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_NNRat_instDiv() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NNRat_instDiv___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instZPow___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Rat_instNNRatCast___lam__0(x_1);
x_4 = l_Rat_zpow(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instZPow___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NNRat_instZPow___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_NNRat_instZPow() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NNRat_instZPow___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Rat_instNNRatCast___lam__0(x_2);
x_4 = l_Rat_zpow(x_3, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_NNRat_instSemifield___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_instCommSemiringNNRat;
x_2 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_NNRat_instSemifield___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NNRat_instInv___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_NNRat_instSemifield___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NNRat_instDiv___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_NNRat_instSemifield___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NNRat_instNNRatCast___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NNRat_instSemifield___lam__0(x_1, x_2);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_NNRat_instSemifield() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_1 = lp_mathlib_instCommSemiringNNRat;
x_2 = lp_mathlib_NNRat_instSemifield___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_NNRat_instSemifield___lam__0___boxed), 2, 0);
x_7 = lp_mathlib_NNRat_instSemifield___closed__1;
x_8 = lp_mathlib_NNRat_instSemifield___closed__2;
x_9 = lp_mathlib_NNRat_instSemifield___closed__3;
x_10 = lean_alloc_closure((void*)(lp_mathlib_NNRat_instSemifield___lam__1), 3, 1);
lean_closure_set(x_10, 0, x_5);
x_11 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_11, 0, x_1);
lean_ctor_set(x_11, 1, x_7);
lean_ctor_set(x_11, 2, x_8);
lean_ctor_set(x_11, 3, x_6);
lean_ctor_set(x_11, 4, x_9);
lean_ctor_set(x_11, 5, x_10);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_NNRat_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_NNRat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_instField___closed__0 = _init_lp_mathlib_Rat_instField___closed__0();
lean_mark_persistent(lp_mathlib_Rat_instField___closed__0);
lp_mathlib_Rat_instField___closed__1 = _init_lp_mathlib_Rat_instField___closed__1();
lean_mark_persistent(lp_mathlib_Rat_instField___closed__1);
lp_mathlib_Rat_instField = _init_lp_mathlib_Rat_instField();
lean_mark_persistent(lp_mathlib_Rat_instField);
lp_mathlib_Rat_instDivisionRing___closed__0 = _init_lp_mathlib_Rat_instDivisionRing___closed__0();
lean_mark_persistent(lp_mathlib_Rat_instDivisionRing___closed__0);
lp_mathlib_Rat_instDivisionRing = _init_lp_mathlib_Rat_instDivisionRing();
lean_mark_persistent(lp_mathlib_Rat_instDivisionRing);
lp_mathlib_NNRat_instInv = _init_lp_mathlib_NNRat_instInv();
lean_mark_persistent(lp_mathlib_NNRat_instInv);
lp_mathlib_NNRat_instDiv = _init_lp_mathlib_NNRat_instDiv();
lean_mark_persistent(lp_mathlib_NNRat_instDiv);
lp_mathlib_NNRat_instZPow = _init_lp_mathlib_NNRat_instZPow();
lean_mark_persistent(lp_mathlib_NNRat_instZPow);
lp_mathlib_NNRat_instSemifield___closed__0 = _init_lp_mathlib_NNRat_instSemifield___closed__0();
lean_mark_persistent(lp_mathlib_NNRat_instSemifield___closed__0);
lp_mathlib_NNRat_instSemifield___closed__1 = _init_lp_mathlib_NNRat_instSemifield___closed__1();
lean_mark_persistent(lp_mathlib_NNRat_instSemifield___closed__1);
lp_mathlib_NNRat_instSemifield___closed__2 = _init_lp_mathlib_NNRat_instSemifield___closed__2();
lean_mark_persistent(lp_mathlib_NNRat_instSemifield___closed__2);
lp_mathlib_NNRat_instSemifield___closed__3 = _init_lp_mathlib_NNRat_instSemifield___closed__3();
lean_mark_persistent(lp_mathlib_NNRat_instSemifield___closed__3);
lp_mathlib_NNRat_instSemifield = _init_lp_mathlib_NNRat_instSemifield();
lean_mark_persistent(lp_mathlib_NNRat_instSemifield);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
