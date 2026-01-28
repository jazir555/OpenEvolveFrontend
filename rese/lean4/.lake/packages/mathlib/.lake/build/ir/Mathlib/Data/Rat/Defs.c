// Lean compiler output
// Module: Mathlib.Data.Rat.Defs
// Imports: public import Init public import Mathlib.Algebra.Group.Defs public import Mathlib.Data.Nat.Basic public import Mathlib.Data.Rat.Init public import Mathlib.Order.Basic public import Mathlib.Tactic.Common
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
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommMonoid;
static lean_object* lp_mathlib_Rat_commMonoid___closed__1;
static lean_object* lp_mathlib_Rat_addCommGroup___closed__2;
static lean_object* lp_mathlib_Rat_addCommGroup___closed__0;
static lean_object* lp_mathlib_Rat_addCommGroup___closed__1;
lean_object* l_Rat_mul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_addRightCancelSemigroup;
lean_object* l_Rat_instNatCast___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_addMonoid;
static lean_object* lp_mathlib_Rat_commMonoid___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid;
lean_object* l_Rat_mul(lean_object*, lean_object*);
lean_object* l_Rat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_semigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_divCasesOn(lean_object*, lean_object*, lean_object*);
lean_object* l_Rat_pow(lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_addCommGroup___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid___lam__0(lean_object*, lean_object*);
lean_object* l_Rat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_addGroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_divCasesOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_monoid;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addLeftCancelSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommSemigroup;
lean_object* l_Rat_ofInt(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup;
lean_object* l_Rat_neg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = l_Rat_instNatCast___lam__0(x_1);
x_4 = l_Rat_mul(x_3, x_2);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = l_Rat_ofInt(x_1);
x_4 = l_Rat_mul(x_3, x_2);
lean_dec_ref(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommGroup___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_add), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommGroup___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_neg), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommGroup___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_sub), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommGroup___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = l_Rat_instNatCast___lam__0(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommGroup() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Rat_addCommGroup___lam__0), 2, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Rat_addCommGroup___lam__1), 2, 0);
x_3 = lp_mathlib_Rat_addCommGroup___closed__0;
x_4 = lp_mathlib_Rat_addCommGroup___closed__1;
x_5 = lp_mathlib_Rat_addCommGroup___closed__2;
x_6 = lp_mathlib_Rat_addCommGroup___closed__3;
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_3);
lean_ctor_set(x_7, 1, x_6);
lean_ctor_set(x_7, 2, x_1);
x_8 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_4);
lean_ctor_set(x_8, 2, x_5);
lean_ctor_set(x_8, 3, x_2);
return x_8;
}
}
static lean_object* _init_lp_mathlib_Rat_addGroup() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_addCommGroup;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommMonoid() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_addCommGroup;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_addMonoid() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_addCommGroup;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_addLeftCancelSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Rat_addCommGroup;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_addRightCancelSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Rat_addCommGroup;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_addCommMonoid;
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_addSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_addMonoid;
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_Rat_pow(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_commMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_commMonoid___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = l_Rat_instNatCast___lam__0(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Rat_commMonoid___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_commMonoid() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Rat_commMonoid___lam__0___boxed), 2, 0);
x_2 = lp_mathlib_Rat_commMonoid___closed__0;
x_3 = lp_mathlib_Rat_commMonoid___closed__1;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Rat_monoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_commMonoid;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_commSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_commMonoid;
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_semigroup() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_commMonoid___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_divCasesOn___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_4(x_2, x_3, x_4, lean_box(0), lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_divCasesOn(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Rat_divCasesOn___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_addCommGroup___closed__0 = _init_lp_mathlib_Rat_addCommGroup___closed__0();
lean_mark_persistent(lp_mathlib_Rat_addCommGroup___closed__0);
lp_mathlib_Rat_addCommGroup___closed__1 = _init_lp_mathlib_Rat_addCommGroup___closed__1();
lean_mark_persistent(lp_mathlib_Rat_addCommGroup___closed__1);
lp_mathlib_Rat_addCommGroup___closed__2 = _init_lp_mathlib_Rat_addCommGroup___closed__2();
lean_mark_persistent(lp_mathlib_Rat_addCommGroup___closed__2);
lp_mathlib_Rat_addCommGroup___closed__3 = _init_lp_mathlib_Rat_addCommGroup___closed__3();
lean_mark_persistent(lp_mathlib_Rat_addCommGroup___closed__3);
lp_mathlib_Rat_addCommGroup = _init_lp_mathlib_Rat_addCommGroup();
lean_mark_persistent(lp_mathlib_Rat_addCommGroup);
lp_mathlib_Rat_addGroup = _init_lp_mathlib_Rat_addGroup();
lean_mark_persistent(lp_mathlib_Rat_addGroup);
lp_mathlib_Rat_addCommMonoid = _init_lp_mathlib_Rat_addCommMonoid();
lean_mark_persistent(lp_mathlib_Rat_addCommMonoid);
lp_mathlib_Rat_addMonoid = _init_lp_mathlib_Rat_addMonoid();
lean_mark_persistent(lp_mathlib_Rat_addMonoid);
lp_mathlib_Rat_addLeftCancelSemigroup = _init_lp_mathlib_Rat_addLeftCancelSemigroup();
lean_mark_persistent(lp_mathlib_Rat_addLeftCancelSemigroup);
lp_mathlib_Rat_addRightCancelSemigroup = _init_lp_mathlib_Rat_addRightCancelSemigroup();
lean_mark_persistent(lp_mathlib_Rat_addRightCancelSemigroup);
lp_mathlib_Rat_addCommSemigroup = _init_lp_mathlib_Rat_addCommSemigroup();
lean_mark_persistent(lp_mathlib_Rat_addCommSemigroup);
lp_mathlib_Rat_addSemigroup = _init_lp_mathlib_Rat_addSemigroup();
lean_mark_persistent(lp_mathlib_Rat_addSemigroup);
lp_mathlib_Rat_commMonoid___closed__0 = _init_lp_mathlib_Rat_commMonoid___closed__0();
lean_mark_persistent(lp_mathlib_Rat_commMonoid___closed__0);
lp_mathlib_Rat_commMonoid___closed__1 = _init_lp_mathlib_Rat_commMonoid___closed__1();
lean_mark_persistent(lp_mathlib_Rat_commMonoid___closed__1);
lp_mathlib_Rat_commMonoid = _init_lp_mathlib_Rat_commMonoid();
lean_mark_persistent(lp_mathlib_Rat_commMonoid);
lp_mathlib_Rat_monoid = _init_lp_mathlib_Rat_monoid();
lean_mark_persistent(lp_mathlib_Rat_monoid);
lp_mathlib_Rat_commSemigroup = _init_lp_mathlib_Rat_commSemigroup();
lean_mark_persistent(lp_mathlib_Rat_commSemigroup);
lp_mathlib_Rat_semigroup = _init_lp_mathlib_Rat_semigroup();
lean_mark_persistent(lp_mathlib_Rat_semigroup);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
