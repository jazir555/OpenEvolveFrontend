// Lean compiler output
// Module: Mathlib.Algebra.Group.Int.Defs
// Imports: public import Init public import Mathlib.Algebra.Group.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddCommGroup;
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddCommMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommMonoid___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_instAddCommGroup___closed__3;
static lean_object* lp_mathlib_Int_instAddCommGroup___closed__1;
lean_object* l_Int_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Int_instSemigroup;
lean_object* l_Int_sub___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_instCommMonoid___closed__0;
lean_object* l_instNatCastInt___lam__0(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddMonoid;
lean_object* l_Int_pow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instMonoid;
static lean_object* lp_mathlib_Int_instAddCommGroup___closed__0;
static lean_object* lp_mathlib_Int_instAddCommGroup___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommMonoid___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddCommGroup___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddGroup;
lean_object* l_Int_neg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddCommGroup___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Int_mul___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_instCommMonoid___closed__1;
static lean_object* lp_mathlib_Int_instAddCommGroup___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddCommSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommMonoid___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_Int_pow(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Int_instCommMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instCommMonoid___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommMonoid___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Int_instCommMonoid___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Int_instCommMonoid() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Int_instCommMonoid___lam__0___boxed), 2, 0);
x_2 = lp_mathlib_Int_instCommMonoid___closed__0;
x_3 = lp_mathlib_Int_instCommMonoid___closed__1;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddCommGroup___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = l_instNatCastInt___lam__0(x_1);
x_4 = lean_int_mul(x_3, x_2);
lean_dec(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Int_instAddCommGroup___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_mul___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instAddCommGroup___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_add___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instAddCommGroup___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_neg___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instAddCommGroup___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_sub___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instAddCommGroup___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instAddCommGroup___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Int_instAddCommGroup___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Int_instAddCommGroup() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_1 = lp_mathlib_Int_instAddCommGroup___closed__0;
x_2 = lean_alloc_closure((void*)(lp_mathlib_Int_instAddCommGroup___lam__0___boxed), 2, 0);
x_3 = lp_mathlib_Int_instAddCommGroup___closed__1;
x_4 = lp_mathlib_Int_instAddCommGroup___closed__2;
x_5 = lp_mathlib_Int_instAddCommGroup___closed__3;
x_6 = lp_mathlib_Int_instAddCommGroup___closed__4;
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_3);
lean_ctor_set(x_7, 1, x_6);
lean_ctor_set(x_7, 2, x_2);
x_8 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_4);
lean_ctor_set(x_8, 2, x_5);
lean_ctor_set(x_8, 3, x_1);
return x_8;
}
}
static lean_object* _init_lp_mathlib_Int_instAddCommMonoid() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instAddCommGroup;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instAddMonoid() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instAddCommGroup;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instCommMonoid;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instCommSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instCommMonoid;
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instSemigroup() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instCommMonoid___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instAddGroup() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_instAddCommGroup;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instAddCommSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instAddCommMonoid;
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instAddSemigroup() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instAddMonoid;
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_instCommMonoid___closed__0 = _init_lp_mathlib_Int_instCommMonoid___closed__0();
lean_mark_persistent(lp_mathlib_Int_instCommMonoid___closed__0);
lp_mathlib_Int_instCommMonoid___closed__1 = _init_lp_mathlib_Int_instCommMonoid___closed__1();
lean_mark_persistent(lp_mathlib_Int_instCommMonoid___closed__1);
lp_mathlib_Int_instCommMonoid = _init_lp_mathlib_Int_instCommMonoid();
lean_mark_persistent(lp_mathlib_Int_instCommMonoid);
lp_mathlib_Int_instAddCommGroup___closed__0 = _init_lp_mathlib_Int_instAddCommGroup___closed__0();
lean_mark_persistent(lp_mathlib_Int_instAddCommGroup___closed__0);
lp_mathlib_Int_instAddCommGroup___closed__1 = _init_lp_mathlib_Int_instAddCommGroup___closed__1();
lean_mark_persistent(lp_mathlib_Int_instAddCommGroup___closed__1);
lp_mathlib_Int_instAddCommGroup___closed__2 = _init_lp_mathlib_Int_instAddCommGroup___closed__2();
lean_mark_persistent(lp_mathlib_Int_instAddCommGroup___closed__2);
lp_mathlib_Int_instAddCommGroup___closed__3 = _init_lp_mathlib_Int_instAddCommGroup___closed__3();
lean_mark_persistent(lp_mathlib_Int_instAddCommGroup___closed__3);
lp_mathlib_Int_instAddCommGroup___closed__4 = _init_lp_mathlib_Int_instAddCommGroup___closed__4();
lean_mark_persistent(lp_mathlib_Int_instAddCommGroup___closed__4);
lp_mathlib_Int_instAddCommGroup = _init_lp_mathlib_Int_instAddCommGroup();
lean_mark_persistent(lp_mathlib_Int_instAddCommGroup);
lp_mathlib_Int_instAddCommMonoid = _init_lp_mathlib_Int_instAddCommMonoid();
lean_mark_persistent(lp_mathlib_Int_instAddCommMonoid);
lp_mathlib_Int_instAddMonoid = _init_lp_mathlib_Int_instAddMonoid();
lean_mark_persistent(lp_mathlib_Int_instAddMonoid);
lp_mathlib_Int_instMonoid = _init_lp_mathlib_Int_instMonoid();
lean_mark_persistent(lp_mathlib_Int_instMonoid);
lp_mathlib_Int_instCommSemigroup = _init_lp_mathlib_Int_instCommSemigroup();
lean_mark_persistent(lp_mathlib_Int_instCommSemigroup);
lp_mathlib_Int_instSemigroup = _init_lp_mathlib_Int_instSemigroup();
lean_mark_persistent(lp_mathlib_Int_instSemigroup);
lp_mathlib_Int_instAddGroup = _init_lp_mathlib_Int_instAddGroup();
lean_mark_persistent(lp_mathlib_Int_instAddGroup);
lp_mathlib_Int_instAddCommSemigroup = _init_lp_mathlib_Int_instAddCommSemigroup();
lean_mark_persistent(lp_mathlib_Int_instAddCommSemigroup);
lp_mathlib_Int_instAddSemigroup = _init_lp_mathlib_Int_instAddSemigroup();
lean_mark_persistent(lp_mathlib_Int_instAddSemigroup);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
