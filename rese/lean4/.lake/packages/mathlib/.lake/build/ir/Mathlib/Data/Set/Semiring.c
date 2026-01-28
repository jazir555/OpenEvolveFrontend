// Lean compiler output
// Module: Mathlib.Data.Set.Semiring
// Imports: public import Init public import Mathlib.Algebra.Order.Kleene public import Mathlib.Algebra.Order.Ring.Canonical public import Mathlib.Data.Set.BooleanAlgebra public import Mathlib.Algebra.Group.Pointwise.Set.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_singletonMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_down(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_singletonMonoidHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOrderBotSetSemiring(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring(lean_object*, lean_object*);
static lean_object* lp_mathlib_instPartialOrderSetSemiring___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instAdd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instAddCommMonoid(lean_object*);
static lean_object* lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__1;
static lean_object* lp_mathlib_Set_up___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSetSemiring(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instOne(lean_object*, lean_object*);
static lean_object* lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderSetSemiring(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instZero(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_up(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_SetSemiring_instAddCommMonoid___closed__0;
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup___redArg(lean_object*);
lean_object* lp_mathlib_Set_instCompleteAtomicBooleanAlgebra(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSetSemiring(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
static lean_object* _init_lp_mathlib_instPartialOrderSetSemiring___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Set_instCompleteAtomicBooleanAlgebra(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderSetSemiring(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_instPartialOrderSetSemiring___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOrderBotSetSemiring(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
static lean_object* _init_lp_mathlib_Set_up___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_up(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_up___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_down(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_up___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instZero(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instAdd(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
static lean_object* _init_lp_mathlib_SetSemiring_instAddCommMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_1, 0, lean_box(0));
lean_ctor_set(x_1, 1, lean_box(0));
lean_ctor_set(x_1, 2, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instAddCommMonoid(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SetSemiring_instAddCommMonoid___closed__0;
return x_2;
}
}
static lean_object* _init_lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_SetSemiring_instAddCommMonoid(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, lean_box(0));
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__1;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instOne(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instOne___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SetSemiring_instOne(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring(lean_box(0), x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring(lean_box(0), x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SetSemiring_instNonUnitalSemiringOfSemigroup___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring(lean_box(0), x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring(lean_box(0), x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SetSemiring_instNonUnitalCommSemiringOfCommSemigroup___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_singletonMonoidHom(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetSemiring_singletonMonoidHom___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SetSemiring_singletonMonoidHom(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Kleene(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Semiring(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Kleene(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instPartialOrderSetSemiring___closed__0 = _init_lp_mathlib_instPartialOrderSetSemiring___closed__0();
lean_mark_persistent(lp_mathlib_instPartialOrderSetSemiring___closed__0);
lp_mathlib_Set_up___closed__0 = _init_lp_mathlib_Set_up___closed__0();
lean_mark_persistent(lp_mathlib_Set_up___closed__0);
lp_mathlib_SetSemiring_instAddCommMonoid___closed__0 = _init_lp_mathlib_SetSemiring_instAddCommMonoid___closed__0();
lean_mark_persistent(lp_mathlib_SetSemiring_instAddCommMonoid___closed__0);
lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__0 = _init_lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__0();
lean_mark_persistent(lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__0);
lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__1 = _init_lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__1();
lean_mark_persistent(lp_mathlib_SetSemiring_instNonUnitalNonAssocSemiring___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
