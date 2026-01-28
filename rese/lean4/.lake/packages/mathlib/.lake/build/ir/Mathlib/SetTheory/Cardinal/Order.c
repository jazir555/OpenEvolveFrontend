// Lean compiler output
// Module: Mathlib.SetTheory.Cardinal.Order
// Imports: public import Init public import Mathlib.Algebra.Order.GroupWithZero.Canonical public import Mathlib.Algebra.Order.Ring.Canonical public import Mathlib.Data.Fintype.Option public import Mathlib.Order.InitialSeg public import Mathlib.Order.Nat public import Mathlib.Order.SuccPred.CompleteLinearOrder public import Mathlib.SetTheory.Cardinal.Defs public import Mathlib.SetTheory.Cardinal.SchroederBernstein
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
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_liftInitialSeg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instWellFoundedRelation;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring;
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Cardinal_instMul;
lean_object* lp_mathlib_Cardinal_instNatCast___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_orderBot;
static lean_object* lp_mathlib_Cardinal_partialOrder___closed__0;
extern lean_object* lp_mathlib_Cardinal_instAdd;
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instLE;
static lean_object* lp_mathlib_Cardinal_instCommMonoidWithZero___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_partialOrder;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_liftInitialSeg;
lean_object* lp_mathlib_Quotient_map_u2082___at___00Cardinal_map_u2082_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instCommMonoidWithZero;
static lean_object* lp_mathlib_Cardinal_commSemiring___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_liftInitialSeg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0___boxed(lean_object*);
static lean_object* lp_mathlib_Cardinal_commSemiring___closed__2;
static lean_object* lp_mathlib_Cardinal_instCommMonoidWithZero___closed__2;
static lean_object* lp_mathlib_Cardinal_instCommMonoidWithZero___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instCommMonoid;
static lean_object* lp_mathlib_Cardinal_commSemiring___closed__1;
static lean_object* _init_lp_mathlib_Cardinal_instLE() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Cardinal_partialOrder___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Cardinal_partialOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Cardinal_partialOrder___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_liftInitialSeg___lam__0(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_liftInitialSeg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Cardinal_liftInitialSeg___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Cardinal_liftInitialSeg() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Cardinal_liftInitialSeg___lam__0___boxed), 1, 0);
x_2 = lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__0(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Cardinal_instNatCast___lam__0(x_1);
x_4 = lp_mathlib_Quotient_map_u2082___at___00Cardinal_map_u2082_spec__0(lean_box(0), lean_box(0), x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Cardinal_commSemiring___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Cardinal_instAdd;
x_2 = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Cardinal_commSemiring___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Cardinal_commSemiring___closed__0;
x_2 = lp_mathlib_Cardinal_instAdd;
x_3 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, lean_box(0));
lean_ctor_set(x_3, 2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Cardinal_commSemiring___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Cardinal_instMul;
x_2 = lp_mathlib_Cardinal_commSemiring___closed__1;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Cardinal_commSemiring___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Cardinal_commSemiring___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Cardinal_commSemiring() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Cardinal_commSemiring___lam__0___boxed), 1, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Cardinal_commSemiring___lam__1___boxed), 2, 0);
x_3 = lp_mathlib_Cardinal_commSemiring___closed__2;
x_4 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, lean_box(0));
lean_ctor_set(x_4, 2, x_1);
lean_ctor_set(x_4, 3, x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Cardinal_orderBot() {
_start:
{
return lean_box(0);
}
}
static lean_object* _init_lp_mathlib_Cardinal_instCommMonoidWithZero___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Cardinal_commSemiring___lam__1___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instCommMonoidWithZero___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Cardinal_instCommMonoidWithZero___closed__0;
x_2 = lp_mathlib_Cardinal_instMul;
x_3 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, lean_box(0));
lean_ctor_set(x_3, 2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instCommMonoidWithZero___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Cardinal_instCommMonoidWithZero___closed__1;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, lean_box(0));
return x_2;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instCommMonoidWithZero() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Cardinal_instCommMonoidWithZero___closed__2;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instCommMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Cardinal_instCommMonoidWithZero___closed__1;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instWellFoundedRelation() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_InitialSeg(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_CompleteLinearOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_SchroederBernstein(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_InitialSeg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_CompleteLinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_SchroederBernstein(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Cardinal_instLE = _init_lp_mathlib_Cardinal_instLE();
lean_mark_persistent(lp_mathlib_Cardinal_instLE);
lp_mathlib_Cardinal_partialOrder___closed__0 = _init_lp_mathlib_Cardinal_partialOrder___closed__0();
lean_mark_persistent(lp_mathlib_Cardinal_partialOrder___closed__0);
lp_mathlib_Cardinal_partialOrder = _init_lp_mathlib_Cardinal_partialOrder();
lean_mark_persistent(lp_mathlib_Cardinal_partialOrder);
lp_mathlib_Cardinal_liftInitialSeg = _init_lp_mathlib_Cardinal_liftInitialSeg();
lean_mark_persistent(lp_mathlib_Cardinal_liftInitialSeg);
lp_mathlib_Cardinal_commSemiring___closed__0 = _init_lp_mathlib_Cardinal_commSemiring___closed__0();
lean_mark_persistent(lp_mathlib_Cardinal_commSemiring___closed__0);
lp_mathlib_Cardinal_commSemiring___closed__1 = _init_lp_mathlib_Cardinal_commSemiring___closed__1();
lean_mark_persistent(lp_mathlib_Cardinal_commSemiring___closed__1);
lp_mathlib_Cardinal_commSemiring___closed__2 = _init_lp_mathlib_Cardinal_commSemiring___closed__2();
lean_mark_persistent(lp_mathlib_Cardinal_commSemiring___closed__2);
lp_mathlib_Cardinal_commSemiring = _init_lp_mathlib_Cardinal_commSemiring();
lean_mark_persistent(lp_mathlib_Cardinal_commSemiring);
lp_mathlib_Cardinal_orderBot = _init_lp_mathlib_Cardinal_orderBot();
lean_mark_persistent(lp_mathlib_Cardinal_orderBot);
lp_mathlib_Cardinal_instCommMonoidWithZero___closed__0 = _init_lp_mathlib_Cardinal_instCommMonoidWithZero___closed__0();
lean_mark_persistent(lp_mathlib_Cardinal_instCommMonoidWithZero___closed__0);
lp_mathlib_Cardinal_instCommMonoidWithZero___closed__1 = _init_lp_mathlib_Cardinal_instCommMonoidWithZero___closed__1();
lean_mark_persistent(lp_mathlib_Cardinal_instCommMonoidWithZero___closed__1);
lp_mathlib_Cardinal_instCommMonoidWithZero___closed__2 = _init_lp_mathlib_Cardinal_instCommMonoidWithZero___closed__2();
lean_mark_persistent(lp_mathlib_Cardinal_instCommMonoidWithZero___closed__2);
lp_mathlib_Cardinal_instCommMonoidWithZero = _init_lp_mathlib_Cardinal_instCommMonoidWithZero();
lean_mark_persistent(lp_mathlib_Cardinal_instCommMonoidWithZero);
lp_mathlib_Cardinal_instCommMonoid = _init_lp_mathlib_Cardinal_instCommMonoid();
lean_mark_persistent(lp_mathlib_Cardinal_instCommMonoid);
lp_mathlib_Cardinal_instWellFoundedRelation = _init_lp_mathlib_Cardinal_instWellFoundedRelation();
lean_mark_persistent(lp_mathlib_Cardinal_instWellFoundedRelation);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
