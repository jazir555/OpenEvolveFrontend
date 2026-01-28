// Lean compiler output
// Module: Mathlib.Data.PNat.Basic
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Divisibility public import Mathlib.Algebra.Order.Positive.Ring public import Mathlib.Algebra.Order.Ring.Nat public import Mathlib.Algebra.Order.Sub.Basic public import Mathlib.Data.PNat.Equiv
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
lean_object* lp_mathlib_Positive_instDistribSubtypeLtOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeAddHom___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_instDistribPNat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PNat_instCancelCommMonoid;
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_instAddPNat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PNat_instCommMonoid;
lean_object* lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeMonoidHom;
static lean_object* lp_mathlib_instAddLeftCancelSemigroupPNat___closed__0;
static lean_object* lp_mathlib_instMulPNat___closed__0;
lean_object* lp_mathlib_Nat_toPNat_x27(lean_object*);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
static lean_object* lp_mathlib_PNat_instCommMonoid___closed__0;
lean_object* lp_mathlib_Positive_instMonoidSubtypeLtOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulPNat;
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_PNat_coeMonoidHom___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommSemigroupPNat;
lean_object* lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_instSub;
LEAN_EXPORT lean_object* lp_mathlib_PNat_instOrderBot;
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeAddHom___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddRightCancelSemigroupPNat;
extern lean_object* lp_mathlib_Equiv_pnatEquivNat;
LEAN_EXPORT lean_object* lp_mathlib_instDistribPNat;
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeAddHom;
extern lean_object* lp_mathlib_Nat_instSemiring;
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_pnatIsoNat;
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PNat_strongInductionOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddLeftCancelSemigroupPNat;
LEAN_EXPORT lean_object* lp_mathlib_PNat_instSub___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddPNat;
LEAN_EXPORT lean_object* lp_mathlib_PNat_instSub___lam__0___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_instAddLeftCancelSemigroupPNat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid;
x_2 = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instAddLeftCancelSemigroupPNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instAddLeftCancelSemigroupPNat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_instAddRightCancelSemigroupPNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instAddLeftCancelSemigroupPNat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_instAddCommSemigroupPNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instAddLeftCancelSemigroupPNat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_instAddPNat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid;
x_2 = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instAddPNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instAddPNat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_instMulPNat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instSemiring;
x_2 = lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instMulPNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instMulPNat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_instDistribPNat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instSemiring;
x_2 = lp_mathlib_Positive_instDistribSubtypeLtOfNat___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instDistribPNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instDistribPNat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_PNat_instCommMonoid___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instSemiring;
x_2 = lp_mathlib_Positive_instMonoidSubtypeLtOfNat___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_PNat_instCommMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_PNat_instCommMonoid___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_PNat_instCancelCommMonoid() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_PNat_instCommMonoid;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeAddHom___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeAddHom___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_PNat_coeAddHom___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_PNat_coeAddHom() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PNat_coeAddHom___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_OrderIso_pnatIsoNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_pnatEquivNat;
return x_1;
}
}
static lean_object* _init_lp_mathlib_PNat_instOrderBot() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(1u);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, lean_box(0));
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_eq(x_3, x_5);
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_3, x_7);
x_9 = lean_nat_dec_eq(x_8, x_5);
if (x_9 == 1)
{
lean_dec(x_8);
lean_dec(x_4);
lean_dec(x_2);
lean_inc(x_1);
return x_1;
}
else
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_4);
x_11 = lean_apply_2(x_2, x_8, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1___boxed), 4, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
x_5 = lp_mathlib_PNat_strongInductionOn___redArg(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PNat_caseStrongInductionOn___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_nat_dec_eq(x_3, x_6);
if (x_7 == 1)
{
lean_dec(x_4);
lean_dec(x_2);
lean_inc(x_1);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_unsigned_to_nat(1u);
x_9 = lean_nat_sub(x_3, x_8);
x_10 = lean_nat_add(x_9, x_8);
lean_dec(x_9);
x_11 = lean_apply_1(x_4, lean_box(0));
x_12 = lean_apply_2(x_2, x_10, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_PNat_recOn___redArg___lam__1(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_3);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_PNat_recOn___redArg___lam__0), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_PNat_recOn___redArg___lam__1___boxed), 5, 2);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_3);
x_6 = l_Nat_recCompiled___redArg(x_4, x_5, x_1);
lean_dec_ref(x_4);
x_7 = lean_apply_1(x_6, lean_box(0));
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PNat_recOn___redArg(x_1, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PNat_recOn(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_PNat_recOn___redArg(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_PNat_coeMonoidHom___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PNat_coeAddHom___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_PNat_coeMonoidHom() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_PNat_coeMonoidHom___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_instSub___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_nat_sub(x_1, x_2);
x_4 = lp_mathlib_Nat_toPNat_x27(x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_instSub___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PNat_instSub___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_PNat_instSub() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PNat_instSub___lam__0___boxed), 2, 0);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_PNat_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_PNat_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_PNat_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instAddLeftCancelSemigroupPNat___closed__0 = _init_lp_mathlib_instAddLeftCancelSemigroupPNat___closed__0();
lean_mark_persistent(lp_mathlib_instAddLeftCancelSemigroupPNat___closed__0);
lp_mathlib_instAddLeftCancelSemigroupPNat = _init_lp_mathlib_instAddLeftCancelSemigroupPNat();
lean_mark_persistent(lp_mathlib_instAddLeftCancelSemigroupPNat);
lp_mathlib_instAddRightCancelSemigroupPNat = _init_lp_mathlib_instAddRightCancelSemigroupPNat();
lean_mark_persistent(lp_mathlib_instAddRightCancelSemigroupPNat);
lp_mathlib_instAddCommSemigroupPNat = _init_lp_mathlib_instAddCommSemigroupPNat();
lean_mark_persistent(lp_mathlib_instAddCommSemigroupPNat);
lp_mathlib_instAddPNat___closed__0 = _init_lp_mathlib_instAddPNat___closed__0();
lean_mark_persistent(lp_mathlib_instAddPNat___closed__0);
lp_mathlib_instAddPNat = _init_lp_mathlib_instAddPNat();
lean_mark_persistent(lp_mathlib_instAddPNat);
lp_mathlib_instMulPNat___closed__0 = _init_lp_mathlib_instMulPNat___closed__0();
lean_mark_persistent(lp_mathlib_instMulPNat___closed__0);
lp_mathlib_instMulPNat = _init_lp_mathlib_instMulPNat();
lean_mark_persistent(lp_mathlib_instMulPNat);
lp_mathlib_instDistribPNat___closed__0 = _init_lp_mathlib_instDistribPNat___closed__0();
lean_mark_persistent(lp_mathlib_instDistribPNat___closed__0);
lp_mathlib_instDistribPNat = _init_lp_mathlib_instDistribPNat();
lean_mark_persistent(lp_mathlib_instDistribPNat);
lp_mathlib_PNat_instCommMonoid___closed__0 = _init_lp_mathlib_PNat_instCommMonoid___closed__0();
lean_mark_persistent(lp_mathlib_PNat_instCommMonoid___closed__0);
lp_mathlib_PNat_instCommMonoid = _init_lp_mathlib_PNat_instCommMonoid();
lean_mark_persistent(lp_mathlib_PNat_instCommMonoid);
lp_mathlib_PNat_instCancelCommMonoid = _init_lp_mathlib_PNat_instCancelCommMonoid();
lean_mark_persistent(lp_mathlib_PNat_instCancelCommMonoid);
lp_mathlib_PNat_coeAddHom = _init_lp_mathlib_PNat_coeAddHom();
lean_mark_persistent(lp_mathlib_PNat_coeAddHom);
lp_mathlib_OrderIso_pnatIsoNat = _init_lp_mathlib_OrderIso_pnatIsoNat();
lean_mark_persistent(lp_mathlib_OrderIso_pnatIsoNat);
lp_mathlib_PNat_instOrderBot = _init_lp_mathlib_PNat_instOrderBot();
lean_mark_persistent(lp_mathlib_PNat_instOrderBot);
lp_mathlib_PNat_coeMonoidHom___closed__0 = _init_lp_mathlib_PNat_coeMonoidHom___closed__0();
lean_mark_persistent(lp_mathlib_PNat_coeMonoidHom___closed__0);
lp_mathlib_PNat_coeMonoidHom = _init_lp_mathlib_PNat_coeMonoidHom();
lean_mark_persistent(lp_mathlib_PNat_coeMonoidHom);
lp_mathlib_PNat_instSub = _init_lp_mathlib_PNat_instSub();
lean_mark_persistent(lp_mathlib_PNat_instSub);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
