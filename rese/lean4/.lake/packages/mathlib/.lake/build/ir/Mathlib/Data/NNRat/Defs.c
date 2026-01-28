// Lean compiler output
// Module: Mathlib.Data.NNRat.Defs
// Imports: public import Init public import Mathlib.Algebra.Order.Group.Unbundled.Int public import Mathlib.Algebra.Order.Nonneg.Basic public import Mathlib.Algebra.Order.Ring.Unbundled.Rat public import Mathlib.Algebra.Ring.Rat public import Mathlib.Data.Set.Operations public import Mathlib.Order.Bounds.Defs public import Mathlib.Order.GaloisConnection.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_NNRat_divNat(lean_object*, lean_object*);
lean_object* lp_mathlib_Nonneg_sub___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Rat_instDecidableLe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_specialised_x20high_x20priority_x20simp_x20lemma;
static lean_object* lp_mathlib_NNRat_instOrderBot___closed__0;
static lean_object* lp_mathlib_NNRat_gi___closed__0;
lean_object* lp_mathlib_Subtype_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedNNRat;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00instInhabitedNNRat_spec__0_spec__0(lean_object*);
lean_object* lp_mathlib_Nonneg_addCancelCommMonoid___redArg(lean_object*);
lean_object* l_Rat_divInt(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Rat_instSemilatticeSup;
extern lean_object* lp_mathlib_Rat_linearOrder;
lean_object* l_Rat_instNatCast___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_coeHom;
extern lean_object* lp_mathlib_Rat_commSemiring;
static lean_object* lp_mathlib_instSubNNRat___closed__2;
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
static lean_object* lp_mathlib_Rat_toNNRat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_numDenCasesOn(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0(lean_object*);
static lean_object* lp_mathlib_instCommSemiringNNRat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_num(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_numDenCasesOn___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_instSubNNRat___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_coeHom___lam__0___boxed(lean_object*);
static lean_object* lp_mathlib_instLinearOrderNNRat___closed__0;
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instOrderBot;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_abs___at___00Rat_nnabs_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat;
lean_object* l_Rat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Nonneg_commSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_coeHom___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_nnabs(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_toNNRat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCancelCommMonoidNNRat;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00instInhabitedNNRat_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg(lean_object*);
static lean_object* lp_mathlib_instSubNNRat___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSubNNRat;
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_gi;
lean_object* l_Rat_ofInt(lean_object*);
extern lean_object* lp_mathlib_Rat_commRing;
static lean_object* lp_mathlib_instSubNNRat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat;
extern lean_object* lp_mathlib_Rat_addCommGroup;
lean_object* l_Rat_neg(lean_object*);
static lean_object* _init_lp_mathlib_LibraryNote_specialised_x20high_x20priority_x20simp_x20lemma() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_commSemiring;
x_2 = lp_mathlib_Nonneg_commSemiring___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instCommSemiringNNRat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_instAddCancelCommMonoidNNRat() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Rat_addCommGroup;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_Nonneg_addCancelCommMonoid___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderNNRat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_linearOrder;
x_2 = lp_mathlib_Subtype_instLinearOrder___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderNNRat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instLinearOrderNNRat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_instSubNNRat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_commRing;
x_2 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instSubNNRat___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_instSubNNRat___closed__0;
x_2 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instSubNNRat___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_instSubNNRat___closed__1;
x_2 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instSubNNRat___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_sub), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instSubNNRat() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_instSubNNRat___closed__2;
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
x_3 = lp_mathlib_Rat_instSemilatticeSup;
x_4 = lp_mathlib_instSubNNRat___closed__3;
x_5 = lp_mathlib_Nonneg_sub___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00instInhabitedNNRat_spec__0_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00instInhabitedNNRat_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_nat_to_int(x_1);
x_3 = l_Rat_ofInt(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instInhabitedNNRat() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_mathlib_Nat_cast___at___00instInhabitedNNRat_spec__0(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_NNRat_instOrderBot___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = l_Rat_instNatCast___lam__0(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_NNRat_instOrderBot() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_NNRat_instOrderBot___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_toNNRat___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_mathlib_Nat_cast___at___00instInhabitedNNRat_spec__0(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_toNNRat(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lp_mathlib_Rat_toNNRat___closed__0;
lean_inc_ref(x_1);
x_3 = l_Rat_instDecidableLe(x_1, x_2);
if (x_3 == 0)
{
return x_1;
}
else
{
lean_dec_ref(x_1);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg(x_1);
return x_7;
}
}
static lean_object* _init_lp_mathlib_NNRat_gi___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Rat_toNNRat), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_NNRat_gi() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_NNRat_gi___closed__0;
x_2 = lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_coeHom___lam__0(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_coeHom___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NNRat_coeHom___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_NNRat_coeHom() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NNRat_coeHom___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_abs___at___00Rat_nnabs_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
lean_inc_ref(x_1);
x_2 = l_Rat_neg(x_1);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_3 = l_Rat_instDecidableLe(x_1, x_2);
if (x_3 == 0)
{
lean_dec_ref(x_2);
return x_1;
}
else
{
lean_dec_ref(x_1);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_nnabs(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_abs___at___00Rat_nnabs_spec__0(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_divNat(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_nat_to_int(x_1);
x_4 = lean_nat_to_int(x_2);
x_5 = l_Rat_divInt(x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_numDenCasesOn___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lp_mathlib_NNRat_num(x_1);
lean_dec_ref(x_1);
x_5 = lean_apply_4(x_2, x_4, x_3, lean_box(0), lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_numDenCasesOn(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NNRat_numDenCasesOn___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_NNRat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_specialised_x20high_x20priority_x20simp_x20lemma = _init_lp_mathlib_LibraryNote_specialised_x20high_x20priority_x20simp_x20lemma();
lean_mark_persistent(lp_mathlib_LibraryNote_specialised_x20high_x20priority_x20simp_x20lemma);
lp_mathlib_instCommSemiringNNRat___closed__0 = _init_lp_mathlib_instCommSemiringNNRat___closed__0();
lean_mark_persistent(lp_mathlib_instCommSemiringNNRat___closed__0);
lp_mathlib_instCommSemiringNNRat = _init_lp_mathlib_instCommSemiringNNRat();
lean_mark_persistent(lp_mathlib_instCommSemiringNNRat);
lp_mathlib_instAddCancelCommMonoidNNRat = _init_lp_mathlib_instAddCancelCommMonoidNNRat();
lean_mark_persistent(lp_mathlib_instAddCancelCommMonoidNNRat);
lp_mathlib_instLinearOrderNNRat___closed__0 = _init_lp_mathlib_instLinearOrderNNRat___closed__0();
lean_mark_persistent(lp_mathlib_instLinearOrderNNRat___closed__0);
lp_mathlib_instLinearOrderNNRat = _init_lp_mathlib_instLinearOrderNNRat();
lean_mark_persistent(lp_mathlib_instLinearOrderNNRat);
lp_mathlib_instSubNNRat___closed__0 = _init_lp_mathlib_instSubNNRat___closed__0();
lean_mark_persistent(lp_mathlib_instSubNNRat___closed__0);
lp_mathlib_instSubNNRat___closed__1 = _init_lp_mathlib_instSubNNRat___closed__1();
lean_mark_persistent(lp_mathlib_instSubNNRat___closed__1);
lp_mathlib_instSubNNRat___closed__2 = _init_lp_mathlib_instSubNNRat___closed__2();
lean_mark_persistent(lp_mathlib_instSubNNRat___closed__2);
lp_mathlib_instSubNNRat___closed__3 = _init_lp_mathlib_instSubNNRat___closed__3();
lean_mark_persistent(lp_mathlib_instSubNNRat___closed__3);
lp_mathlib_instSubNNRat = _init_lp_mathlib_instSubNNRat();
lean_mark_persistent(lp_mathlib_instSubNNRat);
lp_mathlib_instInhabitedNNRat = _init_lp_mathlib_instInhabitedNNRat();
lean_mark_persistent(lp_mathlib_instInhabitedNNRat);
lp_mathlib_NNRat_instOrderBot___closed__0 = _init_lp_mathlib_NNRat_instOrderBot___closed__0();
lean_mark_persistent(lp_mathlib_NNRat_instOrderBot___closed__0);
lp_mathlib_NNRat_instOrderBot = _init_lp_mathlib_NNRat_instOrderBot();
lean_mark_persistent(lp_mathlib_NNRat_instOrderBot);
lp_mathlib_Rat_toNNRat___closed__0 = _init_lp_mathlib_Rat_toNNRat___closed__0();
lean_mark_persistent(lp_mathlib_Rat_toNNRat___closed__0);
lp_mathlib_NNRat_gi___closed__0 = _init_lp_mathlib_NNRat_gi___closed__0();
lean_mark_persistent(lp_mathlib_NNRat_gi___closed__0);
lp_mathlib_NNRat_gi = _init_lp_mathlib_NNRat_gi();
lean_mark_persistent(lp_mathlib_NNRat_gi);
lp_mathlib_NNRat_coeHom = _init_lp_mathlib_NNRat_coeHom();
lean_mark_persistent(lp_mathlib_NNRat_coeHom);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
