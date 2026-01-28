// Lean compiler output
// Module: Aesop.Forward.PremiseIndex
// Imports: public import Init
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqPremiseIndex___boxed(lean_object*, lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqPremiseIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelPremiseIndexLe___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelPremiseIndexLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqPremiseIndex_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdPremiseIndex_ord(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instToStringPremiseIndex___closed__0;
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqPremiseIndex(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instBEqPremiseIndex___closed__0;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelPremiseIndexLe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringPremiseIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashablePremiseIndex;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLTPremiseIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedPremiseIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelPremiseIndexLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedPremiseIndex_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqPremiseIndex_beq(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instOrdPremiseIndex___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLEPremiseIndex;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdPremiseIndex_ord___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashablePremiseIndex_hash(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdPremiseIndex;
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashablePremiseIndex_hash___boxed(lean_object*);
static lean_object* lp_aesop_Aesop_instHashablePremiseIndex___closed__0;
static lean_object* _init_lp_aesop_Aesop_instInhabitedPremiseIndex_default() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedPremiseIndex() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqPremiseIndex_beq(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqPremiseIndex_beq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instBEqPremiseIndex_beq(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqPremiseIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instBEqPremiseIndex_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqPremiseIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instBEqPremiseIndex___closed__0;
return x_1;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashablePremiseIndex_hash(lean_object* x_1) {
_start:
{
uint64_t x_2; uint64_t x_3; uint64_t x_4; 
x_2 = 0;
x_3 = lean_uint64_of_nat(x_1);
x_4 = lean_uint64_mix_hash(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashablePremiseIndex_hash___boxed(lean_object* x_1) {
_start:
{
uint64_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_instHashablePremiseIndex_hash(x_1);
lean_dec(x_1);
x_3 = lean_box_uint64(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instHashablePremiseIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instHashablePremiseIndex_hash___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instHashablePremiseIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instHashablePremiseIndex___closed__0;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqPremiseIndex(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqPremiseIndex___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqPremiseIndex(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdPremiseIndex_ord(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_lt(x_1, x_2);
if (x_3 == 0)
{
uint8_t x_4; 
x_4 = lean_nat_dec_eq(x_1, x_2);
if (x_4 == 0)
{
uint8_t x_5; 
x_5 = 2;
return x_5;
}
else
{
uint8_t x_6; 
x_6 = 1;
return x_6;
}
}
else
{
uint8_t x_7; 
x_7 = 0;
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdPremiseIndex_ord___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instOrdPremiseIndex_ord(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instOrdPremiseIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instOrdPremiseIndex_ord___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instOrdPremiseIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instOrdPremiseIndex___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instLTPremiseIndex() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelPremiseIndexLt(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_lt(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelPremiseIndexLt___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableRelPremiseIndexLt(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instLEPremiseIndex() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelPremiseIndexLe(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_le(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelPremiseIndexLe___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableRelPremiseIndexLe(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instToStringPremiseIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_reprFast), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToStringPremiseIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instToStringPremiseIndex___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_PremiseIndex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedPremiseIndex_default = _init_lp_aesop_Aesop_instInhabitedPremiseIndex_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedPremiseIndex_default);
lp_aesop_Aesop_instInhabitedPremiseIndex = _init_lp_aesop_Aesop_instInhabitedPremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedPremiseIndex);
lp_aesop_Aesop_instBEqPremiseIndex___closed__0 = _init_lp_aesop_Aesop_instBEqPremiseIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instBEqPremiseIndex___closed__0);
lp_aesop_Aesop_instBEqPremiseIndex = _init_lp_aesop_Aesop_instBEqPremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instBEqPremiseIndex);
lp_aesop_Aesop_instHashablePremiseIndex___closed__0 = _init_lp_aesop_Aesop_instHashablePremiseIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instHashablePremiseIndex___closed__0);
lp_aesop_Aesop_instHashablePremiseIndex = _init_lp_aesop_Aesop_instHashablePremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instHashablePremiseIndex);
lp_aesop_Aesop_instOrdPremiseIndex___closed__0 = _init_lp_aesop_Aesop_instOrdPremiseIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instOrdPremiseIndex___closed__0);
lp_aesop_Aesop_instOrdPremiseIndex = _init_lp_aesop_Aesop_instOrdPremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instOrdPremiseIndex);
lp_aesop_Aesop_instLTPremiseIndex = _init_lp_aesop_Aesop_instLTPremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instLTPremiseIndex);
lp_aesop_Aesop_instLEPremiseIndex = _init_lp_aesop_Aesop_instLEPremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instLEPremiseIndex);
lp_aesop_Aesop_instToStringPremiseIndex___closed__0 = _init_lp_aesop_Aesop_instToStringPremiseIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instToStringPremiseIndex___closed__0);
lp_aesop_Aesop_instToStringPremiseIndex = _init_lp_aesop_Aesop_instToStringPremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instToStringPremiseIndex);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
