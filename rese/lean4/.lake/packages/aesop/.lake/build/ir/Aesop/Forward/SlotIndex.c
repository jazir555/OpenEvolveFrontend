// Lean compiler output
// Module: Aesop.Forward.SlotIndex
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdSlotIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableSlotIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelSlotIndexLe___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instHSubSlotIndexNat___closed__0;
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
static lean_object* lp_aesop_Aesop_instToStringSlotIndex___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableSlotIndex_hash___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelSlotIndexLt(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instHashableSlotIndex___closed__0;
static lean_object* lp_aesop_Aesop_instHAddSlotIndexNat___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringSlotIndex;
lean_object* l_Nat_sub___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqSlotIndex_beq___boxed(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableSlotIndex_hash(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedSlotIndex;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdSlotIndex_ord(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelSlotIndexLe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqSlotIndex_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqSlotIndex(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedSlotIndex_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLTSlotIndex;
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLESlotIndex;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqSlotIndex_beq(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instBEqSlotIndex___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelSlotIndexLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHAddSlotIndexNat;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqSlotIndex_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdSlotIndex_ord___boxed(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqSlotIndex___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHSubSlotIndexNat;
static lean_object* lp_aesop_Aesop_instOrdSlotIndex___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqSlotIndex;
static lean_object* _init_lp_aesop_Aesop_instInhabitedSlotIndex_default() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedSlotIndex() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqSlotIndex_beq(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqSlotIndex_beq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instBEqSlotIndex_beq(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqSlotIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instBEqSlotIndex_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqSlotIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instBEqSlotIndex___closed__0;
return x_1;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableSlotIndex_hash(lean_object* x_1) {
_start:
{
uint64_t x_2; uint64_t x_3; uint64_t x_4; 
x_2 = 0;
x_3 = lean_uint64_of_nat(x_1);
x_4 = lean_uint64_mix_hash(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableSlotIndex_hash___boxed(lean_object* x_1) {
_start:
{
uint64_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_instHashableSlotIndex_hash(x_1);
lean_dec(x_1);
x_3 = lean_box_uint64(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instHashableSlotIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instHashableSlotIndex_hash___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instHashableSlotIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instHashableSlotIndex___closed__0;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqSlotIndex_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqSlotIndex_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqSlotIndex_decEq(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqSlotIndex(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqSlotIndex___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqSlotIndex(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdSlotIndex_ord(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdSlotIndex_ord___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instOrdSlotIndex_ord(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instOrdSlotIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instOrdSlotIndex_ord___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instOrdSlotIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instOrdSlotIndex___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instLTSlotIndex() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelSlotIndexLt(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_lt(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelSlotIndexLt___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableRelSlotIndexLt(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instLESlotIndex() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelSlotIndexLe(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_le(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelSlotIndexLe___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableRelSlotIndexLe(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instHAddSlotIndexNat___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_add___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instHAddSlotIndexNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instHAddSlotIndexNat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instHSubSlotIndexNat___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_sub___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instHSubSlotIndexNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instHSubSlotIndexNat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToStringSlotIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_reprFast), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToStringSlotIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instToStringSlotIndex___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_SlotIndex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedSlotIndex_default = _init_lp_aesop_Aesop_instInhabitedSlotIndex_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedSlotIndex_default);
lp_aesop_Aesop_instInhabitedSlotIndex = _init_lp_aesop_Aesop_instInhabitedSlotIndex();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedSlotIndex);
lp_aesop_Aesop_instBEqSlotIndex___closed__0 = _init_lp_aesop_Aesop_instBEqSlotIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instBEqSlotIndex___closed__0);
lp_aesop_Aesop_instBEqSlotIndex = _init_lp_aesop_Aesop_instBEqSlotIndex();
lean_mark_persistent(lp_aesop_Aesop_instBEqSlotIndex);
lp_aesop_Aesop_instHashableSlotIndex___closed__0 = _init_lp_aesop_Aesop_instHashableSlotIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instHashableSlotIndex___closed__0);
lp_aesop_Aesop_instHashableSlotIndex = _init_lp_aesop_Aesop_instHashableSlotIndex();
lean_mark_persistent(lp_aesop_Aesop_instHashableSlotIndex);
lp_aesop_Aesop_instOrdSlotIndex___closed__0 = _init_lp_aesop_Aesop_instOrdSlotIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instOrdSlotIndex___closed__0);
lp_aesop_Aesop_instOrdSlotIndex = _init_lp_aesop_Aesop_instOrdSlotIndex();
lean_mark_persistent(lp_aesop_Aesop_instOrdSlotIndex);
lp_aesop_Aesop_instLTSlotIndex = _init_lp_aesop_Aesop_instLTSlotIndex();
lean_mark_persistent(lp_aesop_Aesop_instLTSlotIndex);
lp_aesop_Aesop_instLESlotIndex = _init_lp_aesop_Aesop_instLESlotIndex();
lean_mark_persistent(lp_aesop_Aesop_instLESlotIndex);
lp_aesop_Aesop_instHAddSlotIndexNat___closed__0 = _init_lp_aesop_Aesop_instHAddSlotIndexNat___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instHAddSlotIndexNat___closed__0);
lp_aesop_Aesop_instHAddSlotIndexNat = _init_lp_aesop_Aesop_instHAddSlotIndexNat();
lean_mark_persistent(lp_aesop_Aesop_instHAddSlotIndexNat);
lp_aesop_Aesop_instHSubSlotIndexNat___closed__0 = _init_lp_aesop_Aesop_instHSubSlotIndexNat___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instHSubSlotIndexNat___closed__0);
lp_aesop_Aesop_instHSubSlotIndexNat = _init_lp_aesop_Aesop_instHSubSlotIndexNat();
lean_mark_persistent(lp_aesop_Aesop_instHSubSlotIndexNat);
lp_aesop_Aesop_instToStringSlotIndex___closed__0 = _init_lp_aesop_Aesop_instToStringSlotIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instToStringSlotIndex___closed__0);
lp_aesop_Aesop_instToStringSlotIndex = _init_lp_aesop_Aesop_instToStringSlotIndex();
lean_mark_persistent(lp_aesop_Aesop_instToStringSlotIndex);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
