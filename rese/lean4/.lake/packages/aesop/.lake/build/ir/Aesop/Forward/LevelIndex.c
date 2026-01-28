// Lean compiler output
// Module: Aesop.Forward.LevelIndex
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelLevelIndexLt___boxed(lean_object*, lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdLevelIndex_ord___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqLevelIndex_beq(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instBEqLevelIndex___closed__0;
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqLevelIndex___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instToStringLevelIndex___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringLevelIndex;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelLevelIndexLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelLevelIndexLe___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLELevelIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdLevelIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqLevelIndex_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLTLevelIndex;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqLevelIndex_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqLevelIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableLevelIndex;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableLevelIndex_hash___boxed(lean_object*);
static lean_object* lp_aesop_Aesop_instOrdLevelIndex___closed__0;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableLevelIndex_hash(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdLevelIndex_ord(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedLevelIndex;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqLevelIndex(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqLevelIndex_beq___boxed(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelLevelIndexLe(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instHashableLevelIndex___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedLevelIndex_default;
static lean_object* _init_lp_aesop_Aesop_instInhabitedLevelIndex_default() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedLevelIndex() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqLevelIndex_beq(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqLevelIndex_beq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instBEqLevelIndex_beq(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqLevelIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instBEqLevelIndex_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqLevelIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instBEqLevelIndex___closed__0;
return x_1;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableLevelIndex_hash(lean_object* x_1) {
_start:
{
uint64_t x_2; uint64_t x_3; uint64_t x_4; 
x_2 = 0;
x_3 = lean_uint64_of_nat(x_1);
x_4 = lean_uint64_mix_hash(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableLevelIndex_hash___boxed(lean_object* x_1) {
_start:
{
uint64_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_instHashableLevelIndex_hash(x_1);
lean_dec(x_1);
x_3 = lean_box_uint64(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instHashableLevelIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instHashableLevelIndex_hash___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instHashableLevelIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instHashableLevelIndex___closed__0;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqLevelIndex_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqLevelIndex_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqLevelIndex_decEq(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqLevelIndex(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqLevelIndex___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqLevelIndex(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdLevelIndex_ord(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdLevelIndex_ord___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instOrdLevelIndex_ord(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instOrdLevelIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instOrdLevelIndex_ord___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instOrdLevelIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instOrdLevelIndex___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instLTLevelIndex() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelLevelIndexLt(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_lt(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelLevelIndexLt___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableRelLevelIndexLt(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instLELevelIndex() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelLevelIndexLe(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_le(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelLevelIndexLe___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableRelLevelIndexLe(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_instToStringLevelIndex___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_reprFast), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToStringLevelIndex() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instToStringLevelIndex___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_LevelIndex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedLevelIndex_default = _init_lp_aesop_Aesop_instInhabitedLevelIndex_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedLevelIndex_default);
lp_aesop_Aesop_instInhabitedLevelIndex = _init_lp_aesop_Aesop_instInhabitedLevelIndex();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedLevelIndex);
lp_aesop_Aesop_instBEqLevelIndex___closed__0 = _init_lp_aesop_Aesop_instBEqLevelIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instBEqLevelIndex___closed__0);
lp_aesop_Aesop_instBEqLevelIndex = _init_lp_aesop_Aesop_instBEqLevelIndex();
lean_mark_persistent(lp_aesop_Aesop_instBEqLevelIndex);
lp_aesop_Aesop_instHashableLevelIndex___closed__0 = _init_lp_aesop_Aesop_instHashableLevelIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instHashableLevelIndex___closed__0);
lp_aesop_Aesop_instHashableLevelIndex = _init_lp_aesop_Aesop_instHashableLevelIndex();
lean_mark_persistent(lp_aesop_Aesop_instHashableLevelIndex);
lp_aesop_Aesop_instOrdLevelIndex___closed__0 = _init_lp_aesop_Aesop_instOrdLevelIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instOrdLevelIndex___closed__0);
lp_aesop_Aesop_instOrdLevelIndex = _init_lp_aesop_Aesop_instOrdLevelIndex();
lean_mark_persistent(lp_aesop_Aesop_instOrdLevelIndex);
lp_aesop_Aesop_instLTLevelIndex = _init_lp_aesop_Aesop_instLTLevelIndex();
lean_mark_persistent(lp_aesop_Aesop_instLTLevelIndex);
lp_aesop_Aesop_instLELevelIndex = _init_lp_aesop_Aesop_instLELevelIndex();
lean_mark_persistent(lp_aesop_Aesop_instLELevelIndex);
lp_aesop_Aesop_instToStringLevelIndex___closed__0 = _init_lp_aesop_Aesop_instToStringLevelIndex___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instToStringLevelIndex___closed__0);
lp_aesop_Aesop_instToStringLevelIndex = _init_lp_aesop_Aesop_instToStringLevelIndex();
lean_mark_persistent(lp_aesop_Aesop_instToStringLevelIndex);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
