// Lean compiler output
// Module: Mathlib.Data.Set.Basic
// Imports: public import Init public import Mathlib.Order.PropInstances public import Mathlib.Tactic.Lift public import Mathlib.Tactic.Tauto public import Mathlib.Util.Delaborators
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
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableUniv(lean_object*, lean_object*);
static lean_object* lp_mathlib_Set_instBoundedOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Set_instTransSSubsetSubset(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableInter(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableUniv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableInter___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSetOf___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSdiff___redArg(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCElem___lam__0(lean_object*);
static lean_object* lp_mathlib_Set_instDistribLattice___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setSubtypeComm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSdiff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instHasSSubset(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableUnion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableUnion___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setSubtypeComm(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableUnion___redArg(uint8_t, uint8_t);
static lean_object* lp_mathlib_Set_instDistribLattice___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Set_instDistribLattice(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableCompl___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_instTransSubset(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableCompl___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableCompl(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSdiff___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSetOf(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSetOf___redArg(uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableEmptyset(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCElem(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instBoundedOrder(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableInter___redArg(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_instTransSubsetSSubset(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableInsert___redArg(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableEmptyset___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSetOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instTransSSubset(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableInsert___redArg___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Set_instDistribLattice___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableInter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableInsert___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableInsert(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCElem___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableUnion(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSdiff(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableCompl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Set_instDistribLattice___closed__0() {
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
static lean_object* _init_lp_mathlib_Set_instDistribLattice___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Set_instDistribLattice___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, lean_box(0));
return x_2;
}
}
static lean_object* _init_lp_mathlib_Set_instDistribLattice___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Set_instDistribLattice___closed__1;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, lean_box(0));
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instDistribLattice(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_instDistribLattice___closed__2;
return x_2;
}
}
static lean_object* _init_lp_mathlib_Set_instBoundedOrder___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_1, 0, lean_box(0));
lean_ctor_set(x_1, 1, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instBoundedOrder(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_instBoundedOrder___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instHasSSubset(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCElem___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCElem___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_instCoeTCElem___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCElem(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_instCoeTCElem___lam__0___boxed), 1, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instInhabited(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instTransSubset(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instTransSSubset(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instTransSSubsetSubset(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instTransSubsetSSubset(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueEmpty(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSdiff___redArg(uint8_t x_1, uint8_t x_2) {
_start:
{
if (x_1 == 0)
{
return x_1;
}
else
{
if (x_2 == 0)
{
return x_1;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSdiff(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, uint8_t x_6) {
_start:
{
uint8_t x_7; 
x_7 = lp_mathlib_Set_decidableSdiff___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSdiff___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; uint8_t x_8; uint8_t x_9; lean_object* x_10; 
x_7 = lean_unbox(x_5);
x_8 = lean_unbox(x_6);
x_9 = lp_mathlib_Set_decidableSdiff(x_1, x_2, x_3, x_4, x_7, x_8);
lean_dec(x_4);
x_10 = lean_box(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSdiff___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_mathlib_Set_decidableSdiff___redArg(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableInter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, uint8_t x_6) {
_start:
{
if (x_5 == 0)
{
return x_5;
}
else
{
return x_6;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableInter___redArg(uint8_t x_1, uint8_t x_2) {
_start:
{
if (x_1 == 0)
{
return x_1;
}
else
{
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableInter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; uint8_t x_8; uint8_t x_9; lean_object* x_10; 
x_7 = lean_unbox(x_5);
x_8 = lean_unbox(x_6);
x_9 = lp_mathlib_Set_decidableInter(x_1, x_2, x_3, x_4, x_7, x_8);
lean_dec(x_4);
x_10 = lean_box(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableInter___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_mathlib_Set_decidableInter___redArg(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableUnion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, uint8_t x_6) {
_start:
{
if (x_5 == 0)
{
return x_6;
}
else
{
return x_5;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableUnion___redArg(uint8_t x_1, uint8_t x_2) {
_start:
{
if (x_1 == 0)
{
return x_2;
}
else
{
return x_1;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableUnion___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; uint8_t x_8; uint8_t x_9; lean_object* x_10; 
x_7 = lean_unbox(x_5);
x_8 = lean_unbox(x_6);
x_9 = lp_mathlib_Set_decidableUnion(x_1, x_2, x_3, x_4, x_7, x_8);
lean_dec(x_4);
x_10 = lean_box(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableUnion___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_mathlib_Set_decidableUnion___redArg(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableCompl___redArg(uint8_t x_1) {
_start:
{
if (x_1 == 0)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableCompl(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4) {
_start:
{
uint8_t x_5; 
x_5 = lp_mathlib_Set_decidableCompl___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableCompl___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; uint8_t x_6; lean_object* x_7; 
x_5 = lean_unbox(x_4);
x_6 = lp_mathlib_Set_decidableCompl(x_1, x_2, x_3, x_5);
lean_dec(x_3);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableCompl___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_Set_decidableCompl___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableEmptyset(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableEmptyset___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Set_decidableEmptyset(x_1, x_2);
lean_dec(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableUniv(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = 1;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableUniv___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Set_decidableUniv(x_1, x_2);
lean_dec(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableInsert(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, uint8_t x_6) {
_start:
{
if (x_5 == 0)
{
return x_6;
}
else
{
return x_5;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableInsert___redArg(uint8_t x_1, uint8_t x_2) {
_start:
{
if (x_1 == 0)
{
return x_2;
}
else
{
return x_1;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableInsert___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; uint8_t x_8; uint8_t x_9; lean_object* x_10; 
x_7 = lean_unbox(x_5);
x_8 = lean_unbox(x_6);
x_9 = lp_mathlib_Set_decidableInsert(x_1, x_2, x_3, x_4, x_7, x_8);
lean_dec(x_4);
lean_dec(x_3);
x_10 = lean_box(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableInsert___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_mathlib_Set_decidableInsert___redArg(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSetOf(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4) {
_start:
{
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSetOf___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSetOf___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; uint8_t x_6; lean_object* x_7; 
x_5 = lean_unbox(x_4);
x_6 = lp_mathlib_Set_decidableSetOf(x_1, x_2, x_3, x_5);
lean_dec(x_2);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSetOf___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_Set_decidableSetOf___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setSubtypeComm(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, lean_box(0));
lean_ctor_set(x_3, 1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setSubtypeComm___lam__0(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_PropInstances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_PropInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Set_instDistribLattice___closed__0 = _init_lp_mathlib_Set_instDistribLattice___closed__0();
lean_mark_persistent(lp_mathlib_Set_instDistribLattice___closed__0);
lp_mathlib_Set_instDistribLattice___closed__1 = _init_lp_mathlib_Set_instDistribLattice___closed__1();
lean_mark_persistent(lp_mathlib_Set_instDistribLattice___closed__1);
lp_mathlib_Set_instDistribLattice___closed__2 = _init_lp_mathlib_Set_instDistribLattice___closed__2();
lean_mark_persistent(lp_mathlib_Set_instDistribLattice___closed__2);
lp_mathlib_Set_instBoundedOrder___closed__0 = _init_lp_mathlib_Set_instBoundedOrder___closed__0();
lean_mark_persistent(lp_mathlib_Set_instBoundedOrder___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
