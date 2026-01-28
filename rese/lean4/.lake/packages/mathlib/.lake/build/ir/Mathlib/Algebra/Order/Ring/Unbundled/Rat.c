// Lean compiler output
// Module: Mathlib.Algebra.Order.Ring.Unbundled.Rat
// Imports: public import Init public import Mathlib.Algebra.Order.Group.Unbundled.Abs public import Mathlib.Algebra.Order.Group.Unbundled.Basic public import Mathlib.Algebra.Order.Group.Unbundled.Int public import Mathlib.Data.Rat.Defs public import Mathlib.Algebra.Ring.Int.Defs
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
lean_object* l_instDecidableEqRat___boxed(lean_object*, lean_object*);
uint8_t l_instDecidableEqRat_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific(lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_linearOrder___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instInf;
lean_object* l_Rat_instDecidableLe___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_linearOrder___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instDistribLattice;
lean_object* l_Rat_ofScientific(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instSemilatticeSup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_linearOrder;
lean_object* l_Rat_instDecidableLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instSemilatticeInf;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instLattice;
uint8_t l_Rat_blt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*);
static lean_object* lp_mathlib_Rat_linearOrder___closed__0;
static lean_object* lp_mathlib_Rat_linearOrder___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instPreorder;
static lean_object* lp_mathlib_Rat_linearOrder___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instPartialOrder;
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_linearOrder___closed__2;
lean_object* l_Rat_instMin___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_instDistribLattice___closed__0;
lean_object* l_Rat_instMax___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Rat_linearOrder___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Rat_instSemilatticeInf___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instSup;
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = l_Rat_ofScientific(x_2, x_3, x_4);
x_6 = lean_apply_1(x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_3);
x_6 = lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0(x_1, x_2, x_5, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NNRatCast_toOfScientific___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Rat_linearOrder___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_instMin___lam__0), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_linearOrder___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_instMax___lam__0), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_linearOrder___closed__2() {
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
static lean_object* _init_lp_mathlib_Rat_linearOrder___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_instDecidableLe___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_linearOrder___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Rat_instDecidableLt___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Rat_linearOrder___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_3 = l_Rat_blt(x_1, x_2);
if (x_3 == 0)
{
uint8_t x_4; 
x_4 = l_instDecidableEqRat_decEq(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
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
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_7 = 0;
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_linearOrder___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Rat_linearOrder___lam__0(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Rat_linearOrder() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Rat_linearOrder___lam__0___boxed), 2, 0);
x_2 = lp_mathlib_Rat_linearOrder___closed__0;
x_3 = lp_mathlib_Rat_linearOrder___closed__1;
x_4 = lp_mathlib_Rat_linearOrder___closed__2;
x_5 = lp_mathlib_Rat_linearOrder___closed__3;
x_6 = lean_alloc_closure((void*)(l_instDecidableEqRat___boxed), 2, 0);
x_7 = lp_mathlib_Rat_linearOrder___closed__4;
x_8 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_8, 0, x_4);
lean_ctor_set(x_8, 1, x_2);
lean_ctor_set(x_8, 2, x_3);
lean_ctor_set(x_8, 3, x_1);
lean_ctor_set(x_8, 4, x_5);
lean_ctor_set(x_8, 5, x_6);
lean_ctor_set(x_8, 6, x_7);
return x_8;
}
}
static lean_object* _init_lp_mathlib_Rat_instDistribLattice___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_linearOrder;
x_2 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_instDistribLattice() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_instDistribLattice___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_instLattice() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_instDistribLattice;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_instSemilatticeInf___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_instDistribLattice;
x_2 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_instSemilatticeInf() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_instSemilatticeInf___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_instSemilatticeSup() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_instDistribLattice;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_instInf() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_linearOrder___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_instSup() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_linearOrder___closed__1;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Rat_instPartialOrder() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_instSemilatticeInf;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Rat_instPreorder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Rat_instPartialOrder;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_linearOrder___closed__0 = _init_lp_mathlib_Rat_linearOrder___closed__0();
lean_mark_persistent(lp_mathlib_Rat_linearOrder___closed__0);
lp_mathlib_Rat_linearOrder___closed__1 = _init_lp_mathlib_Rat_linearOrder___closed__1();
lean_mark_persistent(lp_mathlib_Rat_linearOrder___closed__1);
lp_mathlib_Rat_linearOrder___closed__2 = _init_lp_mathlib_Rat_linearOrder___closed__2();
lean_mark_persistent(lp_mathlib_Rat_linearOrder___closed__2);
lp_mathlib_Rat_linearOrder___closed__3 = _init_lp_mathlib_Rat_linearOrder___closed__3();
lean_mark_persistent(lp_mathlib_Rat_linearOrder___closed__3);
lp_mathlib_Rat_linearOrder___closed__4 = _init_lp_mathlib_Rat_linearOrder___closed__4();
lean_mark_persistent(lp_mathlib_Rat_linearOrder___closed__4);
lp_mathlib_Rat_linearOrder = _init_lp_mathlib_Rat_linearOrder();
lean_mark_persistent(lp_mathlib_Rat_linearOrder);
lp_mathlib_Rat_instDistribLattice___closed__0 = _init_lp_mathlib_Rat_instDistribLattice___closed__0();
lean_mark_persistent(lp_mathlib_Rat_instDistribLattice___closed__0);
lp_mathlib_Rat_instDistribLattice = _init_lp_mathlib_Rat_instDistribLattice();
lean_mark_persistent(lp_mathlib_Rat_instDistribLattice);
lp_mathlib_Rat_instLattice = _init_lp_mathlib_Rat_instLattice();
lean_mark_persistent(lp_mathlib_Rat_instLattice);
lp_mathlib_Rat_instSemilatticeInf___closed__0 = _init_lp_mathlib_Rat_instSemilatticeInf___closed__0();
lean_mark_persistent(lp_mathlib_Rat_instSemilatticeInf___closed__0);
lp_mathlib_Rat_instSemilatticeInf = _init_lp_mathlib_Rat_instSemilatticeInf();
lean_mark_persistent(lp_mathlib_Rat_instSemilatticeInf);
lp_mathlib_Rat_instSemilatticeSup = _init_lp_mathlib_Rat_instSemilatticeSup();
lean_mark_persistent(lp_mathlib_Rat_instSemilatticeSup);
lp_mathlib_Rat_instInf = _init_lp_mathlib_Rat_instInf();
lean_mark_persistent(lp_mathlib_Rat_instInf);
lp_mathlib_Rat_instSup = _init_lp_mathlib_Rat_instSup();
lean_mark_persistent(lp_mathlib_Rat_instSup);
lp_mathlib_Rat_instPartialOrder = _init_lp_mathlib_Rat_instPartialOrder();
lean_mark_persistent(lp_mathlib_Rat_instPartialOrder);
lp_mathlib_Rat_instPreorder = _init_lp_mathlib_Rat_instPreorder();
lean_mark_persistent(lp_mathlib_Rat_instPreorder);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
