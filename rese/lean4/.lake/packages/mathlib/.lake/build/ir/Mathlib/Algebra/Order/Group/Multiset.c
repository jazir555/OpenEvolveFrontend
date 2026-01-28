// Lean compiler output
// Module: Mathlib.Algebra.Order.Group.Multiset
// Imports: public import Init public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Algebra.Order.Monoid.Unbundled.ExistsOfLE public import Mathlib.Algebra.Order.Sub.Defs public import Mathlib.Data.Multiset.Fold
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
lean_object* lp_mathlib_Multiset_add(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countAddMonoidHom(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countAddMonoidHom___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid(lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1;
static lean_object* lp_mathlib_Multiset_cardHom___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countPAddMonoidHom(lean_object*, lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_cardHom(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_countP(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_card___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countPAddMonoidHom___redArg(lean_object*);
static lean_object* _init_lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Multiset_add), 3, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0;
x_2 = lean_box(0);
x_3 = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
lean_closure_set(x_3, 2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1;
x_2 = lean_box(0);
x_3 = lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2;
return x_2;
}
}
static lean_object* _init_lp_mathlib_Multiset_cardHom___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Multiset_card___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_cardHom(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiset_cardHom___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_List_replicateTR___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiset_replicateAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_replicateAddMonoidHom___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapAddMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Multiset_map), 4, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, lean_box(0));
lean_closure_set(x_4, 2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapAddMonoidHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiset_map), 4, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countPAddMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Multiset_countP), 4, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, lean_box(0));
lean_closure_set(x_4, 2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countPAddMonoidHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiset_countP), 4, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countAddMonoidHom___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Multiset_countP), 4, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, lean_box(0));
lean_closure_set(x_4, 2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countAddMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_countAddMonoidHom___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Fold(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0 = _init_lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0();
lean_mark_persistent(lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0);
lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1 = _init_lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1();
lean_mark_persistent(lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1);
lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2 = _init_lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2();
lean_mark_persistent(lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2);
lp_mathlib_Multiset_cardHom___closed__0 = _init_lp_mathlib_Multiset_cardHom___closed__0();
lean_mark_persistent(lp_mathlib_Multiset_cardHom___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
