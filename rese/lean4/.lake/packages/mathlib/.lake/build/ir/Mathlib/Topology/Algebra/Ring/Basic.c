// Lean compiler output
// Module: Mathlib.Topology.Algebra.Ring.Basic
// Imports: public import Init public import Mathlib.Algebra.Order.AbsoluteValue.Basic public import Mathlib.Algebra.Ring.Opposite public import Mathlib.Algebra.Ring.Prod public import Mathlib.Algebra.Ring.Subring.Basic public import Mathlib.Topology.Algebra.Group.GroupTopology
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
static lean_object* lp_mathlib_RingTopology_inhabited___closed__0;
lean_object* lp_mathlib_MulHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_commSemiringTopologicalClosure___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_nonUnitalCommSemiringTopologicalClosure___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instCompleteLattice(lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfCompleteSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology_orderEmbedding(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_topologicalClosure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_nonUnitalCommRingTopologicalClosure___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topologicalClosure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_topologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_coinduced___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_commRingTopologicalClosure___redArg(lean_object*);
static lean_object* lp_mathlib_RingTopology_instPartialOrder___closed__0;
lean_object* lp_mathlib_TopologicalSpace_instCompleteLattice(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_nonUnitalCommRingTopologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_topologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Topology_Algebra_Ring_Basic_0__RingTopology_def__sInf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_topologicalClosure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology_orderEmbedding___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_instFunLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_SubringClass_toRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_nonUnitalCommSemiringTopologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subsemiring_toSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_coinduced(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_inhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_commSemiringTopologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instCompleteSemilatticeInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_topologicalClosure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_topologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_inhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_commRingTopologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instCompleteSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Topology_Algebra_Ring_Basic_0__RingTopology_def__sInf___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_topologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_topologicalClosure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_NonUnitalSubsemiring_topologicalClosure(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_nonUnitalCommSemiringTopologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_nonUnitalCommSemiringTopologicalClosure___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topologicalClosure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subsemiring_topologicalClosure(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_commSemiringTopologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Subsemiring_toSemiring___redArg(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_commSemiringTopologicalClosure___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Subsemiring_toSemiring___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_topologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_topologicalClosure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_NonUnitalSubring_topologicalClosure(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_nonUnitalCommRingTopologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_nonUnitalCommRingTopologicalClosure___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_topologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_topologicalClosure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subring_topologicalClosure(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_commRingTopologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_SubringClass_toRing___redArg(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_commRingTopologicalClosure___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SubringClass_toRing___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_RingTopology_inhabited___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_TopologicalSpace_instCompleteLattice(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_inhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_RingTopology_inhabited___closed__0;
x_4 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_inhabited___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RingTopology_inhabited(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_RingTopology_instPartialOrder___closed__0() {
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
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instPartialOrder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RingTopology_instPartialOrder___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instPartialOrder___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RingTopology_instPartialOrder(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Topology_Algebra_Ring_Basic_0__RingTopology_def__sInf(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Topology_Algebra_Ring_Basic_0__RingTopology_def__sInf___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib___private_Mathlib_Topology_Algebra_Ring_Basic_0__RingTopology_def__sInf(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instCompleteSemilatticeInf___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_RingTopology_instPartialOrder(lean_box(0), x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Topology_Algebra_Ring_Basic_0__RingTopology_def__sInf___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instCompleteSemilatticeInf(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RingTopology_instCompleteSemilatticeInf___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instCompleteLattice___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_RingTopology_instCompleteSemilatticeInf___redArg(x_1);
x_3 = lp_mathlib_completeLatticeOfCompleteSemilatticeInf___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_instCompleteLattice(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RingTopology_instCompleteLattice___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_coinduced(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_coinduced___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingTopology_coinduced(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingTopology_toAddGroupTopology(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology_orderEmbedding(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_RingTopology_toAddGroupTopology___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTopology_toAddGroupTopology_orderEmbedding___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_RingTopology_toAddGroupTopology___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_comp___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_instFunLike___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_MulHom_comp___redArg(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_comp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_AbsoluteValue_comp___redArg(x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_comp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_AbsoluteValue_comp(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Group_GroupTopology(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Ring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Group_GroupTopology(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_RingTopology_inhabited___closed__0 = _init_lp_mathlib_RingTopology_inhabited___closed__0();
lean_mark_persistent(lp_mathlib_RingTopology_inhabited___closed__0);
lp_mathlib_RingTopology_instPartialOrder___closed__0 = _init_lp_mathlib_RingTopology_instPartialOrder___closed__0();
lean_mark_persistent(lp_mathlib_RingTopology_instPartialOrder___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
