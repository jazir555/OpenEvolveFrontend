// Lean compiler output
// Module: Mathlib.Topology.ContinuousMap.Compact
// Imports: public import Init public import Mathlib.Topology.ContinuousMap.Bounded.Star public import Mathlib.Topology.ContinuousMap.Star public import Mathlib.Topology.UniformSpace.Compact public import Mathlib.Topology.CompactOpen public import Mathlib.Topology.Sets.Compacts public import Mathlib.Analysis.Normed.Group.InfiniteSum
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
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_equivBoundedOfCompact(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_addEquivBoundedOfCompact(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_equivBoundedOfCompact___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_normedSpace___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_BoundedContinuousFunction_mkOfCompact___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNormedAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ContinuousMap_C___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_isometryEquivBoundedOfCompact___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_equivBoundedOfCompact___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_linearIsometryBoundedOfCompact___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNormedAlgebra___redArg(lean_object*);
static lean_object* lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNormedAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_isometryEquivBoundedOfCompact(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_normedSpace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_normedSpace___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_linearIsometryBoundedOfCompact___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_BoundedContinuousFunction_toContinuousMapMonoidHom___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_ContinuousMap_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_linearIsometryBoundedOfCompact(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_equivBoundedOfCompact___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_equivBoundedOfCompact___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_equivBoundedOfCompact___redArg___lam__0), 2, 0);
x_4 = lean_alloc_closure((void*)(lp_mathlib_BoundedContinuousFunction_mkOfCompact___boxed), 6, 5);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, lean_box(0));
lean_closure_set(x_4, 2, x_1);
lean_closure_set(x_4, 3, x_2);
lean_closure_set(x_4, 4, lean_box(0));
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_equivBoundedOfCompact(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ContinuousMap_equivBoundedOfCompact___redArg(x_3, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_BoundedContinuousFunction_toContinuousMapMonoidHom___lam__0), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lp_mathlib_ContinuousMap_equivBoundedOfCompact___redArg(x_1, x_2);
x_4 = lp_mathlib_Equiv_symm___redArg(x_3);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_4, 0);
lean_dec(x_6);
x_7 = lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg___closed__0;
lean_ctor_set(x_4, 0, x_7);
x_8 = lp_mathlib_Equiv_symm___redArg(x_4);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_4, 1);
lean_inc(x_9);
lean_dec(x_4);
x_10 = lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg___closed__0;
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_9);
x_12 = lp_mathlib_Equiv_symm___redArg(x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_addEquivBoundedOfCompact(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg(x_3, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_ContinuousMap_addEquivBoundedOfCompact(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_isometryEquivBoundedOfCompact(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ContinuousMap_equivBoundedOfCompact___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_isometryEquivBoundedOfCompact___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ContinuousMap_equivBoundedOfCompact___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_normedSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_9, 0, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_normedSpace___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_normedSpace___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ContinuousMap_normedSpace(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_linearIsometryBoundedOfCompact___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg(x_1, x_3);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
return x_4;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_4);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_linearIsometryBoundedOfCompact(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_linearIsometryBoundedOfCompact___redArg(x_3, x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_linearIsometryBoundedOfCompact___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_linearIsometryBoundedOfCompact(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNormedAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_ContinuousMap_C___redArg(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNormedAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ContinuousMap_instNormedAlgebra___redArg(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNormedAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ContinuousMap_instNormedAlgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_ContinuousMap_Bounded_Star(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_ContinuousMap_Star(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Compact(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_CompactOpen(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Sets_Compacts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_InfiniteSum(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_ContinuousMap_Compact(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_ContinuousMap_Bounded_Star(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_ContinuousMap_Star(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_Compact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_CompactOpen(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Sets_Compacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Group_InfiniteSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg___closed__0 = _init_lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg___closed__0();
lean_mark_persistent(lp_mathlib_ContinuousMap_addEquivBoundedOfCompact___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
