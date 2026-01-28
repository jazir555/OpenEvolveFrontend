// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Shapes.Pullback.CommSq
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Constructions.ZeroObjects public import Mathlib.CategoryTheory.Limits.Shapes.BinaryBiproducts public import Mathlib.CategoryTheory.Limits.Shapes.Pullback.Pasting public import Mathlib.CategoryTheory.Limits.Shapes.Pullback.Iso
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_cone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_PushoutCocone_mk___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_IsPushout_cocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_PullbackCone_op___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_PullbackCone_unop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coneUnop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_IsPullback_cone___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coneUnop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_PullbackCone_mk___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_cocone___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coconeUnop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coconeOp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coconeUnop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coconeOp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_cocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_PushoutCocone_unop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CategoryStruct_opposite___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coneOp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_IsPushout_cocone___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_IsPullback_cone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_PushoutCocone_op___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_cone___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coneOp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_cone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_PullbackCone_mk___redArg(x_2, x_4, x_5, x_6, x_9, x_10, x_3, x_7, x_8);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_cone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_PullbackCone_mk___redArg(x_1, x_3, x_4, x_5, x_8, x_9, x_2, x_6, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_cocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_PushoutCocone_mk___redArg(x_2, x_3, x_4, x_5, x_7, x_8, x_6, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_cocone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_PushoutCocone_mk___redArg(x_1, x_2, x_3, x_4, x_6, x_7, x_5, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coneOp___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_1);
x_10 = lp_mathlib_CategoryTheory_CategoryStruct_opposite___redArg(x_1);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_1);
x_11 = lp_mathlib_CategoryTheory_Limits_PullbackCone_mk___redArg(x_1, x_3, x_4, x_5, x_8, x_9, x_2, x_6, x_7);
x_12 = lp_mathlib_CategoryTheory_Limits_PullbackCone_op___redArg(x_1, x_3, x_4, x_5, x_8, x_9, x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_10, x_13);
x_15 = lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coneOp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_CommSq_coneOp___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coconeOp___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_1);
x_10 = lp_mathlib_CategoryTheory_CategoryStruct_opposite___redArg(x_1);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_4);
lean_inc(x_3);
lean_inc(x_2);
lean_inc_ref(x_1);
x_11 = lp_mathlib_CategoryTheory_Limits_PushoutCocone_mk___redArg(x_1, x_2, x_3, x_4, x_6, x_7, x_5, x_8, x_9);
x_12 = lp_mathlib_CategoryTheory_Limits_PushoutCocone_op___redArg(x_1, x_2, x_3, x_4, x_6, x_7, x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_10, x_13);
x_15 = lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coconeOp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_CommSq_coconeOp___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coneUnop___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_1);
x_10 = lp_mathlib_CategoryTheory_CategoryStruct_opposite___redArg(x_1);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
x_11 = lp_mathlib_CategoryTheory_Limits_PullbackCone_mk___redArg(x_10, x_3, x_4, x_5, x_8, x_9, x_2, x_6, x_7);
lean_inc_ref(x_1);
x_12 = lp_mathlib_CategoryTheory_Limits_PullbackCone_unop___redArg(x_1, x_3, x_4, x_5, x_8, x_9, x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_1, x_13);
x_15 = lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coneUnop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_CommSq_coneUnop___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coconeUnop___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_1);
x_10 = lp_mathlib_CategoryTheory_CategoryStruct_opposite___redArg(x_1);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_4);
lean_inc(x_3);
lean_inc(x_2);
x_11 = lp_mathlib_CategoryTheory_Limits_PushoutCocone_mk___redArg(x_10, x_2, x_3, x_4, x_6, x_7, x_5, x_8, x_9);
lean_inc_ref(x_1);
x_12 = lp_mathlib_CategoryTheory_Limits_PushoutCocone_unop___redArg(x_1, x_2, x_3, x_4, x_6, x_7, x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_1, x_13);
x_15 = lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_CommSq_coconeUnop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_CommSq_coconeUnop___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_IsPullback_cone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_PullbackCone_mk___redArg(x_2, x_4, x_5, x_6, x_9, x_10, x_3, x_7, x_8);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_IsPullback_cone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_PullbackCone_mk___redArg(x_1, x_3, x_4, x_5, x_8, x_9, x_2, x_6, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_IsPushout_cocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_PushoutCocone_mk___redArg(x_2, x_3, x_4, x_5, x_7, x_8, x_6, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_IsPushout_cocone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_PushoutCocone_mk___redArg(x_1, x_2, x_3, x_4, x_6, x_7, x_5, x_8, x_9);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_ZeroObjects(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryBiproducts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_Pasting(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_Iso(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_CommSq(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_ZeroObjects(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryBiproducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_Pasting(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_Iso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
