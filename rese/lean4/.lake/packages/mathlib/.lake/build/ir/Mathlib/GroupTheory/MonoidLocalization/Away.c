// Lean compiler output
// Module: Mathlib.GroupTheory.MonoidLocalization.Away
// Imports: public import Init public import Mathlib.Algebra.Group.Submonoid.Membership public import Mathlib.GroupTheory.MonoidLocalization.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_addMonoidOf___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_negSelf___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_monoidOf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_invSelf___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_addMonoidOf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_monoidOf___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_monoidOf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_invSelf(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_invSelf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Localization_monoidOf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_addMonoidOf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_invSelf___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_negSelf___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_addMonoidOf___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_monoidOf___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_negSelf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddLocalization_addMonoidOf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_negSelf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_invSelf___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_3, 1);
lean_dec(x_5);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec(x_3);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_2);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_invSelf(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Localization_Away_invSelf___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_invSelf___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Localization_Away_invSelf(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_invSelf___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Localization_Away_invSelf___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_negSelf___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_3, 1);
lean_dec(x_5);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec(x_3);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_2);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_negSelf(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddLocalization_Away_negSelf___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_negSelf___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddLocalization_Away_negSelf(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_negSelf___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddLocalization_Away_negSelf___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_monoidOf(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Localization_monoidOf___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_monoidOf___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Localization_monoidOf___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_monoidOf___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Localization_Away_monoidOf(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_Away_monoidOf___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Localization_Away_monoidOf___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_addMonoidOf(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddLocalization_addMonoidOf___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_addMonoidOf___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddLocalization_addMonoidOf___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_addMonoidOf___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddLocalization_Away_addMonoidOf(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_Away_addMonoidOf___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddLocalization_Away_addMonoidOf___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Away(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
