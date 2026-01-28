// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.ExteriorPower
// Imports: public import Init public import Mathlib.LinearAlgebra.ExteriorPower.Basic public import Mathlib.Algebra.Category.ModuleCat.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_AlternatingMap_postcomp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_AlternatingMap_postcomp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_compMultilinearMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_exteriorPower_00_u03b9Multi___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_AlternatingMap_postcomp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlternatingMap_instFunLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower_mk(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instRingCliffordAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower_mk___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower___redArg(lean_object*);
lean_object* lp_mathlib_instAlgebraCliffordAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
x_3 = lp_mathlib_Ring_toAddCommGroup___redArg(x_2);
x_4 = lp_mathlib_instAlgebraCliffordAlgebra___redArg(x_1);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_dec(x_7);
x_8 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_6);
lean_ctor_set(x_4, 1, x_9);
lean_ctor_set(x_4, 0, x_8);
return x_4;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_ctor_get(x_4, 0);
lean_inc(x_10);
lean_dec(x_4);
x_11 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_12, 0, x_10);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_exteriorPower___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_exteriorPower(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_AlternatingMap_instFunLike___lam__0), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier___closed__0;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_AlternatingMap_postcomp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_LinearMap_compMultilinearMap___redArg(x_8, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_AlternatingMap_postcomp___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LinearMap_compMultilinearMap___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_AlternatingMap_postcomp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ModuleCat_AlternatingMap_postcomp(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower_mk(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_exteriorPower_00_u03b9Multi___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower_mk___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_exteriorPower_00_u03b9Multi___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_exteriorPower_mk___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_exteriorPower_mk(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_ExteriorPower_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_ExteriorPower(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_ExteriorPower_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier___closed__0 = _init_lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier___closed__0();
lean_mark_persistent(lp_mathlib_ModuleCat_instFunLikeAlternatingMapForallFinCarrier___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
