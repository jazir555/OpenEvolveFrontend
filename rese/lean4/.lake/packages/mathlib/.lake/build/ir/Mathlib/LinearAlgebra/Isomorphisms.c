// Lean compiler output
// Module: Mathlib.LinearAlgebra.Isomorphisms
// Imports: public import Init public import Mathlib.LinearAlgebra.Quotient.Basic public import Mathlib.LinearAlgebra.Quotient.Card
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
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_Quotient_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_subToSupQuotient___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientSup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_quotEquivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_quotientInfToSupQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_subToSupQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_liftQ___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_quotientInfToSupQuotient___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_mapQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_subToSupQuotient(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_8 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_8);
x_9 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_9);
x_10 = lean_box(0);
x_11 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_4);
lean_inc(x_5);
x_12 = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_12, 0, x_5);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(x_13, 0, lean_box(0));
lean_closure_set(x_13, 1, lean_box(0));
lean_closure_set(x_13, 2, x_3);
lean_closure_set(x_13, 3, x_11);
lean_closure_set(x_13, 4, x_12);
lean_closure_set(x_13, 5, x_10);
x_14 = lp_mathlib_Submodule_inclusion(lean_box(0), lean_box(0), x_8, x_9, x_5, x_6, x_10, lean_box(0));
lean_dec(x_5);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
x_15 = lp_mathlib_LinearMap_comp___redArg(x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_subToSupQuotient___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_6);
x_7 = lean_box(0);
x_8 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_2);
lean_inc(x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, x_1);
lean_closure_set(x_10, 3, x_8);
lean_closure_set(x_10, 4, x_9);
lean_closure_set(x_10, 5, x_7);
x_11 = lp_mathlib_Submodule_inclusion(lean_box(0), lean_box(0), x_5, x_6, x_3, x_4, x_7, lean_box(0));
lean_dec(x_3);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
x_12 = lp_mathlib_LinearMap_comp___redArg(x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_quotientInfToSupQuotient___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_6);
x_7 = lean_box(0);
x_8 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_2);
lean_inc(x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, x_1);
lean_closure_set(x_10, 3, x_8);
lean_closure_set(x_10, 4, x_9);
lean_closure_set(x_10, 5, x_7);
x_11 = lp_mathlib_Submodule_inclusion(lean_box(0), lean_box(0), x_5, x_6, x_3, x_4, x_7, lean_box(0));
lean_dec(x_3);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
x_12 = lp_mathlib_LinearMap_comp___redArg(x_10, x_11);
x_13 = lp_mathlib_Submodule_liftQ___redArg(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_quotientInfToSupQuotient(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_LinearMap_quotientInfToSupQuotient___redArg(x_3, x_4, x_5, x_6);
return x_8;
}
}
static lean_object* _init_lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0;
x_6 = lp_mathlib_Submodule_mapQ___redArg(x_1, x_2, x_3, x_4, x_5);
x_7 = lp_mathlib_Submodule_liftQ___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg(x_3, x_4, x_5, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg(x_1, x_2, x_3, x_4);
x_7 = lean_apply_1(x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_Submodule_mapQ___redArg(x_1, x_2, x_3, x_4, x_5);
x_8 = lean_apply_1(x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__0), 5, 4);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_3);
lean_closure_set(x_6, 3, x_5);
lean_inc_ref(x_2);
x_7 = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(x_2);
lean_inc(x_3);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_3);
lean_inc_ref(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, x_1);
lean_closure_set(x_9, 3, x_2);
lean_closure_set(x_9, 4, x_3);
lean_closure_set(x_9, 5, x_4);
x_10 = lean_box(0);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__1), 6, 5);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_7);
lean_closure_set(x_11, 2, x_8);
lean_closure_set(x_11, 3, x_10);
lean_closure_set(x_11, 4, x_9);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_6);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg(x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientSup___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_2);
x_5 = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(x_2);
lean_inc(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_3);
x_7 = lean_box(0);
x_8 = lp_mathlib_Submodule_quotEquivOfEq(lean_box(0), lean_box(0), x_1, x_5, x_6, x_7, x_7, lean_box(0));
lean_dec_ref(x_6);
lean_dec_ref(x_5);
x_9 = lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg(x_1, x_2, x_3, x_4, x_7);
x_10 = lp_mathlib_LinearEquiv_trans___redArg(x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientSup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Submodule_quotientQuotientEquivQuotientSup___redArg(x_3, x_4, x_5, x_6);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Card(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0 = _init_lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
