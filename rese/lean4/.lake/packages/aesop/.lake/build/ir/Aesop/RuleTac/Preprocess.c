// Lean compiler output
// Module: Aesop.RuleTac.Preprocess
// Imports: public import Init public import Aesop.RuleTac.Basic public import Aesop.Script.SpecificTactics
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_preprocess___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_renameInaccessibleFVarsS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static lean_object* lp_aesop_Aesop_RuleTac_preprocess___closed__0;
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_RuleTac_preprocess___closed__1;
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_preprocess(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_RuleTac_preprocess___closed__2;
static lean_object* _init_lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0;
x_9 = lean_st_mk_ref(x_8);
lean_inc(x_9);
x_10 = lean_apply_7(x_1, x_9, x_2, x_3, x_4, x_5, x_6, lean_box(0));
if (lean_obj_tag(x_10) == 0)
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_st_ref_get(x_9);
lean_dec(x_9);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
lean_ctor_set(x_10, 0, x_14);
return x_10;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = lean_ctor_get(x_10, 0);
lean_inc(x_15);
lean_dec(x_10);
x_16 = lean_st_ref_get(x_9);
lean_dec(x_9);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
else
{
uint8_t x_19; 
lean_dec(x_9);
x_19 = !lean_is_exclusive(x_10);
if (x_19 == 0)
{
return x_10;
}
else
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_10, 0);
lean_inc(x_20);
lean_dec(x_10);
x_21 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_preprocess___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_unsigned_to_nat(16u);
x_3 = lean_mk_array(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_preprocess___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_RuleTac_preprocess___closed__0;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_preprocess___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_preprocess(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_1, 0);
lean_inc(x_8);
lean_dec_ref(x_1);
lean_inc(x_8);
x_9 = lean_alloc_closure((void*)(lp_aesop_Aesop_renameInaccessibleFVarsS___boxed), 8, 1);
lean_closure_set(x_9, 0, x_8);
lean_inc(x_6);
lean_inc(x_4);
x_10 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(x_9, x_2, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_10) == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_11, 1);
lean_inc(x_13);
lean_dec(x_11);
x_14 = lean_ctor_get(x_12, 0);
lean_inc(x_14);
lean_dec(x_12);
x_15 = l_Lean_Meta_saveState___redArg(x_4, x_6);
lean_dec(x_6);
lean_dec(x_4);
if (lean_obj_tag(x_15) == 0)
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = lp_aesop_Aesop_RuleTac_preprocess___closed__1;
x_19 = 0;
x_20 = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(x_20, 0, x_8);
lean_ctor_set(x_20, 1, x_14);
lean_ctor_set(x_20, 2, x_18);
lean_ctor_set(x_20, 3, x_18);
lean_ctor_set_uint8(x_20, sizeof(void*)*4, x_19);
x_21 = lp_aesop_Aesop_RuleTac_preprocess___closed__2;
x_22 = lean_array_push(x_21, x_20);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_13);
x_24 = lean_box(0);
x_25 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_25, 0, x_22);
lean_ctor_set(x_25, 1, x_17);
lean_ctor_set(x_25, 2, x_23);
lean_ctor_set(x_25, 3, x_24);
x_26 = lean_array_push(x_21, x_25);
lean_ctor_set(x_15, 0, x_26);
return x_15;
}
else
{
lean_object* x_27; lean_object* x_28; uint8_t x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_27 = lean_ctor_get(x_15, 0);
lean_inc(x_27);
lean_dec(x_15);
x_28 = lp_aesop_Aesop_RuleTac_preprocess___closed__1;
x_29 = 0;
x_30 = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(x_30, 0, x_8);
lean_ctor_set(x_30, 1, x_14);
lean_ctor_set(x_30, 2, x_28);
lean_ctor_set(x_30, 3, x_28);
lean_ctor_set_uint8(x_30, sizeof(void*)*4, x_29);
x_31 = lp_aesop_Aesop_RuleTac_preprocess___closed__2;
x_32 = lean_array_push(x_31, x_30);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_13);
x_34 = lean_box(0);
x_35 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_35, 0, x_32);
lean_ctor_set(x_35, 1, x_27);
lean_ctor_set(x_35, 2, x_33);
lean_ctor_set(x_35, 3, x_34);
x_36 = lean_array_push(x_31, x_35);
x_37 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_37, 0, x_36);
return x_37;
}
}
else
{
uint8_t x_38; 
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_8);
x_38 = !lean_is_exclusive(x_15);
if (x_38 == 0)
{
return x_15;
}
else
{
lean_object* x_39; lean_object* x_40; 
x_39 = lean_ctor_get(x_15, 0);
lean_inc(x_39);
lean_dec(x_15);
x_40 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_40, 0, x_39);
return x_40;
}
}
}
else
{
uint8_t x_41; 
lean_dec(x_8);
lean_dec(x_6);
lean_dec(x_4);
x_41 = !lean_is_exclusive(x_10);
if (x_41 == 0)
{
return x_10;
}
else
{
lean_object* x_42; lean_object* x_43; 
x_42 = lean_ctor_get(x_10, 0);
lean_inc(x_42);
lean_dec(x_10);
x_43 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_43, 0, x_42);
return x_43;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_preprocess___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_RuleTac_preprocess(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_Preprocess(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0 = _init_lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0);
lp_aesop_Aesop_RuleTac_preprocess___closed__0 = _init_lp_aesop_Aesop_RuleTac_preprocess___closed__0();
lean_mark_persistent(lp_aesop_Aesop_RuleTac_preprocess___closed__0);
lp_aesop_Aesop_RuleTac_preprocess___closed__1 = _init_lp_aesop_Aesop_RuleTac_preprocess___closed__1();
lean_mark_persistent(lp_aesop_Aesop_RuleTac_preprocess___closed__1);
lp_aesop_Aesop_RuleTac_preprocess___closed__2 = _init_lp_aesop_Aesop_RuleTac_preprocess___closed__2();
lean_mark_persistent(lp_aesop_Aesop_RuleTac_preprocess___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
