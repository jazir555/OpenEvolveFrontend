// Lean compiler output
// Module: Aesop.RuleTac
// Imports: public import Init public import Aesop.RuleTac.Apply public import Aesop.RuleTac.Basic public import Aesop.RuleTac.Cases public import Aesop.RuleTac.Forward public import Aesop.RuleTac.Preprocess public import Aesop.RuleTac.Tactic
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
lean_object* lp_aesop_Aesop_RuleTac_applyConsts(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_tacticStx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTacDescr_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_singleRuleTacImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_preprocess(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTacDescr_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_cases(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_ruleTacImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_apply(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_forward(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_forwardMatches(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTacDescr_run(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
lean_object* x_9; uint8_t x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get_uint8(x_1, sizeof(void*)*1);
lean_dec_ref(x_1);
x_11 = lp_aesop_Aesop_RuleTac_apply(x_9, x_10, x_2, x_3, x_4, x_5, x_6, x_7);
return x_11;
}
case 1:
{
lean_object* x_12; uint8_t x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_12);
x_13 = lean_ctor_get_uint8(x_1, sizeof(void*)*1);
lean_dec_ref(x_1);
x_14 = lp_aesop_Aesop_RuleTac_applyConsts(x_12, x_13, x_2, x_3, x_4, x_5, x_6, x_7);
return x_14;
}
case 2:
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; lean_object* x_18; 
x_15 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_15);
x_16 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_16);
x_17 = lean_ctor_get_uint8(x_1, sizeof(void*)*2);
lean_dec_ref(x_1);
x_18 = lp_aesop_Aesop_RuleTac_forward(x_15, x_16, x_17, x_2, x_3, x_4, x_5, x_6, x_7);
return x_18;
}
case 3:
{
lean_object* x_19; uint8_t x_20; uint8_t x_21; lean_object* x_22; lean_object* x_23; 
x_19 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_19);
x_20 = lean_ctor_get_uint8(x_1, sizeof(void*)*2);
x_21 = lean_ctor_get_uint8(x_1, sizeof(void*)*2 + 1);
x_22 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_22);
lean_dec_ref(x_1);
x_23 = lp_aesop_Aesop_RuleTac_cases(x_19, x_20, x_21, x_22, x_2, x_3, x_4, x_5, x_6, x_7);
return x_23;
}
case 4:
{
lean_object* x_24; lean_object* x_25; 
x_24 = lean_ctor_get(x_1, 0);
lean_inc(x_24);
lean_dec_ref(x_1);
x_25 = lp_aesop_Aesop_RuleTac_tacticMImpl(x_24, x_2, x_3, x_4, x_5, x_6, x_7);
return x_25;
}
case 5:
{
lean_object* x_26; lean_object* x_27; 
x_26 = lean_ctor_get(x_1, 0);
lean_inc(x_26);
lean_dec_ref(x_1);
x_27 = lp_aesop_Aesop_RuleTac_ruleTacImpl(x_26, x_2, x_3, x_4, x_5, x_6, x_7);
return x_27;
}
case 6:
{
lean_object* x_28; lean_object* x_29; 
x_28 = lean_ctor_get(x_1, 0);
lean_inc(x_28);
lean_dec_ref(x_1);
x_29 = lp_aesop_Aesop_RuleTac_tacGenImpl(x_28, x_2, x_3, x_4, x_5, x_6, x_7);
return x_29;
}
case 7:
{
lean_object* x_30; lean_object* x_31; 
x_30 = lean_ctor_get(x_1, 0);
lean_inc(x_30);
lean_dec_ref(x_1);
x_31 = lp_aesop_Aesop_RuleTac_singleRuleTacImpl(x_30, x_2, x_3, x_4, x_5, x_6, x_7);
return x_31;
}
case 8:
{
lean_object* x_32; lean_object* x_33; 
x_32 = lean_ctor_get(x_1, 0);
lean_inc(x_32);
lean_dec_ref(x_1);
x_33 = lp_aesop_Aesop_RuleTac_tacticStx(x_32, x_2, x_3, x_4, x_5, x_6, x_7);
return x_33;
}
case 9:
{
lean_object* x_34; 
x_34 = lp_aesop_Aesop_RuleTac_preprocess(x_2, x_3, x_4, x_5, x_6, x_7);
return x_34;
}
default: 
{
lean_object* x_35; lean_object* x_36; 
x_35 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_35);
lean_dec_ref(x_1);
x_36 = lp_aesop_Aesop_RuleTac_forwardMatches(x_35, x_2, x_3, x_4, x_5, x_6, x_7);
return x_36;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTacDescr_run___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_RuleTacDescr_run(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Apply(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Cases(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Forward(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Preprocess(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Tactic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Preprocess(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
