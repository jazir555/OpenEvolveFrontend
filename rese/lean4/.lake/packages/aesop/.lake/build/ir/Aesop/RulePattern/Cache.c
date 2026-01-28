// Lean compiler output
// Module: Aesop.RulePattern.Cache
// Imports: public import Init public import Aesop.Forward.Substitution public import Aesop.Rule.Name
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
lean_object* lean_mk_array(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRulePatternCache;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instEmptyCollectionRulePatternCache;
static lean_object* lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRulePatternCache_default;
static lean_object* lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0;
static lean_object* _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_unsigned_to_nat(16u);
x_3 = lean_mk_array(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRulePatternCache() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedRulePatternCache_default;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instEmptyCollectionRulePatternCache() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_Substitution(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Rule_Name(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RulePattern_Cache(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_Substitution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0 = _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0);
lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1 = _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1);
lp_aesop_Aesop_instInhabitedRulePatternCache_default = _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRulePatternCache_default);
lp_aesop_Aesop_instInhabitedRulePatternCache = _init_lp_aesop_Aesop_instInhabitedRulePatternCache();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRulePatternCache);
lp_aesop_Aesop_instEmptyCollectionRulePatternCache = _init_lp_aesop_Aesop_instEmptyCollectionRulePatternCache();
lean_mark_persistent(lp_aesop_Aesop_instEmptyCollectionRulePatternCache);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
