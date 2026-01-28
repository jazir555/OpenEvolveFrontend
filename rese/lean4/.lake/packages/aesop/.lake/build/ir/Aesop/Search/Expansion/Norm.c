// Lean compiler output
// Module: Aesop.Search.Expansion.Norm
// Imports: public import Init public import Aesop.Forward.State.ApplyGoalDiff public import Aesop.RuleTac public import Aesop.RuleTac.ElabRuleTerm public import Aesop.Script.SpecificTactics public import Aesop.Search.Expansion.Basic public import Aesop.Search.Expansion.Simp public import Aesop.Search.RuleSelection public import Aesop.Search.SearchM public import Aesop.Tree.State public import Batteries.Lean.HashSet
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
extern lean_object* l_Lean_Core_instMonadTraceCoreM;
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkNormSimpScriptStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPreSimpRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__1;
static lean_object* lp_aesop_Aesop_normSimp___closed__0;
lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_optNormRuleResultEmoji___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfoldCore___lam__0(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__11;
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormRuleTac___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstNormRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__33;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPreSimpRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorIdx___boxed(lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__22;
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
static lean_object* lp_aesop_Aesop_NormStep_simp___redArg___closed__1;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__1;
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__31;
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__16;
size_t lean_usize_shift_right(size_t, size_t);
static lean_object* lp_aesop_Aesop_checkSimp___closed__34;
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5(lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__8;
static lean_object* lp_aesop_Aesop_normSimp___closed__1;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimpCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPreSimpRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__9;
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static size_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyForwardState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__0(lean_object*);
static lean_object* lp_aesop_Aesop_NormRuleResult_toNormSeqResult___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___redArg___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormRule___closed__2;
static lean_object* lp_aesop_Aesop_updateForwardState___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateForwardState___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
double lean_float_div(double, double);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Array_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormSteps___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__26;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withNormTraceNode___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_uint64_to_usize(uint64_t);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__6;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__20;
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__36;
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__11;
lean_object* l_Lean_MessageData_ofList(lean_object*);
static lean_object* lp_aesop_Aesop_runNormRuleTac___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withNormTraceNode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
static lean_object* lp_aesop_Aesop_checkSimp___closed__21;
lean_object* lean_array_push(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normUnfold___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_unfold___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__15;
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_runNormRule_spec__8___boxed(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Check_get(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateForwardState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__17;
static lean_object* lp_aesop_Aesop_checkSimp___closed__8;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_KVMap_find(lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__4;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__29;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_steps_x3f(lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__28;
static double lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runFirstNormRule_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadTraceOfMonadLift___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormRuleTac___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfold___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object*, lean_object*);
uint8_t lean_float_decLt(double, double);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__2;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__12;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_succeeded_elim___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_isAssigned___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__11;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_unchanged_elim___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__23;
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__13;
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__43;
static lean_object* lp_aesop_Aesop_checkSimp___closed__19;
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___redArg___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__9;
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormSteps___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withNormTraceNode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normSimp___lam__0___closed__1;
static lean_object* lp_aesop_Aesop_checkSimp___closed__5;
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__1;
lean_object* l_Lean_stringToMessageData(lean_object*);
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__12;
LEAN_EXPORT lean_object* lp_aesop_Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__2;
static lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormSteps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_NormM_instInhabitedState_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optNormRuleResultToNormSeqResult(lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__18;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_eraseForwardRuleMatch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__17;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfoldCore___lam__0___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__42;
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__13;
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg(lean_object*);
static lean_object* lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__0;
lean_object* lp_aesop_Aesop_selectNormRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__10;
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getMVarDependencies___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__4;
lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormRuleTac___closed__1;
static lean_object* lp_aesop_Aesop_NormStep_unfold___redArg___closed__0;
static lean_object* lp_aesop_Aesop_normSimp___closed__2;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__41;
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___boxed(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedForwardState_default;
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyForwardState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getResetForwardState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__16;
lean_object* lp_aesop_Aesop_diffGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyForwardState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__5;
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__3;
static lean_object* lp_aesop_Aesop_checkSimp___closed__27;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRuleTac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__1;
static lean_object* lp_aesop_Aesop_normUnfold___lam__0___closed__0;
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__16;
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPostSimpRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normUnfold___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__1;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__18;
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_unfold___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_withTraceNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__10;
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__7;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__1;
static lean_object* lp_aesop_Aesop_checkSimp___closed__15;
static lean_object* lp_aesop_Aesop_checkSimp___closed__4;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3_spec__3(size_t, size_t, lean_object*);
lean_object* lp_aesop_Aesop_runRuleTac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getResetForwardState___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Check_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
static lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimpCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__5(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadExceptOfEST(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__1;
static lean_object* lp_aesop_Aesop_checkSimp___closed__7;
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__9;
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_aesop_Aesop_RuleTacDescr_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardState___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPreSimpRules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEST(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* l_Array_empty(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormSteps_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__0;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__0;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__10;
lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__32;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstNormRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__30;
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__34;
static lean_object* lp_aesop_Aesop_runNormRule___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfoldCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__4;
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateForwardState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormM_instInhabitedState_default;
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__8;
extern lean_object* lp_aesop_Aesop_Check_rules;
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_proved_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2(lean_object*);
lean_object* lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__0;
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimpCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormRuleTac___closed__8;
static lean_object* lp_aesop_Aesop_checkSimp___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_runNormRule_spec__8(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormRuleTac___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormM_instInhabitedState;
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_checkTraceOption(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__28;
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getResetForwardState___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__40;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPostSimpRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_simp___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___redArg(lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5_spec__5___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__14;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_BaseM_instMonadStats;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__2;
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__12(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__22;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_changed_elim___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__35;
static lean_object* lp_aesop_Aesop_optNormRuleResultEmoji___closed__2;
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__4;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__2;
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__9(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_unchanged_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_changed_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__2;
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__15(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
static double lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0;
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_proved_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__0;
extern lean_object* l_Lean_trace_profiler_threshold;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfold___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optNormRuleResultEmoji(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_GoalRef_markProvenByNormalization(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_succeeded_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__17;
lean_object* l_Lean_instMonadAlwaysExceptReaderT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormSteps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__19;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__0;
extern lean_object* lp_aesop_Aesop_treeImpl;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normUnfoldCore___closed__0;
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__12;
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withNormTraceNode___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalMVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_simpAll(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__13;
lean_object* lean_usize_to_nat(size_t);
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__8;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__14(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadFinallyEST___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_updateForwardState___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_newGoal_x3f(lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_updateForwardState___redArg___closed__0;
lean_object* lp_aesop_Aesop_getRootMetaState___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
static lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPostSimpRules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_NormStep_unfold___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyDiffToForwardState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__6;
extern lean_object* lp_aesop_Aesop_ruleFailureEmoji;
lean_object* lp_aesop_Aesop_ForwardRuleMatches_erase(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Goal_isRoot(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_proved_elim___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__25;
static lean_object* lp_aesop_Aesop_NormM_instInhabitedState_default___closed__2;
extern lean_object* lp_aesop_Aesop_instInhabitedForwardRuleMatches_default;
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_steps;
static lean_object* lp_aesop_Aesop_normUnfold___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg___closed__0;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__25;
lean_object* lean_array_fget(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__10;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0___redArg(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorElim___redArg(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_steps_x3f___boxed(lean_object*);
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimpCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__24;
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__10;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__31;
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1(lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Option_instBEq_beq___at___00Aesop_normSimp_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__13(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_eraseForwardRuleMatch___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__29;
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormRule___closed__0;
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
static lean_object* lp_aesop_Aesop_runNormRuleTac___closed__7;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__6;
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normSimp___lam__0___closed__0;
lean_object* l_instMonadLiftBaseIOEIO___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_eraseForwardRuleMatch___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_unfoldManyStarS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__26;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runFirstNormRule_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__3;
lean_object* lp_aesop_Aesop_elabRuleTermForSimpMetaM(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_ruleSuccessEmoji;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfold___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_toNormSeqResult(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_Name_hash___override(lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__14;
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__39;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__9;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__8;
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__11;
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__15;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__18;
static lean_object* lp_aesop_Aesop_normUnfold___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_unfold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__5;
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instBEqMVarId_beq___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormRuleTac___closed__6;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__24;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__1;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__27;
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__4;
static lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfoldCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__7;
static lean_object* lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_newGoal_x3f___boxed(lean_object*);
lean_object* l_instMonadLiftT___lam__0___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__8;
static double lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_liftIOCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorIdx(lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__23;
static lean_object* lp_aesop_Aesop_normalizeGoalMVar___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateForwardState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__30;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRuleTac___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__38;
lean_object* lean_array_uget(lean_object*, size_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
extern lean_object* l_Lean_trace_profiler;
static lean_object* lp_aesop_Aesop_checkSimp___closed__33;
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormSteps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getResetForwardState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Option_instBEq_beq___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__0;
static lean_object* lp_aesop_Aesop_runFirstNormRule___closed__0;
lean_object* lp_batteries_Lean_Meta_getAssignedExprMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__32;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optNormRuleResultEmoji___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(size_t, size_t, lean_object*);
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_simp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__11;
static size_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__0;
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__14;
static lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__1;
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Check_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__14;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
static lean_object* lp_aesop_Aesop_optNormRuleResultEmoji___closed__1;
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0___redArg___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPostSimpRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorIdx(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__21;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardRuleMatches_update(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__2;
static lean_object* lp_aesop_Aesop_normUnfoldCore___closed__1;
extern lean_object* l_Lean_Meta_instMonadMCtxMetaM;
lean_object* lp_aesop_Aesop_simpGoalWithAllHypotheses(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_NormM_instInhabitedState_default___closed__1;
LEAN_EXPORT uint8_t lp_aesop_Option_instBEq_beq___at___00Aesop_normSimp_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormRuleTac___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_proved_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyDiffToForwardState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkSimp___closed__12;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__20;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_simp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__7;
lean_object* lp_aesop_Aesop_RuleTacDescr_forwardRuleMatches_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__12;
lean_object* l_instMonadLiftTOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_tryFinally___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyForwardState___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__0;
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__37;
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardState_update(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_runNormSteps___redArg___closed__6;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_withNormTraceNode___closed__3;
static lean_object* lp_aesop_Aesop_normSimpCore___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_eraseForwardRuleMatch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardState___redArg(lean_object*);
size_t lean_usize_land(size_t, size_t);
static lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__13;
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnly___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_unfold___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorIdx___boxed(lean_object*);
extern lean_object* lp_aesop_Aesop_ruleProvedEmoji;
static lean_object* lp_aesop_Aesop_NormStep_simp___redArg___closed__0;
double lean_float_sub(double, double);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_NormM_instInhabitedState_default___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedForwardState_default;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NormM_instInhabitedState_default___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedForwardRuleMatches_default;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NormM_instInhabitedState_default___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_NormM_instInhabitedState_default___closed__1;
x_2 = lp_aesop_Aesop_NormM_instInhabitedState_default___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_NormM_instInhabitedState_default() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_NormM_instInhabitedState_default___closed__2;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NormM_instInhabitedState() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_NormM_instInhabitedState_default;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardState___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_st_ref_get(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec(x_3);
x_5 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardState(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_getForwardState___redArg(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardState___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_getForwardState(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardState___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_getForwardState___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__0;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__1;
return x_2;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__0;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__1;
return x_2;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__0;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__1;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getResetForwardState___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_st_ref_take(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0(lean_box(0));
x_7 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1(lean_box(0));
x_8 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2(lean_box(0));
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_7);
lean_ctor_set(x_9, 2, x_8);
lean_ctor_set(x_3, 0, x_9);
x_10 = lean_st_ref_set(x_1, x_3);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_5);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_12 = lean_ctor_get(x_3, 0);
x_13 = lean_ctor_get(x_3, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_3);
x_14 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0(lean_box(0));
x_15 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1(lean_box(0));
x_16 = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2(lean_box(0));
x_17 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_17, 0, x_14);
lean_ctor_set(x_17, 1, x_15);
lean_ctor_set(x_17, 2, x_16);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_13);
x_19 = lean_st_ref_set(x_1, x_18);
x_20 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_20, 0, x_12);
return x_20;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getResetForwardState(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_getResetForwardState___redArg(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getResetForwardState___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_getResetForwardState(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getResetForwardState___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_getResetForwardState___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_modifyForwardState___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyForwardState___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_st_ref_take(x_4);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_8 = lean_ctor_get(x_6, 1);
x_9 = lean_ctor_get(x_6, 0);
lean_dec(x_9);
x_10 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_11 = lp_aesop_Aesop_ForwardRuleMatches_update(x_2, x_3, x_10, x_8);
lean_ctor_set(x_6, 1, x_11);
lean_ctor_set(x_6, 0, x_1);
x_12 = lean_st_ref_set(x_4, x_6);
x_13 = lean_box(0);
x_14 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_15 = lean_ctor_get(x_6, 1);
lean_inc(x_15);
lean_dec(x_6);
x_16 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_17 = lp_aesop_Aesop_ForwardRuleMatches_update(x_2, x_3, x_16, x_15);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_1);
lean_ctor_set(x_18, 1, x_17);
x_19 = lean_st_ref_set(x_4, x_18);
x_20 = lean_box(0);
x_21 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyForwardState(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_modifyForwardState___redArg(x_1, x_2, x_3, x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyForwardState___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_modifyForwardState(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyForwardState___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_modifyForwardState___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
static lean_object* _init_lp_aesop_Aesop_updateForwardState___redArg___closed__0() {
_start:
{
uint8_t x_1; lean_object* x_2; lean_object* x_3; 
x_1 = 0;
x_2 = lean_box(x_1);
x_3 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_updateForwardState___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_unsigned_to_nat(16u);
x_3 = lean_mk_array(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_updateForwardState___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_updateForwardState___redArg___closed__1;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateForwardState___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lp_aesop_Aesop_getResetForwardState___redArg(x_2);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_aesop_Aesop_updateForwardState___redArg___closed__0;
x_12 = lp_aesop_Aesop_ForwardState_update(x_1, x_10, x_11, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc(x_15);
lean_dec(x_13);
x_16 = lp_aesop_Aesop_updateForwardState___redArg___closed__2;
x_17 = lp_aesop_Aesop_modifyForwardState___redArg(x_14, x_15, x_16, x_2);
lean_dec(x_15);
return x_17;
}
else
{
uint8_t x_18; 
x_18 = !lean_is_exclusive(x_12);
if (x_18 == 0)
{
return x_12;
}
else
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_12, 0);
lean_inc(x_19);
lean_dec(x_12);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateForwardState(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_updateForwardState___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateForwardState___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_updateForwardState(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateForwardState___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_updateForwardState___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_eraseForwardRuleMatch___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_st_ref_take(x_2);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_4, 1);
x_7 = lp_aesop_Aesop_ForwardRuleMatches_erase(x_1, x_6);
lean_ctor_set(x_4, 1, x_7);
x_8 = lean_st_ref_set(x_2, x_4);
x_9 = lean_box(0);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_11 = lean_ctor_get(x_4, 0);
x_12 = lean_ctor_get(x_4, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_4);
x_13 = lp_aesop_Aesop_ForwardRuleMatches_erase(x_1, x_12);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_11);
lean_ctor_set(x_14, 1, x_13);
x_15 = lean_st_ref_set(x_2, x_14);
x_16 = lean_box(0);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_eraseForwardRuleMatch(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_eraseForwardRuleMatch___redArg(x_1, x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_eraseForwardRuleMatch___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_eraseForwardRuleMatch(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_eraseForwardRuleMatch___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_eraseForwardRuleMatch___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyDiffToForwardState(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lp_aesop_Aesop_getResetForwardState___redArg(x_3);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_12);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
x_13 = lp_aesop_Aesop_ForwardState_applyGoalDiff(x_12, x_1, x_11, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lean_ctor_get(x_1, 3);
lean_inc_ref(x_15);
lean_dec_ref(x_1);
x_16 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_17 = lp_aesop_Aesop_modifyForwardState___redArg(x_14, x_16, x_15, x_3);
return x_17;
}
else
{
uint8_t x_18; 
lean_dec_ref(x_1);
x_18 = !lean_is_exclusive(x_13);
if (x_18 == 0)
{
return x_13;
}
else
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_13, 0);
lean_inc(x_19);
lean_dec(x_13);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyDiffToForwardState___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_applyDiffToForwardState(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorIdx(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; 
x_3 = lean_unsigned_to_nat(1u);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorIdx___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NormRuleResult_ctorIdx(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_2, x_3, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_1(x_2, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_NormRuleResult_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_NormRuleResult_ctorElim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_succeeded_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_NormRuleResult_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_succeeded_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NormRuleResult_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_proved_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_NormRuleResult_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_proved_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NormRuleResult_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_newGoal_x3f(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
else
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_newGoal_x3f___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_steps_x3f(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
else
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_steps_x3f___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NormRuleResult_steps_x3f(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_optNormRuleResultEmoji___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_ruleFailureEmoji;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_optNormRuleResultEmoji___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_ruleSuccessEmoji;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_optNormRuleResultEmoji___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_ruleProvedEmoji;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optNormRuleResultEmoji(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_optNormRuleResultEmoji___closed__0;
return x_2;
}
else
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_1, 0);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_optNormRuleResultEmoji___closed__1;
return x_4;
}
else
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_optNormRuleResultEmoji___closed__2;
return x_5;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optNormRuleResultEmoji___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_optNormRuleResultEmoji(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_optNormRuleResultEmoji___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" ", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__1;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("global", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("local", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("|", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("apply", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cases", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("constructors", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("destruct", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("forward", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("simp", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("unfold", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("norm", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("safe", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("unsafe", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("<norm simp>", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("<norm unfold>", 13, 13);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__0;
x_5 = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(x_4, x_2);
x_6 = l_Lean_stringToMessageData(x_5);
x_7 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__2;
x_8 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
switch (lean_obj_tag(x_1)) {
case 0:
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; uint8_t x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_37; 
x_15 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_15);
lean_dec_ref(x_1);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
x_17 = lean_ctor_get_uint8(x_15, sizeof(void*)*1 + 8);
x_18 = lean_ctor_get_uint8(x_15, sizeof(void*)*1 + 9);
x_19 = lean_ctor_get_uint8(x_15, sizeof(void*)*1 + 10);
lean_dec_ref(x_15);
switch (x_18) {
case 0:
{
lean_object* x_49; 
x_49 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__14;
x_37 = x_49;
goto block_48;
}
case 1:
{
lean_object* x_50; 
x_50 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__15;
x_37 = x_50;
goto block_48;
}
default: 
{
lean_object* x_51; 
x_51 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__16;
x_37 = x_51;
goto block_48;
}
}
block_28:
{
lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; 
x_23 = lean_string_append(x_20, x_22);
lean_dec_ref(x_22);
x_24 = lean_string_append(x_23, x_21);
lean_dec_ref(x_21);
x_25 = 1;
x_26 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_16, x_25);
x_27 = lean_string_append(x_24, x_26);
lean_dec_ref(x_26);
x_9 = x_27;
goto block_14;
}
block_36:
{
lean_object* x_32; lean_object* x_33; 
x_32 = lean_string_append(x_29, x_31);
lean_dec_ref(x_31);
x_33 = lean_string_append(x_32, x_30);
if (x_19 == 0)
{
lean_object* x_34; 
x_34 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__3;
x_20 = x_33;
x_21 = x_30;
x_22 = x_34;
goto block_28;
}
else
{
lean_object* x_35; 
x_35 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__4;
x_20 = x_33;
x_21 = x_30;
x_22 = x_35;
goto block_28;
}
}
block_48:
{
lean_object* x_38; lean_object* x_39; 
x_38 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__5;
x_39 = lean_string_append(x_37, x_38);
switch (x_17) {
case 0:
{
lean_object* x_40; 
x_40 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__6;
x_29 = x_39;
x_30 = x_38;
x_31 = x_40;
goto block_36;
}
case 1:
{
lean_object* x_41; 
x_41 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__7;
x_29 = x_39;
x_30 = x_38;
x_31 = x_41;
goto block_36;
}
case 2:
{
lean_object* x_42; 
x_42 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__8;
x_29 = x_39;
x_30 = x_38;
x_31 = x_42;
goto block_36;
}
case 3:
{
lean_object* x_43; 
x_43 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__9;
x_29 = x_39;
x_30 = x_38;
x_31 = x_43;
goto block_36;
}
case 4:
{
lean_object* x_44; 
x_44 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__10;
x_29 = x_39;
x_30 = x_38;
x_31 = x_44;
goto block_36;
}
case 5:
{
lean_object* x_45; 
x_45 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__11;
x_29 = x_39;
x_30 = x_38;
x_31 = x_45;
goto block_36;
}
case 6:
{
lean_object* x_46; 
x_46 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__12;
x_29 = x_39;
x_30 = x_38;
x_31 = x_46;
goto block_36;
}
default: 
{
lean_object* x_47; 
x_47 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__13;
x_29 = x_39;
x_30 = x_38;
x_31 = x_47;
goto block_36;
}
}
}
}
case 1:
{
lean_object* x_52; 
x_52 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__17;
x_9 = x_52;
goto block_14;
}
default: 
{
lean_object* x_53; 
x_53 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__18;
x_9 = x_53;
goto block_14;
}
}
block_14:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_10, 0, x_9);
x_11 = l_Lean_MessageData_ofFormat(x_10);
x_12 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_12, 0, x_8);
lean_ctor_set(x_12, 1, x_11);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg(x_1, x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withNormTraceNode___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_16; 
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_16 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_16, 0, x_7);
return x_16;
}
else
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_7, 0);
x_18 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_17);
if (lean_obj_tag(x_18) == 1)
{
uint8_t x_19; 
x_19 = !lean_is_exclusive(x_18);
if (x_19 == 0)
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_20 = lean_ctor_get(x_18, 0);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_21 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_1, x_2, x_3);
lean_inc(x_14);
lean_inc_ref(x_13);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
x_22 = lean_apply_8(x_21, x_8, x_9, x_10, x_11, x_12, x_13, x_14, lean_box(0));
if (lean_obj_tag(x_22) == 0)
{
uint8_t x_23; 
x_23 = !lean_is_exclusive(x_22);
if (x_23 == 0)
{
lean_object* x_24; uint8_t x_25; 
x_24 = lean_ctor_get(x_22, 0);
x_25 = lean_unbox(x_24);
lean_dec(x_24);
if (x_25 == 0)
{
lean_free_object(x_18);
lean_dec(x_20);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
lean_ctor_set(x_22, 0, x_7);
return x_22;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; 
lean_free_object(x_22);
x_26 = lean_ctor_get(x_3, 0);
lean_inc(x_26);
lean_dec_ref(x_3);
x_27 = l_Lean_addTrace___redArg(x_1, x_4, x_5, x_6, x_26, x_18);
x_28 = lean_apply_8(x_27, x_8, x_9, x_10, x_11, x_12, x_13, x_14, lean_box(0));
if (lean_obj_tag(x_28) == 0)
{
uint8_t x_29; 
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; 
x_30 = lean_ctor_get(x_28, 0);
lean_dec(x_30);
lean_ctor_set(x_28, 0, x_7);
return x_28;
}
else
{
lean_object* x_31; 
lean_dec(x_28);
x_31 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_31, 0, x_7);
return x_31;
}
}
else
{
uint8_t x_32; 
lean_dec_ref(x_7);
x_32 = !lean_is_exclusive(x_28);
if (x_32 == 0)
{
return x_28;
}
else
{
lean_object* x_33; lean_object* x_34; 
x_33 = lean_ctor_get(x_28, 0);
lean_inc(x_33);
lean_dec(x_28);
x_34 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_34, 0, x_33);
return x_34;
}
}
}
}
else
{
lean_object* x_35; uint8_t x_36; 
x_35 = lean_ctor_get(x_22, 0);
lean_inc(x_35);
lean_dec(x_22);
x_36 = lean_unbox(x_35);
lean_dec(x_35);
if (x_36 == 0)
{
lean_object* x_37; 
lean_free_object(x_18);
lean_dec(x_20);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_37 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_37, 0, x_7);
return x_37;
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_38 = lean_ctor_get(x_3, 0);
lean_inc(x_38);
lean_dec_ref(x_3);
x_39 = l_Lean_addTrace___redArg(x_1, x_4, x_5, x_6, x_38, x_18);
x_40 = lean_apply_8(x_39, x_8, x_9, x_10, x_11, x_12, x_13, x_14, lean_box(0));
if (lean_obj_tag(x_40) == 0)
{
lean_object* x_41; lean_object* x_42; 
if (lean_is_exclusive(x_40)) {
 lean_ctor_release(x_40, 0);
 x_41 = x_40;
} else {
 lean_dec_ref(x_40);
 x_41 = lean_box(0);
}
if (lean_is_scalar(x_41)) {
 x_42 = lean_alloc_ctor(0, 1, 0);
} else {
 x_42 = x_41;
}
lean_ctor_set(x_42, 0, x_7);
return x_42;
}
else
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; 
lean_dec_ref(x_7);
x_43 = lean_ctor_get(x_40, 0);
lean_inc(x_43);
if (lean_is_exclusive(x_40)) {
 lean_ctor_release(x_40, 0);
 x_44 = x_40;
} else {
 lean_dec_ref(x_40);
 x_44 = lean_box(0);
}
if (lean_is_scalar(x_44)) {
 x_45 = lean_alloc_ctor(1, 1, 0);
} else {
 x_45 = x_44;
}
lean_ctor_set(x_45, 0, x_43);
return x_45;
}
}
}
}
else
{
uint8_t x_46; 
lean_free_object(x_18);
lean_dec(x_20);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_46 = !lean_is_exclusive(x_22);
if (x_46 == 0)
{
return x_22;
}
else
{
lean_object* x_47; lean_object* x_48; 
x_47 = lean_ctor_get(x_22, 0);
lean_inc(x_47);
lean_dec(x_22);
x_48 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_48, 0, x_47);
return x_48;
}
}
}
else
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; 
x_49 = lean_ctor_get(x_18, 0);
lean_inc(x_49);
lean_dec(x_18);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_50 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_1, x_2, x_3);
lean_inc(x_14);
lean_inc_ref(x_13);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
x_51 = lean_apply_8(x_50, x_8, x_9, x_10, x_11, x_12, x_13, x_14, lean_box(0));
if (lean_obj_tag(x_51) == 0)
{
lean_object* x_52; lean_object* x_53; uint8_t x_54; 
x_52 = lean_ctor_get(x_51, 0);
lean_inc(x_52);
if (lean_is_exclusive(x_51)) {
 lean_ctor_release(x_51, 0);
 x_53 = x_51;
} else {
 lean_dec_ref(x_51);
 x_53 = lean_box(0);
}
x_54 = lean_unbox(x_52);
lean_dec(x_52);
if (x_54 == 0)
{
lean_object* x_55; 
lean_dec(x_49);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_53)) {
 x_55 = lean_alloc_ctor(0, 1, 0);
} else {
 x_55 = x_53;
}
lean_ctor_set(x_55, 0, x_7);
return x_55;
}
else
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; 
lean_dec(x_53);
x_56 = lean_ctor_get(x_3, 0);
lean_inc(x_56);
lean_dec_ref(x_3);
x_57 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_57, 0, x_49);
x_58 = l_Lean_addTrace___redArg(x_1, x_4, x_5, x_6, x_56, x_57);
x_59 = lean_apply_8(x_58, x_8, x_9, x_10, x_11, x_12, x_13, x_14, lean_box(0));
if (lean_obj_tag(x_59) == 0)
{
lean_object* x_60; lean_object* x_61; 
if (lean_is_exclusive(x_59)) {
 lean_ctor_release(x_59, 0);
 x_60 = x_59;
} else {
 lean_dec_ref(x_59);
 x_60 = lean_box(0);
}
if (lean_is_scalar(x_60)) {
 x_61 = lean_alloc_ctor(0, 1, 0);
} else {
 x_61 = x_60;
}
lean_ctor_set(x_61, 0, x_7);
return x_61;
}
else
{
lean_object* x_62; lean_object* x_63; lean_object* x_64; 
lean_dec_ref(x_7);
x_62 = lean_ctor_get(x_59, 0);
lean_inc(x_62);
if (lean_is_exclusive(x_59)) {
 lean_ctor_release(x_59, 0);
 x_63 = x_59;
} else {
 lean_dec_ref(x_59);
 x_63 = lean_box(0);
}
if (lean_is_scalar(x_63)) {
 x_64 = lean_alloc_ctor(1, 1, 0);
} else {
 x_64 = x_63;
}
lean_ctor_set(x_64, 0, x_62);
return x_64;
}
}
}
else
{
lean_object* x_65; lean_object* x_66; lean_object* x_67; 
lean_dec(x_49);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_65 = lean_ctor_get(x_51, 0);
lean_inc(x_65);
if (lean_is_exclusive(x_51)) {
 lean_ctor_release(x_51, 0);
 x_66 = x_51;
} else {
 lean_dec_ref(x_51);
 x_66 = lean_box(0);
}
if (lean_is_scalar(x_66)) {
 x_67 = lean_alloc_ctor(1, 1, 0);
} else {
 x_67 = x_66;
}
lean_ctor_set(x_67, 0, x_65);
return x_67;
}
}
}
else
{
lean_object* x_68; 
lean_dec(x_18);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_68 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_68, 0, x_7);
return x_68;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_instMonadEST(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__0;
x_2 = l_ReaderT_instMonad___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_instMonadCoreM___lam__0___boxed), 5, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_instMonadCoreM___lam__1___boxed), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instMonadMetaM___lam__0___boxed), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instMonadMetaM___lam__1___boxed), 9, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 3);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
lean_closure_set(x_1, 2, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = l_Lean_Core_instMonadTraceCoreM;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__8;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__9;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__10;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__11;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_ReaderT_instMonadFunctor___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = l_Lean_Core_instMonadQuotationCoreM;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_3 = lp_aesop_Aesop_withNormTraceNode___closed__13;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__14;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_3 = lp_aesop_Aesop_withNormTraceNode___closed__13;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__15;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_3 = lp_aesop_Aesop_withNormTraceNode___closed__13;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__16;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_3 = lp_aesop_Aesop_withNormTraceNode___closed__13;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__17;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_3 = lp_aesop_Aesop_withNormTraceNode___closed__13;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_BaseM_instMonadStats;
x_2 = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__19;
x_2 = lp_aesop_Aesop_instMonadStatsReaderT___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__21() {
_start:
{
lean_object* x_1; 
x_1 = l_instMonadExceptOfEST(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__21;
x_2 = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__23() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__22;
x_2 = l_Lean_instMonadAlwaysExceptReaderT___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__24() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__23;
x_2 = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__25() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__24;
x_2 = l_Lean_instMonadAlwaysExceptReaderT___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__26() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__25;
x_2 = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__27() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__26;
x_2 = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__28() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__27;
x_2 = l_Lean_instMonadAlwaysExceptReaderT___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__29() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_TraceOption_steps;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__30() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_liftIOCore___boxed), 5, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__31() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadLiftBaseIOEIO___lam__0___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__32() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadLiftT___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__33() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__31;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__32;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__34() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__30;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__33;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__35() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__34;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__36() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__35;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__37() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__36;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__38() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__37;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__39() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__38;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__40() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = l_Lean_Meta_instAddMessageContextMetaM;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_3 = l_Lean_instAddMessageContextOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__41() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__40;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_3 = l_Lean_instAddMessageContextOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__42() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__41;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_3 = l_Lean_instAddMessageContextOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_withNormTraceNode___closed__43() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("", 0, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withNormTraceNode___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_aesop_Aesop_withNormTraceNode___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withNormTraceNode(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; uint8_t x_12; 
x_11 = lp_aesop_Aesop_withNormTraceNode___closed__1;
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_13 = lean_ctor_get(x_11, 0);
x_14 = lean_ctor_get(x_11, 1);
lean_dec(x_14);
x_15 = !lean_is_exclusive(x_13);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; uint8_t x_30; 
x_16 = lean_ctor_get(x_13, 0);
x_17 = lean_ctor_get(x_13, 2);
x_18 = lean_ctor_get(x_13, 3);
x_19 = lean_ctor_get(x_13, 4);
x_20 = lean_ctor_get(x_13, 1);
lean_dec(x_20);
x_21 = lp_aesop_Aesop_withNormTraceNode___closed__2;
x_22 = lp_aesop_Aesop_withNormTraceNode___closed__3;
lean_inc_ref(x_16);
x_23 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_23, 0, x_16);
x_24 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_24, 0, x_16);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_23);
lean_ctor_set(x_25, 1, x_24);
x_26 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_26, 0, x_19);
x_27 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_27, 0, x_18);
x_28 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_28, 0, x_17);
lean_ctor_set(x_13, 4, x_26);
lean_ctor_set(x_13, 3, x_27);
lean_ctor_set(x_13, 2, x_28);
lean_ctor_set(x_13, 1, x_21);
lean_ctor_set(x_13, 0, x_25);
lean_ctor_set(x_11, 1, x_22);
x_29 = l_ReaderT_instMonad___redArg(x_11);
x_30 = !lean_is_exclusive(x_29);
if (x_30 == 0)
{
lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_31 = lean_ctor_get(x_29, 0);
x_32 = lean_ctor_get(x_29, 1);
lean_dec(x_32);
x_33 = !lean_is_exclusive(x_31);
if (x_33 == 0)
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; uint8_t x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; 
x_34 = lean_ctor_get(x_31, 0);
x_35 = lean_ctor_get(x_31, 2);
x_36 = lean_ctor_get(x_31, 3);
x_37 = lean_ctor_get(x_31, 4);
x_38 = lean_ctor_get(x_31, 1);
lean_dec(x_38);
x_39 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_40 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_34);
x_41 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_41, 0, x_34);
x_42 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_42, 0, x_34);
x_43 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_43, 0, x_41);
lean_ctor_set(x_43, 1, x_42);
x_44 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_44, 0, x_37);
x_45 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_45, 0, x_36);
x_46 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_46, 0, x_35);
lean_ctor_set(x_31, 4, x_44);
lean_ctor_set(x_31, 3, x_45);
lean_ctor_set(x_31, 2, x_46);
lean_ctor_set(x_31, 1, x_39);
lean_ctor_set(x_31, 0, x_43);
lean_ctor_set(x_29, 1, x_40);
x_47 = l_ReaderT_instMonad___redArg(x_29);
x_48 = l_ReaderT_instMonad___redArg(x_47);
lean_inc_ref(x_48);
x_49 = l_ReaderT_instMonad___redArg(x_48);
x_50 = lp_aesop_Aesop_withNormTraceNode___closed__12;
x_51 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_52 = lean_ctor_get(x_51, 0);
lean_inc_ref(x_52);
x_53 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_54 = lean_ctor_get(x_53, 0);
lean_inc(x_54);
x_55 = lp_aesop_Aesop_withNormTraceNode___closed__28;
x_56 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_57 = lean_ctor_get(x_56, 0);
lean_inc(x_57);
x_58 = lp_aesop_Aesop_withNormTraceNode___closed__39;
x_59 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_52);
lean_inc(x_54);
lean_inc_ref(x_49);
x_60 = lean_alloc_closure((void*)(lp_aesop_Aesop_withNormTraceNode___lam__0___boxed), 15, 6);
lean_closure_set(x_60, 0, x_49);
lean_closure_set(x_60, 1, x_54);
lean_closure_set(x_60, 2, x_56);
lean_closure_set(x_60, 3, x_50);
lean_closure_set(x_60, 4, x_52);
lean_closure_set(x_60, 5, x_59);
x_61 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed), 10, 1);
lean_closure_set(x_61, 0, x_1);
x_62 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_62, 0, lean_box(0));
lean_closure_set(x_62, 1, lean_box(0));
lean_closure_set(x_62, 2, x_48);
lean_closure_set(x_62, 3, lean_box(0));
lean_closure_set(x_62, 4, lean_box(0));
lean_closure_set(x_62, 5, x_2);
lean_closure_set(x_62, 6, x_60);
x_63 = 1;
x_64 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_65 = l_Lean_withTraceNode___redArg(x_49, x_50, x_52, x_59, x_54, x_55, x_58, x_57, x_61, x_62, x_63, x_64);
x_66 = lean_apply_8(x_65, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_66;
}
else
{
lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; uint8_t x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; 
x_67 = lean_ctor_get(x_31, 0);
x_68 = lean_ctor_get(x_31, 2);
x_69 = lean_ctor_get(x_31, 3);
x_70 = lean_ctor_get(x_31, 4);
lean_inc(x_70);
lean_inc(x_69);
lean_inc(x_68);
lean_inc(x_67);
lean_dec(x_31);
x_71 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_72 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_67);
x_73 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_73, 0, x_67);
x_74 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_74, 0, x_67);
x_75 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_75, 0, x_73);
lean_ctor_set(x_75, 1, x_74);
x_76 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_76, 0, x_70);
x_77 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_77, 0, x_69);
x_78 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_78, 0, x_68);
x_79 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_79, 0, x_75);
lean_ctor_set(x_79, 1, x_71);
lean_ctor_set(x_79, 2, x_78);
lean_ctor_set(x_79, 3, x_77);
lean_ctor_set(x_79, 4, x_76);
lean_ctor_set(x_29, 1, x_72);
lean_ctor_set(x_29, 0, x_79);
x_80 = l_ReaderT_instMonad___redArg(x_29);
x_81 = l_ReaderT_instMonad___redArg(x_80);
lean_inc_ref(x_81);
x_82 = l_ReaderT_instMonad___redArg(x_81);
x_83 = lp_aesop_Aesop_withNormTraceNode___closed__12;
x_84 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_85 = lean_ctor_get(x_84, 0);
lean_inc_ref(x_85);
x_86 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_87 = lean_ctor_get(x_86, 0);
lean_inc(x_87);
x_88 = lp_aesop_Aesop_withNormTraceNode___closed__28;
x_89 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_90 = lean_ctor_get(x_89, 0);
lean_inc(x_90);
x_91 = lp_aesop_Aesop_withNormTraceNode___closed__39;
x_92 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_85);
lean_inc(x_87);
lean_inc_ref(x_82);
x_93 = lean_alloc_closure((void*)(lp_aesop_Aesop_withNormTraceNode___lam__0___boxed), 15, 6);
lean_closure_set(x_93, 0, x_82);
lean_closure_set(x_93, 1, x_87);
lean_closure_set(x_93, 2, x_89);
lean_closure_set(x_93, 3, x_83);
lean_closure_set(x_93, 4, x_85);
lean_closure_set(x_93, 5, x_92);
x_94 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed), 10, 1);
lean_closure_set(x_94, 0, x_1);
x_95 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_95, 0, lean_box(0));
lean_closure_set(x_95, 1, lean_box(0));
lean_closure_set(x_95, 2, x_81);
lean_closure_set(x_95, 3, lean_box(0));
lean_closure_set(x_95, 4, lean_box(0));
lean_closure_set(x_95, 5, x_2);
lean_closure_set(x_95, 6, x_93);
x_96 = 1;
x_97 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_98 = l_Lean_withTraceNode___redArg(x_82, x_83, x_85, x_92, x_87, x_88, x_91, x_90, x_94, x_95, x_96, x_97);
x_99 = lean_apply_8(x_98, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_99;
}
}
else
{
lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; uint8_t x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; 
x_100 = lean_ctor_get(x_29, 0);
lean_inc(x_100);
lean_dec(x_29);
x_101 = lean_ctor_get(x_100, 0);
lean_inc_ref(x_101);
x_102 = lean_ctor_get(x_100, 2);
lean_inc(x_102);
x_103 = lean_ctor_get(x_100, 3);
lean_inc(x_103);
x_104 = lean_ctor_get(x_100, 4);
lean_inc(x_104);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 lean_ctor_release(x_100, 1);
 lean_ctor_release(x_100, 2);
 lean_ctor_release(x_100, 3);
 lean_ctor_release(x_100, 4);
 x_105 = x_100;
} else {
 lean_dec_ref(x_100);
 x_105 = lean_box(0);
}
x_106 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_107 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_101);
x_108 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_108, 0, x_101);
x_109 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_109, 0, x_101);
x_110 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_110, 0, x_108);
lean_ctor_set(x_110, 1, x_109);
x_111 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_111, 0, x_104);
x_112 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_112, 0, x_103);
x_113 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_113, 0, x_102);
if (lean_is_scalar(x_105)) {
 x_114 = lean_alloc_ctor(0, 5, 0);
} else {
 x_114 = x_105;
}
lean_ctor_set(x_114, 0, x_110);
lean_ctor_set(x_114, 1, x_106);
lean_ctor_set(x_114, 2, x_113);
lean_ctor_set(x_114, 3, x_112);
lean_ctor_set(x_114, 4, x_111);
x_115 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_115, 0, x_114);
lean_ctor_set(x_115, 1, x_107);
x_116 = l_ReaderT_instMonad___redArg(x_115);
x_117 = l_ReaderT_instMonad___redArg(x_116);
lean_inc_ref(x_117);
x_118 = l_ReaderT_instMonad___redArg(x_117);
x_119 = lp_aesop_Aesop_withNormTraceNode___closed__12;
x_120 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_121 = lean_ctor_get(x_120, 0);
lean_inc_ref(x_121);
x_122 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_123 = lean_ctor_get(x_122, 0);
lean_inc(x_123);
x_124 = lp_aesop_Aesop_withNormTraceNode___closed__28;
x_125 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_126 = lean_ctor_get(x_125, 0);
lean_inc(x_126);
x_127 = lp_aesop_Aesop_withNormTraceNode___closed__39;
x_128 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_121);
lean_inc(x_123);
lean_inc_ref(x_118);
x_129 = lean_alloc_closure((void*)(lp_aesop_Aesop_withNormTraceNode___lam__0___boxed), 15, 6);
lean_closure_set(x_129, 0, x_118);
lean_closure_set(x_129, 1, x_123);
lean_closure_set(x_129, 2, x_125);
lean_closure_set(x_129, 3, x_119);
lean_closure_set(x_129, 4, x_121);
lean_closure_set(x_129, 5, x_128);
x_130 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed), 10, 1);
lean_closure_set(x_130, 0, x_1);
x_131 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_131, 0, lean_box(0));
lean_closure_set(x_131, 1, lean_box(0));
lean_closure_set(x_131, 2, x_117);
lean_closure_set(x_131, 3, lean_box(0));
lean_closure_set(x_131, 4, lean_box(0));
lean_closure_set(x_131, 5, x_2);
lean_closure_set(x_131, 6, x_129);
x_132 = 1;
x_133 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_134 = l_Lean_withTraceNode___redArg(x_118, x_119, x_121, x_128, x_123, x_124, x_127, x_126, x_130, x_131, x_132, x_133);
x_135 = lean_apply_8(x_134, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_135;
}
}
else
{
lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; uint8_t x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; 
x_136 = lean_ctor_get(x_13, 0);
x_137 = lean_ctor_get(x_13, 2);
x_138 = lean_ctor_get(x_13, 3);
x_139 = lean_ctor_get(x_13, 4);
lean_inc(x_139);
lean_inc(x_138);
lean_inc(x_137);
lean_inc(x_136);
lean_dec(x_13);
x_140 = lp_aesop_Aesop_withNormTraceNode___closed__2;
x_141 = lp_aesop_Aesop_withNormTraceNode___closed__3;
lean_inc_ref(x_136);
x_142 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_142, 0, x_136);
x_143 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_143, 0, x_136);
x_144 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_144, 0, x_142);
lean_ctor_set(x_144, 1, x_143);
x_145 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_145, 0, x_139);
x_146 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_146, 0, x_138);
x_147 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_147, 0, x_137);
x_148 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_148, 0, x_144);
lean_ctor_set(x_148, 1, x_140);
lean_ctor_set(x_148, 2, x_147);
lean_ctor_set(x_148, 3, x_146);
lean_ctor_set(x_148, 4, x_145);
lean_ctor_set(x_11, 1, x_141);
lean_ctor_set(x_11, 0, x_148);
x_149 = l_ReaderT_instMonad___redArg(x_11);
x_150 = lean_ctor_get(x_149, 0);
lean_inc_ref(x_150);
if (lean_is_exclusive(x_149)) {
 lean_ctor_release(x_149, 0);
 lean_ctor_release(x_149, 1);
 x_151 = x_149;
} else {
 lean_dec_ref(x_149);
 x_151 = lean_box(0);
}
x_152 = lean_ctor_get(x_150, 0);
lean_inc_ref(x_152);
x_153 = lean_ctor_get(x_150, 2);
lean_inc(x_153);
x_154 = lean_ctor_get(x_150, 3);
lean_inc(x_154);
x_155 = lean_ctor_get(x_150, 4);
lean_inc(x_155);
if (lean_is_exclusive(x_150)) {
 lean_ctor_release(x_150, 0);
 lean_ctor_release(x_150, 1);
 lean_ctor_release(x_150, 2);
 lean_ctor_release(x_150, 3);
 lean_ctor_release(x_150, 4);
 x_156 = x_150;
} else {
 lean_dec_ref(x_150);
 x_156 = lean_box(0);
}
x_157 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_158 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_152);
x_159 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_159, 0, x_152);
x_160 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_160, 0, x_152);
x_161 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_161, 0, x_159);
lean_ctor_set(x_161, 1, x_160);
x_162 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_162, 0, x_155);
x_163 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_163, 0, x_154);
x_164 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_164, 0, x_153);
if (lean_is_scalar(x_156)) {
 x_165 = lean_alloc_ctor(0, 5, 0);
} else {
 x_165 = x_156;
}
lean_ctor_set(x_165, 0, x_161);
lean_ctor_set(x_165, 1, x_157);
lean_ctor_set(x_165, 2, x_164);
lean_ctor_set(x_165, 3, x_163);
lean_ctor_set(x_165, 4, x_162);
if (lean_is_scalar(x_151)) {
 x_166 = lean_alloc_ctor(0, 2, 0);
} else {
 x_166 = x_151;
}
lean_ctor_set(x_166, 0, x_165);
lean_ctor_set(x_166, 1, x_158);
x_167 = l_ReaderT_instMonad___redArg(x_166);
x_168 = l_ReaderT_instMonad___redArg(x_167);
lean_inc_ref(x_168);
x_169 = l_ReaderT_instMonad___redArg(x_168);
x_170 = lp_aesop_Aesop_withNormTraceNode___closed__12;
x_171 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_172 = lean_ctor_get(x_171, 0);
lean_inc_ref(x_172);
x_173 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_174 = lean_ctor_get(x_173, 0);
lean_inc(x_174);
x_175 = lp_aesop_Aesop_withNormTraceNode___closed__28;
x_176 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_177 = lean_ctor_get(x_176, 0);
lean_inc(x_177);
x_178 = lp_aesop_Aesop_withNormTraceNode___closed__39;
x_179 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_172);
lean_inc(x_174);
lean_inc_ref(x_169);
x_180 = lean_alloc_closure((void*)(lp_aesop_Aesop_withNormTraceNode___lam__0___boxed), 15, 6);
lean_closure_set(x_180, 0, x_169);
lean_closure_set(x_180, 1, x_174);
lean_closure_set(x_180, 2, x_176);
lean_closure_set(x_180, 3, x_170);
lean_closure_set(x_180, 4, x_172);
lean_closure_set(x_180, 5, x_179);
x_181 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed), 10, 1);
lean_closure_set(x_181, 0, x_1);
x_182 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_182, 0, lean_box(0));
lean_closure_set(x_182, 1, lean_box(0));
lean_closure_set(x_182, 2, x_168);
lean_closure_set(x_182, 3, lean_box(0));
lean_closure_set(x_182, 4, lean_box(0));
lean_closure_set(x_182, 5, x_2);
lean_closure_set(x_182, 6, x_180);
x_183 = 1;
x_184 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_185 = l_Lean_withTraceNode___redArg(x_169, x_170, x_172, x_179, x_174, x_175, x_178, x_177, x_181, x_182, x_183, x_184);
x_186 = lean_apply_8(x_185, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_186;
}
}
else
{
lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; uint8_t x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; 
x_187 = lean_ctor_get(x_11, 0);
lean_inc(x_187);
lean_dec(x_11);
x_188 = lean_ctor_get(x_187, 0);
lean_inc_ref(x_188);
x_189 = lean_ctor_get(x_187, 2);
lean_inc(x_189);
x_190 = lean_ctor_get(x_187, 3);
lean_inc(x_190);
x_191 = lean_ctor_get(x_187, 4);
lean_inc(x_191);
if (lean_is_exclusive(x_187)) {
 lean_ctor_release(x_187, 0);
 lean_ctor_release(x_187, 1);
 lean_ctor_release(x_187, 2);
 lean_ctor_release(x_187, 3);
 lean_ctor_release(x_187, 4);
 x_192 = x_187;
} else {
 lean_dec_ref(x_187);
 x_192 = lean_box(0);
}
x_193 = lp_aesop_Aesop_withNormTraceNode___closed__2;
x_194 = lp_aesop_Aesop_withNormTraceNode___closed__3;
lean_inc_ref(x_188);
x_195 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_195, 0, x_188);
x_196 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_196, 0, x_188);
x_197 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_197, 0, x_195);
lean_ctor_set(x_197, 1, x_196);
x_198 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_198, 0, x_191);
x_199 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_199, 0, x_190);
x_200 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_200, 0, x_189);
if (lean_is_scalar(x_192)) {
 x_201 = lean_alloc_ctor(0, 5, 0);
} else {
 x_201 = x_192;
}
lean_ctor_set(x_201, 0, x_197);
lean_ctor_set(x_201, 1, x_193);
lean_ctor_set(x_201, 2, x_200);
lean_ctor_set(x_201, 3, x_199);
lean_ctor_set(x_201, 4, x_198);
x_202 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_202, 0, x_201);
lean_ctor_set(x_202, 1, x_194);
x_203 = l_ReaderT_instMonad___redArg(x_202);
x_204 = lean_ctor_get(x_203, 0);
lean_inc_ref(x_204);
if (lean_is_exclusive(x_203)) {
 lean_ctor_release(x_203, 0);
 lean_ctor_release(x_203, 1);
 x_205 = x_203;
} else {
 lean_dec_ref(x_203);
 x_205 = lean_box(0);
}
x_206 = lean_ctor_get(x_204, 0);
lean_inc_ref(x_206);
x_207 = lean_ctor_get(x_204, 2);
lean_inc(x_207);
x_208 = lean_ctor_get(x_204, 3);
lean_inc(x_208);
x_209 = lean_ctor_get(x_204, 4);
lean_inc(x_209);
if (lean_is_exclusive(x_204)) {
 lean_ctor_release(x_204, 0);
 lean_ctor_release(x_204, 1);
 lean_ctor_release(x_204, 2);
 lean_ctor_release(x_204, 3);
 lean_ctor_release(x_204, 4);
 x_210 = x_204;
} else {
 lean_dec_ref(x_204);
 x_210 = lean_box(0);
}
x_211 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_212 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_206);
x_213 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_213, 0, x_206);
x_214 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_214, 0, x_206);
x_215 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_215, 0, x_213);
lean_ctor_set(x_215, 1, x_214);
x_216 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_216, 0, x_209);
x_217 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_217, 0, x_208);
x_218 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_218, 0, x_207);
if (lean_is_scalar(x_210)) {
 x_219 = lean_alloc_ctor(0, 5, 0);
} else {
 x_219 = x_210;
}
lean_ctor_set(x_219, 0, x_215);
lean_ctor_set(x_219, 1, x_211);
lean_ctor_set(x_219, 2, x_218);
lean_ctor_set(x_219, 3, x_217);
lean_ctor_set(x_219, 4, x_216);
if (lean_is_scalar(x_205)) {
 x_220 = lean_alloc_ctor(0, 2, 0);
} else {
 x_220 = x_205;
}
lean_ctor_set(x_220, 0, x_219);
lean_ctor_set(x_220, 1, x_212);
x_221 = l_ReaderT_instMonad___redArg(x_220);
x_222 = l_ReaderT_instMonad___redArg(x_221);
lean_inc_ref(x_222);
x_223 = l_ReaderT_instMonad___redArg(x_222);
x_224 = lp_aesop_Aesop_withNormTraceNode___closed__12;
x_225 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_226 = lean_ctor_get(x_225, 0);
lean_inc_ref(x_226);
x_227 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_228 = lean_ctor_get(x_227, 0);
lean_inc(x_228);
x_229 = lp_aesop_Aesop_withNormTraceNode___closed__28;
x_230 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_231 = lean_ctor_get(x_230, 0);
lean_inc(x_231);
x_232 = lp_aesop_Aesop_withNormTraceNode___closed__39;
x_233 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_226);
lean_inc(x_228);
lean_inc_ref(x_223);
x_234 = lean_alloc_closure((void*)(lp_aesop_Aesop_withNormTraceNode___lam__0___boxed), 15, 6);
lean_closure_set(x_234, 0, x_223);
lean_closure_set(x_234, 1, x_228);
lean_closure_set(x_234, 2, x_230);
lean_closure_set(x_234, 3, x_224);
lean_closure_set(x_234, 4, x_226);
lean_closure_set(x_234, 5, x_233);
x_235 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed), 10, 1);
lean_closure_set(x_235, 0, x_1);
x_236 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_236, 0, lean_box(0));
lean_closure_set(x_236, 1, lean_box(0));
lean_closure_set(x_236, 2, x_222);
lean_closure_set(x_236, 3, lean_box(0));
lean_closure_set(x_236, 4, lean_box(0));
lean_closure_set(x_236, 5, x_2);
lean_closure_set(x_236, 6, x_234);
x_237 = 1;
x_238 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_239 = l_Lean_withTraceNode___redArg(x_223, x_224, x_226, x_233, x_228, x_229, x_232, x_231, x_235, x_236, x_237, x_238);
x_240 = lean_apply_8(x_239, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_240;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withNormTraceNode___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_withNormTraceNode(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_7 = lean_st_ref_get(x_5);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec(x_7);
x_9 = lean_st_ref_get(x_3);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec(x_9);
x_11 = lean_ctor_get(x_2, 2);
x_12 = lean_ctor_get(x_4, 2);
lean_inc(x_12);
lean_inc_ref(x_11);
x_13 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_13, 0, x_8);
lean_ctor_set(x_13, 1, x_10);
lean_ctor_set(x_13, 2, x_11);
lean_ctor_set(x_13, 3, x_12);
x_14 = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_1);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_ctor_get(x_4, 5);
x_8 = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_7);
lean_ctor_set(x_11, 1, x_10);
lean_ctor_set_tag(x_8, 1);
lean_ctor_set(x_8, 0, x_11);
return x_8;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_8, 0);
lean_inc(x_12);
lean_dec(x_8);
lean_inc(x_7);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_7);
lean_ctor_set(x_13, 1, x_12);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0___redArg(x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: error while running norm rule ", 37, 37);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(": ", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("\nThe rule was run on this goal:", 31, 31);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__4;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; uint8_t x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_45; 
x_9 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_1);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
x_11 = lean_ctor_get_uint8(x_9, sizeof(void*)*1 + 8);
x_12 = lean_ctor_get_uint8(x_9, sizeof(void*)*1 + 9);
x_13 = lean_ctor_get_uint8(x_9, sizeof(void*)*1 + 10);
lean_dec_ref(x_9);
x_14 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__1;
switch (x_12) {
case 0:
{
lean_object* x_57; 
x_57 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__14;
x_45 = x_57;
goto block_56;
}
case 1:
{
lean_object* x_58; 
x_58 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__15;
x_45 = x_58;
goto block_56;
}
default: 
{
lean_object* x_59; 
x_59 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__16;
x_45 = x_59;
goto block_56;
}
}
block_36:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_18 = lean_ctor_get(x_2, 0);
x_19 = lean_string_append(x_16, x_17);
lean_dec_ref(x_17);
x_20 = lean_string_append(x_19, x_15);
lean_dec_ref(x_15);
x_21 = 1;
x_22 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_10, x_21);
x_23 = lean_string_append(x_20, x_22);
lean_dec_ref(x_22);
x_24 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_24, 0, x_23);
x_25 = l_Lean_MessageData_ofFormat(x_24);
x_26 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_26, 0, x_14);
lean_ctor_set(x_26, 1, x_25);
x_27 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_28 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_28, 0, x_26);
lean_ctor_set(x_28, 1, x_27);
x_29 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_3);
x_30 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__5;
x_31 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_31, 0, x_29);
lean_ctor_set(x_31, 1, x_30);
lean_inc(x_18);
x_32 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_32, 0, x_18);
x_33 = l_Lean_indentD(x_32);
x_34 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_34, 0, x_31);
lean_ctor_set(x_34, 1, x_33);
x_35 = lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0___redArg(x_34, x_4, x_5, x_6, x_7);
return x_35;
}
block_44:
{
lean_object* x_40; lean_object* x_41; 
x_40 = lean_string_append(x_37, x_39);
lean_dec_ref(x_39);
x_41 = lean_string_append(x_40, x_38);
if (x_13 == 0)
{
lean_object* x_42; 
x_42 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__3;
x_15 = x_38;
x_16 = x_41;
x_17 = x_42;
goto block_36;
}
else
{
lean_object* x_43; 
x_43 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__4;
x_15 = x_38;
x_16 = x_41;
x_17 = x_43;
goto block_36;
}
}
block_56:
{
lean_object* x_46; lean_object* x_47; 
x_46 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__5;
x_47 = lean_string_append(x_45, x_46);
switch (x_11) {
case 0:
{
lean_object* x_48; 
x_48 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__6;
x_37 = x_47;
x_38 = x_46;
x_39 = x_48;
goto block_44;
}
case 1:
{
lean_object* x_49; 
x_49 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__7;
x_37 = x_47;
x_38 = x_46;
x_39 = x_49;
goto block_44;
}
case 2:
{
lean_object* x_50; 
x_50 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__8;
x_37 = x_47;
x_38 = x_46;
x_39 = x_50;
goto block_44;
}
case 3:
{
lean_object* x_51; 
x_51 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__9;
x_37 = x_47;
x_38 = x_46;
x_39 = x_51;
goto block_44;
}
case 4:
{
lean_object* x_52; 
x_52 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__10;
x_37 = x_47;
x_38 = x_46;
x_39 = x_52;
goto block_44;
}
case 5:
{
lean_object* x_53; 
x_53 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__11;
x_37 = x_47;
x_38 = x_46;
x_39 = x_53;
goto block_44;
}
case 6:
{
lean_object* x_54; 
x_54 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__12;
x_37 = x_47;
x_38 = x_46;
x_39 = x_54;
goto block_44;
}
default: 
{
lean_object* x_55; 
x_55 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__13;
x_37 = x_47;
x_38 = x_46;
x_39 = x_55;
goto block_44;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(x_1, x_2, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; uint8_t x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 2);
x_5 = lp_aesop_Aesop_Check_get(x_4, x_1);
x_6 = lean_box(x_5);
x_7 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(x_1, x_7);
return x_10;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = l_Lean_KVMap_find(x_1, x_3);
if (lean_obj_tag(x_5) == 0)
{
uint8_t x_6; 
x_6 = lean_unbox(x_4);
return x_6;
}
else
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec_ref(x_5);
if (lean_obj_tag(x_7) == 1)
{
uint8_t x_8; 
x_8 = lean_ctor_get_uint8(x_7, 0);
lean_dec_ref(x_7);
return x_8;
}
else
{
uint8_t x_9; 
lean_dec(x_7);
x_9 = lean_unbox(x_4);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 2);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(x_4, x_5);
x_7 = lean_box(x_6);
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_1, x_7);
return x_10;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0() {
_start:
{
lean_object* x_1; double x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_float_of_nat(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_8 = lean_ctor_get(x_5, 5);
x_9 = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0(x_2, x_3, x_4, x_5, x_6);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_st_ref_take(x_6);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; uint8_t x_15; 
x_14 = lean_ctor_get(x_12, 4);
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; double x_17; uint8_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0;
x_18 = 0;
x_19 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_20 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_20, 0, x_1);
lean_ctor_set(x_20, 1, x_19);
lean_ctor_set_float(x_20, sizeof(void*)*2, x_17);
lean_ctor_set_float(x_20, sizeof(void*)*2 + 8, x_17);
lean_ctor_set_uint8(x_20, sizeof(void*)*2 + 16, x_18);
x_21 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__1;
x_22 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_22, 0, x_20);
lean_ctor_set(x_22, 1, x_11);
lean_ctor_set(x_22, 2, x_21);
lean_inc(x_8);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_8);
lean_ctor_set(x_23, 1, x_22);
x_24 = l_Lean_PersistentArray_push___redArg(x_16, x_23);
lean_ctor_set(x_14, 0, x_24);
x_25 = lean_st_ref_set(x_6, x_12);
x_26 = lean_box(0);
lean_ctor_set(x_9, 0, x_26);
return x_9;
}
else
{
uint64_t x_27; lean_object* x_28; double x_29; uint8_t x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_27 = lean_ctor_get_uint64(x_14, sizeof(void*)*1);
x_28 = lean_ctor_get(x_14, 0);
lean_inc(x_28);
lean_dec(x_14);
x_29 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0;
x_30 = 0;
x_31 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_32 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_32, 0, x_1);
lean_ctor_set(x_32, 1, x_31);
lean_ctor_set_float(x_32, sizeof(void*)*2, x_29);
lean_ctor_set_float(x_32, sizeof(void*)*2 + 8, x_29);
lean_ctor_set_uint8(x_32, sizeof(void*)*2 + 16, x_30);
x_33 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__1;
x_34 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_11);
lean_ctor_set(x_34, 2, x_33);
lean_inc(x_8);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_8);
lean_ctor_set(x_35, 1, x_34);
x_36 = l_Lean_PersistentArray_push___redArg(x_28, x_35);
x_37 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_37, 0, x_36);
lean_ctor_set_uint64(x_37, sizeof(void*)*1, x_27);
lean_ctor_set(x_12, 4, x_37);
x_38 = lean_st_ref_set(x_6, x_12);
x_39 = lean_box(0);
lean_ctor_set(x_9, 0, x_39);
return x_9;
}
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; uint64_t x_49; lean_object* x_50; lean_object* x_51; double x_52; uint8_t x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_40 = lean_ctor_get(x_12, 4);
x_41 = lean_ctor_get(x_12, 0);
x_42 = lean_ctor_get(x_12, 1);
x_43 = lean_ctor_get(x_12, 2);
x_44 = lean_ctor_get(x_12, 3);
x_45 = lean_ctor_get(x_12, 5);
x_46 = lean_ctor_get(x_12, 6);
x_47 = lean_ctor_get(x_12, 7);
x_48 = lean_ctor_get(x_12, 8);
lean_inc(x_48);
lean_inc(x_47);
lean_inc(x_46);
lean_inc(x_45);
lean_inc(x_40);
lean_inc(x_44);
lean_inc(x_43);
lean_inc(x_42);
lean_inc(x_41);
lean_dec(x_12);
x_49 = lean_ctor_get_uint64(x_40, sizeof(void*)*1);
x_50 = lean_ctor_get(x_40, 0);
lean_inc_ref(x_50);
if (lean_is_exclusive(x_40)) {
 lean_ctor_release(x_40, 0);
 x_51 = x_40;
} else {
 lean_dec_ref(x_40);
 x_51 = lean_box(0);
}
x_52 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0;
x_53 = 0;
x_54 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_55 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_55, 0, x_1);
lean_ctor_set(x_55, 1, x_54);
lean_ctor_set_float(x_55, sizeof(void*)*2, x_52);
lean_ctor_set_float(x_55, sizeof(void*)*2 + 8, x_52);
lean_ctor_set_uint8(x_55, sizeof(void*)*2 + 16, x_53);
x_56 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__1;
x_57 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_57, 0, x_55);
lean_ctor_set(x_57, 1, x_11);
lean_ctor_set(x_57, 2, x_56);
lean_inc(x_8);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_8);
lean_ctor_set(x_58, 1, x_57);
x_59 = l_Lean_PersistentArray_push___redArg(x_50, x_58);
if (lean_is_scalar(x_51)) {
 x_60 = lean_alloc_ctor(0, 1, 8);
} else {
 x_60 = x_51;
}
lean_ctor_set(x_60, 0, x_59);
lean_ctor_set_uint64(x_60, sizeof(void*)*1, x_49);
x_61 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_61, 0, x_41);
lean_ctor_set(x_61, 1, x_42);
lean_ctor_set(x_61, 2, x_43);
lean_ctor_set(x_61, 3, x_44);
lean_ctor_set(x_61, 4, x_60);
lean_ctor_set(x_61, 5, x_45);
lean_ctor_set(x_61, 6, x_46);
lean_ctor_set(x_61, 7, x_47);
lean_ctor_set(x_61, 8, x_48);
x_62 = lean_st_ref_set(x_6, x_61);
x_63 = lean_box(0);
lean_ctor_set(x_9, 0, x_63);
return x_9;
}
}
else
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; uint64_t x_76; lean_object* x_77; lean_object* x_78; double x_79; uint8_t x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; 
x_64 = lean_ctor_get(x_9, 0);
lean_inc(x_64);
lean_dec(x_9);
x_65 = lean_st_ref_take(x_6);
x_66 = lean_ctor_get(x_65, 4);
lean_inc_ref(x_66);
x_67 = lean_ctor_get(x_65, 0);
lean_inc_ref(x_67);
x_68 = lean_ctor_get(x_65, 1);
lean_inc(x_68);
x_69 = lean_ctor_get(x_65, 2);
lean_inc_ref(x_69);
x_70 = lean_ctor_get(x_65, 3);
lean_inc_ref(x_70);
x_71 = lean_ctor_get(x_65, 5);
lean_inc_ref(x_71);
x_72 = lean_ctor_get(x_65, 6);
lean_inc_ref(x_72);
x_73 = lean_ctor_get(x_65, 7);
lean_inc_ref(x_73);
x_74 = lean_ctor_get(x_65, 8);
lean_inc_ref(x_74);
if (lean_is_exclusive(x_65)) {
 lean_ctor_release(x_65, 0);
 lean_ctor_release(x_65, 1);
 lean_ctor_release(x_65, 2);
 lean_ctor_release(x_65, 3);
 lean_ctor_release(x_65, 4);
 lean_ctor_release(x_65, 5);
 lean_ctor_release(x_65, 6);
 lean_ctor_release(x_65, 7);
 lean_ctor_release(x_65, 8);
 x_75 = x_65;
} else {
 lean_dec_ref(x_65);
 x_75 = lean_box(0);
}
x_76 = lean_ctor_get_uint64(x_66, sizeof(void*)*1);
x_77 = lean_ctor_get(x_66, 0);
lean_inc_ref(x_77);
if (lean_is_exclusive(x_66)) {
 lean_ctor_release(x_66, 0);
 x_78 = x_66;
} else {
 lean_dec_ref(x_66);
 x_78 = lean_box(0);
}
x_79 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0;
x_80 = 0;
x_81 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_82 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_82, 0, x_1);
lean_ctor_set(x_82, 1, x_81);
lean_ctor_set_float(x_82, sizeof(void*)*2, x_79);
lean_ctor_set_float(x_82, sizeof(void*)*2 + 8, x_79);
lean_ctor_set_uint8(x_82, sizeof(void*)*2 + 16, x_80);
x_83 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__1;
x_84 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_84, 0, x_82);
lean_ctor_set(x_84, 1, x_64);
lean_ctor_set(x_84, 2, x_83);
lean_inc(x_8);
x_85 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_85, 0, x_8);
lean_ctor_set(x_85, 1, x_84);
x_86 = l_Lean_PersistentArray_push___redArg(x_77, x_85);
if (lean_is_scalar(x_78)) {
 x_87 = lean_alloc_ctor(0, 1, 8);
} else {
 x_87 = x_78;
}
lean_ctor_set(x_87, 0, x_86);
lean_ctor_set_uint64(x_87, sizeof(void*)*1, x_76);
if (lean_is_scalar(x_75)) {
 x_88 = lean_alloc_ctor(0, 9, 0);
} else {
 x_88 = x_75;
}
lean_ctor_set(x_88, 0, x_67);
lean_ctor_set(x_88, 1, x_68);
lean_ctor_set(x_88, 2, x_69);
lean_ctor_set(x_88, 3, x_70);
lean_ctor_set(x_88, 4, x_87);
lean_ctor_set(x_88, 5, x_71);
lean_ctor_set(x_88, 6, x_72);
lean_ctor_set(x_88, 7, x_73);
lean_ctor_set(x_88, 8, x_74);
x_89 = lean_st_ref_set(x_6, x_88);
x_90 = lean_box(0);
x_91 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_1, x_2, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 2);
x_6 = l_Lean_instBEqMVarId_beq(x_4, x_1);
if (x_6 == 0)
{
x_2 = x_5;
goto _start;
}
else
{
return x_6;
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5_spec__5___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
return x_1;
}
else
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint64_t x_7; uint64_t x_8; uint64_t x_9; uint64_t x_10; uint64_t x_11; uint64_t x_12; uint64_t x_13; size_t x_14; size_t x_15; size_t x_16; size_t x_17; size_t x_18; lean_object* x_19; lean_object* x_20; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 2);
x_6 = lean_array_get_size(x_1);
x_7 = l_Lean_instHashableMVarId_hash(x_4);
x_8 = 32;
x_9 = lean_uint64_shift_right(x_7, x_8);
x_10 = lean_uint64_xor(x_7, x_9);
x_11 = 16;
x_12 = lean_uint64_shift_right(x_10, x_11);
x_13 = lean_uint64_xor(x_10, x_12);
x_14 = lean_uint64_to_usize(x_13);
x_15 = lean_usize_of_nat(x_6);
x_16 = 1;
x_17 = lean_usize_sub(x_15, x_16);
x_18 = lean_usize_land(x_14, x_17);
x_19 = lean_array_uget(x_1, x_18);
lean_ctor_set(x_2, 2, x_19);
x_20 = lean_array_uset(x_1, x_18, x_2);
x_1 = x_20;
x_2 = x_5;
goto _start;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint64_t x_26; uint64_t x_27; uint64_t x_28; uint64_t x_29; uint64_t x_30; uint64_t x_31; uint64_t x_32; size_t x_33; size_t x_34; size_t x_35; size_t x_36; size_t x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_22 = lean_ctor_get(x_2, 0);
x_23 = lean_ctor_get(x_2, 1);
x_24 = lean_ctor_get(x_2, 2);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_2);
x_25 = lean_array_get_size(x_1);
x_26 = l_Lean_instHashableMVarId_hash(x_22);
x_27 = 32;
x_28 = lean_uint64_shift_right(x_26, x_27);
x_29 = lean_uint64_xor(x_26, x_28);
x_30 = 16;
x_31 = lean_uint64_shift_right(x_29, x_30);
x_32 = lean_uint64_xor(x_29, x_31);
x_33 = lean_uint64_to_usize(x_32);
x_34 = lean_usize_of_nat(x_25);
x_35 = 1;
x_36 = lean_usize_sub(x_34, x_35);
x_37 = lean_usize_land(x_33, x_36);
x_38 = lean_array_uget(x_1, x_37);
x_39 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_39, 0, x_22);
lean_ctor_set(x_39, 1, x_23);
lean_ctor_set(x_39, 2, x_38);
x_40 = lean_array_uset(x_1, x_37, x_39);
x_1 = x_40;
x_2 = x_24;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5_spec__5___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint64_t x_5; uint64_t x_6; uint64_t x_7; uint64_t x_8; uint64_t x_9; uint64_t x_10; uint64_t x_11; size_t x_12; size_t x_13; size_t x_14; size_t x_15; size_t x_16; lean_object* x_17; uint8_t x_18; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_array_get_size(x_3);
x_5 = l_Lean_instHashableMVarId_hash(x_2);
x_6 = 32;
x_7 = lean_uint64_shift_right(x_5, x_6);
x_8 = lean_uint64_xor(x_5, x_7);
x_9 = 16;
x_10 = lean_uint64_shift_right(x_8, x_9);
x_11 = lean_uint64_xor(x_8, x_10);
x_12 = lean_uint64_to_usize(x_11);
x_13 = lean_usize_of_nat(x_4);
x_14 = 1;
x_15 = lean_usize_sub(x_13, x_14);
x_16 = lean_usize_land(x_12, x_15);
x_17 = lean_array_uget(x_3, x_16);
x_18 = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___redArg(x_2, x_17);
lean_dec(x_17);
return x_18;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_array_get_size(x_2);
x_5 = lean_nat_dec_lt(x_1, x_4);
if (x_5 == 0)
{
lean_dec_ref(x_2);
lean_dec(x_1);
return x_3;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_array_fget(x_2, x_1);
x_7 = lean_box(0);
x_8 = lean_array_fset(x_2, x_1, x_7);
x_9 = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5_spec__5___redArg(x_3, x_6);
x_10 = lean_unsigned_to_nat(1u);
x_11 = lean_nat_add(x_1, x_10);
lean_dec(x_1);
x_1 = x_11;
x_2 = x_8;
x_3 = x_9;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lean_array_get_size(x_1);
x_3 = lean_unsigned_to_nat(2u);
x_4 = lean_nat_mul(x_2, x_3);
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_box(0);
x_7 = lean_mk_array(x_4, x_6);
x_8 = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5___redArg(x_5, x_1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint64_t x_7; uint64_t x_8; uint64_t x_9; uint64_t x_10; uint64_t x_11; uint64_t x_12; uint64_t x_13; size_t x_14; size_t x_15; size_t x_16; size_t x_17; size_t x_18; lean_object* x_19; uint8_t x_20; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_array_get_size(x_5);
x_7 = l_Lean_instHashableMVarId_hash(x_2);
x_8 = 32;
x_9 = lean_uint64_shift_right(x_7, x_8);
x_10 = lean_uint64_xor(x_7, x_9);
x_11 = 16;
x_12 = lean_uint64_shift_right(x_10, x_11);
x_13 = lean_uint64_xor(x_10, x_12);
x_14 = lean_uint64_to_usize(x_13);
x_15 = lean_usize_of_nat(x_6);
x_16 = 1;
x_17 = lean_usize_sub(x_15, x_16);
x_18 = lean_usize_land(x_14, x_17);
x_19 = lean_array_uget(x_5, x_18);
x_20 = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___redArg(x_2, x_19);
if (x_20 == 0)
{
uint8_t x_21; 
lean_inc_ref(x_5);
lean_inc(x_4);
x_21 = !lean_is_exclusive(x_1);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_22 = lean_ctor_get(x_1, 1);
lean_dec(x_22);
x_23 = lean_ctor_get(x_1, 0);
lean_dec(x_23);
x_24 = lean_unsigned_to_nat(1u);
x_25 = lean_nat_add(x_4, x_24);
lean_dec(x_4);
x_26 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_26, 0, x_2);
lean_ctor_set(x_26, 1, x_3);
lean_ctor_set(x_26, 2, x_19);
x_27 = lean_array_uset(x_5, x_18, x_26);
x_28 = lean_unsigned_to_nat(4u);
x_29 = lean_nat_mul(x_25, x_28);
x_30 = lean_unsigned_to_nat(3u);
x_31 = lean_nat_div(x_29, x_30);
lean_dec(x_29);
x_32 = lean_array_get_size(x_27);
x_33 = lean_nat_dec_le(x_31, x_32);
lean_dec(x_31);
if (x_33 == 0)
{
lean_object* x_34; 
x_34 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5___redArg(x_27);
lean_ctor_set(x_1, 1, x_34);
lean_ctor_set(x_1, 0, x_25);
return x_1;
}
else
{
lean_ctor_set(x_1, 1, x_27);
lean_ctor_set(x_1, 0, x_25);
return x_1;
}
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; uint8_t x_44; 
lean_dec(x_1);
x_35 = lean_unsigned_to_nat(1u);
x_36 = lean_nat_add(x_4, x_35);
lean_dec(x_4);
x_37 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_37, 0, x_2);
lean_ctor_set(x_37, 1, x_3);
lean_ctor_set(x_37, 2, x_19);
x_38 = lean_array_uset(x_5, x_18, x_37);
x_39 = lean_unsigned_to_nat(4u);
x_40 = lean_nat_mul(x_36, x_39);
x_41 = lean_unsigned_to_nat(3u);
x_42 = lean_nat_div(x_40, x_41);
lean_dec(x_40);
x_43 = lean_array_get_size(x_38);
x_44 = lean_nat_dec_le(x_42, x_43);
lean_dec(x_42);
if (x_44 == 0)
{
lean_object* x_45; lean_object* x_46; 
x_45 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5___redArg(x_38);
x_46 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_46, 0, x_36);
lean_ctor_set(x_46, 1, x_45);
return x_46;
}
else
{
lean_object* x_47; 
x_47 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_47, 0, x_36);
lean_ctor_set(x_47, 1, x_38);
return x_47;
}
}
}
else
{
lean_dec(x_19);
lean_dec(x_3);
lean_dec(x_2);
return x_1;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__5_spec__5___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__9(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lean_usize_dec_lt(x_3, x_2);
if (x_5 == 0)
{
return x_4;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; size_t x_9; size_t x_10; 
x_6 = lean_array_uget(x_1, x_3);
x_7 = lean_box(0);
x_8 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4___redArg(x_4, x_6, x_7);
x_9 = 1;
x_10 = lean_usize_add(x_3, x_9);
x_3 = x_10;
x_4 = x_8;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4(lean_object* x_1, lean_object* x_2) {
_start:
{
size_t x_3; size_t x_4; lean_object* x_5; 
x_3 = lean_array_size(x_2);
x_4 = 0;
x_5 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__9(x_2, x_3, x_4, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__12(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_7; 
lean_dec_ref(x_1);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_5, 0);
x_9 = lean_ctor_get(x_5, 2);
x_10 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___redArg(x_4, x_8);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_dec_ref(x_1);
x_11 = lean_box(x_2);
x_12 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_12, 0, x_11);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_3);
x_14 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
else
{
lean_inc_ref(x_1);
{
lean_object* _tmp_4 = x_9;
lean_object* _tmp_5 = x_1;
x_5 = _tmp_4;
x_6 = _tmp_5;
}
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__13(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, size_t x_6, size_t x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; 
x_9 = lean_usize_dec_lt(x_7, x_6);
if (x_9 == 0)
{
lean_dec_ref(x_1);
return x_8;
}
else
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_array_uget(x_5, x_7);
lean_inc_ref(x_1);
x_11 = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__12(x_1, x_2, x_3, x_4, x_10, x_8);
lean_dec(x_10);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; 
lean_dec_ref(x_1);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
return x_12;
}
else
{
lean_object* x_13; size_t x_14; size_t x_15; 
x_13 = lean_ctor_get(x_11, 0);
lean_inc(x_13);
lean_dec_ref(x_11);
x_14 = 1;
x_15 = lean_usize_add(x_7, x_14);
x_7 = x_15;
x_8 = x_13;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__14(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_7; 
lean_dec_ref(x_1);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_5, 0);
x_9 = lean_ctor_get(x_5, 2);
x_10 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___redArg(x_4, x_8);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_dec_ref(x_1);
x_11 = lean_box(x_2);
x_12 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_12, 0, x_11);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_3);
x_14 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
else
{
lean_inc_ref(x_1);
{
lean_object* _tmp_4 = x_9;
lean_object* _tmp_5 = x_1;
x_5 = _tmp_4;
x_6 = _tmp_5;
}
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__15(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, size_t x_6, size_t x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; 
x_9 = lean_usize_dec_lt(x_7, x_6);
if (x_9 == 0)
{
lean_dec_ref(x_1);
return x_8;
}
else
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_array_uget(x_5, x_7);
lean_inc_ref(x_1);
x_11 = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__14(x_1, x_2, x_3, x_4, x_10, x_8);
lean_dec(x_10);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; 
lean_dec_ref(x_1);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
return x_12;
}
else
{
lean_object* x_13; size_t x_14; size_t x_15; 
x_13 = lean_ctor_get(x_11, 0);
lean_inc(x_13);
lean_dec_ref(x_11);
x_14 = 1;
x_15 = lean_usize_add(x_7, x_14);
x_7 = x_15;
x_8 = x_13;
goto _start;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRuleTac___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("the goal produced by the rule depends on different metavariables than the original goal.", 88, 88);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRuleTac___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_runNormRuleTac___closed__0;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRuleTac___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_runNormRuleTac___closed__1;
x_2 = l_Lean_MessageData_ofFormat(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRuleTac___closed__3() {
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
static lean_object* _init_lp_aesop_Aesop_runNormRuleTac___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("rule did not produce exactly one rule application.", 50, 50);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRuleTac___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_runNormRuleTac___closed__4;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRuleTac___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("rule produced more than one subgoal.", 36, 36);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRuleTac___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_runNormRuleTac___closed__6;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRuleTac___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_Check_rules;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRuleTac(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; lean_object* x_14; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; uint8_t x_48; lean_object* x_49; lean_object* x_50; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; uint8_t x_58; lean_object* x_59; lean_object* x_60; lean_object* x_71; 
x_71 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_71) == 0)
{
lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_72 = lean_ctor_get(x_71, 0);
lean_inc(x_72);
lean_dec_ref(x_71);
x_73 = lean_ctor_get(x_1, 0);
x_74 = lean_ctor_get(x_1, 4);
lean_inc(x_74);
x_75 = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTacDescr_run___boxed), 8, 1);
lean_closure_set(x_75, 0, x_74);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_2);
lean_inc_ref(x_73);
x_76 = lp_aesop_Aesop_runRuleTac(x_75, x_73, x_72, x_2, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_72);
if (lean_obj_tag(x_76) == 0)
{
lean_object* x_77; lean_object* x_78; lean_object* x_257; 
x_77 = lean_ctor_get(x_76, 0);
lean_inc(x_77);
lean_dec_ref(x_76);
lean_inc(x_74);
x_257 = lp_aesop_Aesop_RuleTacDescr_forwardRuleMatches_x3f(x_74);
if (lean_obj_tag(x_257) == 0)
{
lean_object* x_258; 
x_258 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_78 = x_258;
goto block_256;
}
else
{
lean_object* x_259; 
x_259 = lean_ctor_get(x_257, 0);
lean_inc(x_259);
lean_dec_ref(x_257);
x_78 = x_259;
goto block_256;
}
block_256:
{
if (lean_obj_tag(x_77) == 0)
{
lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; uint8_t x_83; 
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_79 = lean_ctor_get(x_77, 0);
lean_inc(x_79);
lean_dec_ref(x_77);
x_80 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_81 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_80, x_10);
x_82 = lean_ctor_get(x_81, 0);
lean_inc(x_82);
lean_dec_ref(x_81);
x_83 = lean_unbox(x_82);
lean_dec(x_82);
if (x_83 == 0)
{
lean_dec(x_79);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
x_13 = x_78;
x_14 = lean_box(0);
goto block_18;
}
else
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; 
x_84 = lean_ctor_get(x_80, 0);
lean_inc(x_84);
x_85 = l_Lean_Exception_toMessageData(x_79);
x_86 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_84, x_85, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
if (lean_obj_tag(x_86) == 0)
{
lean_dec_ref(x_86);
x_13 = x_78;
x_14 = lean_box(0);
goto block_18;
}
else
{
uint8_t x_87; 
lean_dec_ref(x_78);
x_87 = !lean_is_exclusive(x_86);
if (x_87 == 0)
{
return x_86;
}
else
{
lean_object* x_88; lean_object* x_89; 
x_88 = lean_ctor_get(x_86, 0);
lean_inc(x_88);
lean_dec(x_86);
x_89 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_89, 0, x_88);
return x_89;
}
}
}
}
else
{
uint8_t x_90; 
x_90 = !lean_is_exclusive(x_77);
if (x_90 == 0)
{
lean_object* x_91; lean_object* x_92; lean_object* x_93; uint8_t x_94; 
x_91 = lean_ctor_get(x_77, 0);
x_92 = lean_array_get_size(x_91);
x_93 = lean_unsigned_to_nat(1u);
x_94 = lean_nat_dec_eq(x_92, x_93);
if (x_94 == 0)
{
lean_object* x_95; lean_object* x_96; 
lean_free_object(x_77);
lean_dec(x_91);
lean_dec_ref(x_78);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_95 = lp_aesop_Aesop_runNormRuleTac___closed__5;
x_96 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(x_1, x_2, x_95, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
return x_96;
}
else
{
lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; 
x_97 = lean_unsigned_to_nat(0u);
x_98 = lean_array_fget(x_91, x_97);
lean_dec(x_91);
x_99 = lean_ctor_get(x_98, 0);
lean_inc_ref(x_99);
x_100 = lean_ctor_get(x_98, 1);
lean_inc_ref(x_100);
x_101 = lean_ctor_get(x_98, 2);
lean_inc(x_101);
lean_dec(x_98);
x_102 = l_Lean_Meta_SavedState_restore___redArg(x_100, x_9, x_11);
if (lean_obj_tag(x_102) == 0)
{
uint8_t x_103; 
x_103 = !lean_is_exclusive(x_102);
if (x_103 == 0)
{
lean_object* x_104; uint8_t x_105; 
x_104 = lean_ctor_get(x_102, 0);
lean_dec(x_104);
x_105 = l_Array_isEmpty___redArg(x_99);
if (x_105 == 0)
{
lean_object* x_106; uint8_t x_107; 
lean_free_object(x_102);
lean_free_object(x_77);
x_106 = lean_array_get_size(x_99);
x_107 = lean_nat_dec_eq(x_106, x_93);
if (x_107 == 0)
{
lean_object* x_108; lean_object* x_109; 
lean_dec(x_101);
lean_dec_ref(x_100);
lean_dec_ref(x_99);
lean_dec_ref(x_78);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_108 = lp_aesop_Aesop_runNormRuleTac___closed__7;
x_109 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(x_1, x_2, x_108, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
return x_109;
}
else
{
lean_object* x_110; lean_object* x_111; 
x_110 = lean_array_fget(x_99, x_97);
lean_dec_ref(x_99);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_110);
x_111 = lp_aesop_Aesop_ForwardState_applyGoalDiff(x_4, x_110, x_3, x_7, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_111) == 0)
{
lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; uint8_t x_116; 
x_112 = lean_ctor_get(x_111, 0);
lean_inc(x_112);
lean_dec_ref(x_111);
x_113 = lp_aesop_Aesop_runNormRuleTac___closed__8;
x_114 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(x_113, x_10);
x_115 = lean_ctor_get(x_114, 0);
lean_inc(x_115);
lean_dec_ref(x_114);
x_116 = lean_unbox(x_115);
lean_dec(x_115);
if (x_116 == 0)
{
lean_object* x_117; lean_object* x_118; 
lean_dec_ref(x_100);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_117 = lean_ctor_get(x_110, 1);
lean_inc(x_117);
x_118 = lean_ctor_get(x_110, 3);
lean_inc_ref(x_118);
lean_dec(x_110);
x_19 = x_112;
x_20 = x_101;
x_21 = x_118;
x_22 = x_78;
x_23 = x_117;
x_24 = lean_box(0);
goto block_31;
}
else
{
lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; 
x_119 = lean_ctor_get(x_110, 1);
lean_inc(x_119);
x_120 = lean_ctor_get(x_110, 3);
lean_inc_ref(x_120);
lean_dec(x_110);
x_121 = lean_box(x_105);
lean_inc(x_119);
x_122 = lean_alloc_closure((void*)(l_Lean_MVarId_getMVarDependencies___boxed), 7, 2);
lean_closure_set(x_122, 0, x_119);
lean_closure_set(x_122, 1, x_121);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
x_123 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_100, x_122, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_123) == 0)
{
lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; size_t x_131; size_t x_132; lean_object* x_133; lean_object* x_134; 
x_124 = lean_ctor_get(x_123, 0);
lean_inc(x_124);
lean_dec_ref(x_123);
x_125 = lean_ctor_get(x_2, 1);
x_126 = lean_ctor_get(x_124, 1);
x_127 = lp_aesop_Aesop_updateForwardState___redArg___closed__2;
x_128 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4(x_127, x_125);
x_129 = lean_box(0);
x_130 = lp_aesop_Aesop_runNormRuleTac___closed__3;
x_131 = lean_array_size(x_126);
x_132 = 0;
x_133 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__15(x_130, x_105, x_129, x_128, x_126, x_131, x_132, x_130);
x_134 = lean_ctor_get(x_133, 0);
lean_inc(x_134);
lean_dec_ref(x_133);
if (lean_obj_tag(x_134) == 0)
{
x_52 = lean_box(0);
x_53 = x_128;
x_54 = x_112;
x_55 = x_124;
x_56 = x_101;
x_57 = x_120;
x_58 = x_105;
x_59 = x_78;
x_60 = x_119;
goto block_70;
}
else
{
lean_object* x_135; uint8_t x_136; 
x_135 = lean_ctor_get(x_134, 0);
lean_inc(x_135);
lean_dec_ref(x_134);
x_136 = lean_unbox(x_135);
lean_dec(x_135);
if (x_136 == 0)
{
lean_dec_ref(x_128);
lean_dec(x_124);
x_32 = lean_box(0);
x_33 = x_112;
x_34 = x_101;
x_35 = x_120;
x_36 = x_78;
x_37 = x_119;
goto block_43;
}
else
{
x_52 = lean_box(0);
x_53 = x_128;
x_54 = x_112;
x_55 = x_124;
x_56 = x_101;
x_57 = x_120;
x_58 = x_105;
x_59 = x_78;
x_60 = x_119;
goto block_70;
}
}
}
else
{
uint8_t x_137; 
lean_dec_ref(x_120);
lean_dec(x_119);
lean_dec(x_112);
lean_dec(x_101);
lean_dec_ref(x_78);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_137 = !lean_is_exclusive(x_123);
if (x_137 == 0)
{
return x_123;
}
else
{
lean_object* x_138; lean_object* x_139; 
x_138 = lean_ctor_get(x_123, 0);
lean_inc(x_138);
lean_dec(x_123);
x_139 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_139, 0, x_138);
return x_139;
}
}
}
}
else
{
uint8_t x_140; 
lean_dec(x_110);
lean_dec(x_101);
lean_dec_ref(x_100);
lean_dec_ref(x_78);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_140 = !lean_is_exclusive(x_111);
if (x_140 == 0)
{
return x_111;
}
else
{
lean_object* x_141; lean_object* x_142; 
x_141 = lean_ctor_get(x_111, 0);
lean_inc(x_141);
lean_dec(x_111);
x_142 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_142, 0, x_141);
return x_142;
}
}
}
}
else
{
lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; 
lean_dec_ref(x_100);
lean_dec_ref(x_99);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
lean_ctor_set(x_77, 0, x_101);
x_143 = lp_aesop_Aesop_updateForwardState___redArg___closed__2;
x_144 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_144, 0, x_3);
lean_ctor_set(x_144, 1, x_143);
x_145 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_145, 0, x_77);
lean_ctor_set(x_145, 1, x_144);
x_146 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_146, 0, x_145);
x_147 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_147, 0, x_146);
lean_ctor_set(x_147, 1, x_78);
lean_ctor_set(x_102, 0, x_147);
return x_102;
}
}
else
{
uint8_t x_148; 
lean_dec(x_102);
x_148 = l_Array_isEmpty___redArg(x_99);
if (x_148 == 0)
{
lean_object* x_149; uint8_t x_150; 
lean_free_object(x_77);
x_149 = lean_array_get_size(x_99);
x_150 = lean_nat_dec_eq(x_149, x_93);
if (x_150 == 0)
{
lean_object* x_151; lean_object* x_152; 
lean_dec(x_101);
lean_dec_ref(x_100);
lean_dec_ref(x_99);
lean_dec_ref(x_78);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_151 = lp_aesop_Aesop_runNormRuleTac___closed__7;
x_152 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(x_1, x_2, x_151, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
return x_152;
}
else
{
lean_object* x_153; lean_object* x_154; 
x_153 = lean_array_fget(x_99, x_97);
lean_dec_ref(x_99);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_153);
x_154 = lp_aesop_Aesop_ForwardState_applyGoalDiff(x_4, x_153, x_3, x_7, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_154) == 0)
{
lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; uint8_t x_159; 
x_155 = lean_ctor_get(x_154, 0);
lean_inc(x_155);
lean_dec_ref(x_154);
x_156 = lp_aesop_Aesop_runNormRuleTac___closed__8;
x_157 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(x_156, x_10);
x_158 = lean_ctor_get(x_157, 0);
lean_inc(x_158);
lean_dec_ref(x_157);
x_159 = lean_unbox(x_158);
lean_dec(x_158);
if (x_159 == 0)
{
lean_object* x_160; lean_object* x_161; 
lean_dec_ref(x_100);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_160 = lean_ctor_get(x_153, 1);
lean_inc(x_160);
x_161 = lean_ctor_get(x_153, 3);
lean_inc_ref(x_161);
lean_dec(x_153);
x_19 = x_155;
x_20 = x_101;
x_21 = x_161;
x_22 = x_78;
x_23 = x_160;
x_24 = lean_box(0);
goto block_31;
}
else
{
lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; 
x_162 = lean_ctor_get(x_153, 1);
lean_inc(x_162);
x_163 = lean_ctor_get(x_153, 3);
lean_inc_ref(x_163);
lean_dec(x_153);
x_164 = lean_box(x_148);
lean_inc(x_162);
x_165 = lean_alloc_closure((void*)(l_Lean_MVarId_getMVarDependencies___boxed), 7, 2);
lean_closure_set(x_165, 0, x_162);
lean_closure_set(x_165, 1, x_164);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
x_166 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_100, x_165, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_166) == 0)
{
lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; size_t x_174; size_t x_175; lean_object* x_176; lean_object* x_177; 
x_167 = lean_ctor_get(x_166, 0);
lean_inc(x_167);
lean_dec_ref(x_166);
x_168 = lean_ctor_get(x_2, 1);
x_169 = lean_ctor_get(x_167, 1);
x_170 = lp_aesop_Aesop_updateForwardState___redArg___closed__2;
x_171 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4(x_170, x_168);
x_172 = lean_box(0);
x_173 = lp_aesop_Aesop_runNormRuleTac___closed__3;
x_174 = lean_array_size(x_169);
x_175 = 0;
x_176 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__15(x_173, x_148, x_172, x_171, x_169, x_174, x_175, x_173);
x_177 = lean_ctor_get(x_176, 0);
lean_inc(x_177);
lean_dec_ref(x_176);
if (lean_obj_tag(x_177) == 0)
{
x_52 = lean_box(0);
x_53 = x_171;
x_54 = x_155;
x_55 = x_167;
x_56 = x_101;
x_57 = x_163;
x_58 = x_148;
x_59 = x_78;
x_60 = x_162;
goto block_70;
}
else
{
lean_object* x_178; uint8_t x_179; 
x_178 = lean_ctor_get(x_177, 0);
lean_inc(x_178);
lean_dec_ref(x_177);
x_179 = lean_unbox(x_178);
lean_dec(x_178);
if (x_179 == 0)
{
lean_dec_ref(x_171);
lean_dec(x_167);
x_32 = lean_box(0);
x_33 = x_155;
x_34 = x_101;
x_35 = x_163;
x_36 = x_78;
x_37 = x_162;
goto block_43;
}
else
{
x_52 = lean_box(0);
x_53 = x_171;
x_54 = x_155;
x_55 = x_167;
x_56 = x_101;
x_57 = x_163;
x_58 = x_148;
x_59 = x_78;
x_60 = x_162;
goto block_70;
}
}
}
else
{
lean_object* x_180; lean_object* x_181; lean_object* x_182; 
lean_dec_ref(x_163);
lean_dec(x_162);
lean_dec(x_155);
lean_dec(x_101);
lean_dec_ref(x_78);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_180 = lean_ctor_get(x_166, 0);
lean_inc(x_180);
if (lean_is_exclusive(x_166)) {
 lean_ctor_release(x_166, 0);
 x_181 = x_166;
} else {
 lean_dec_ref(x_166);
 x_181 = lean_box(0);
}
if (lean_is_scalar(x_181)) {
 x_182 = lean_alloc_ctor(1, 1, 0);
} else {
 x_182 = x_181;
}
lean_ctor_set(x_182, 0, x_180);
return x_182;
}
}
}
else
{
lean_object* x_183; lean_object* x_184; lean_object* x_185; 
lean_dec(x_153);
lean_dec(x_101);
lean_dec_ref(x_100);
lean_dec_ref(x_78);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_183 = lean_ctor_get(x_154, 0);
lean_inc(x_183);
if (lean_is_exclusive(x_154)) {
 lean_ctor_release(x_154, 0);
 x_184 = x_154;
} else {
 lean_dec_ref(x_154);
 x_184 = lean_box(0);
}
if (lean_is_scalar(x_184)) {
 x_185 = lean_alloc_ctor(1, 1, 0);
} else {
 x_185 = x_184;
}
lean_ctor_set(x_185, 0, x_183);
return x_185;
}
}
}
else
{
lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; 
lean_dec_ref(x_100);
lean_dec_ref(x_99);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
lean_ctor_set(x_77, 0, x_101);
x_186 = lp_aesop_Aesop_updateForwardState___redArg___closed__2;
x_187 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_187, 0, x_3);
lean_ctor_set(x_187, 1, x_186);
x_188 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_188, 0, x_77);
lean_ctor_set(x_188, 1, x_187);
x_189 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_189, 0, x_188);
x_190 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_190, 0, x_189);
lean_ctor_set(x_190, 1, x_78);
x_191 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_191, 0, x_190);
return x_191;
}
}
}
else
{
uint8_t x_192; 
lean_dec(x_101);
lean_dec_ref(x_100);
lean_dec_ref(x_99);
lean_free_object(x_77);
lean_dec_ref(x_78);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_192 = !lean_is_exclusive(x_102);
if (x_192 == 0)
{
return x_102;
}
else
{
lean_object* x_193; lean_object* x_194; 
x_193 = lean_ctor_get(x_102, 0);
lean_inc(x_193);
lean_dec(x_102);
x_194 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_194, 0, x_193);
return x_194;
}
}
}
}
else
{
lean_object* x_195; lean_object* x_196; lean_object* x_197; uint8_t x_198; 
x_195 = lean_ctor_get(x_77, 0);
lean_inc(x_195);
lean_dec(x_77);
x_196 = lean_array_get_size(x_195);
x_197 = lean_unsigned_to_nat(1u);
x_198 = lean_nat_dec_eq(x_196, x_197);
if (x_198 == 0)
{
lean_object* x_199; lean_object* x_200; 
lean_dec(x_195);
lean_dec_ref(x_78);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_199 = lp_aesop_Aesop_runNormRuleTac___closed__5;
x_200 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(x_1, x_2, x_199, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
return x_200;
}
else
{
lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; 
x_201 = lean_unsigned_to_nat(0u);
x_202 = lean_array_fget(x_195, x_201);
lean_dec(x_195);
x_203 = lean_ctor_get(x_202, 0);
lean_inc_ref(x_203);
x_204 = lean_ctor_get(x_202, 1);
lean_inc_ref(x_204);
x_205 = lean_ctor_get(x_202, 2);
lean_inc(x_205);
lean_dec(x_202);
x_206 = l_Lean_Meta_SavedState_restore___redArg(x_204, x_9, x_11);
if (lean_obj_tag(x_206) == 0)
{
lean_object* x_207; uint8_t x_208; 
if (lean_is_exclusive(x_206)) {
 lean_ctor_release(x_206, 0);
 x_207 = x_206;
} else {
 lean_dec_ref(x_206);
 x_207 = lean_box(0);
}
x_208 = l_Array_isEmpty___redArg(x_203);
if (x_208 == 0)
{
lean_object* x_209; uint8_t x_210; 
lean_dec(x_207);
x_209 = lean_array_get_size(x_203);
x_210 = lean_nat_dec_eq(x_209, x_197);
if (x_210 == 0)
{
lean_object* x_211; lean_object* x_212; 
lean_dec(x_205);
lean_dec_ref(x_204);
lean_dec_ref(x_203);
lean_dec_ref(x_78);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_211 = lp_aesop_Aesop_runNormRuleTac___closed__7;
x_212 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(x_1, x_2, x_211, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
return x_212;
}
else
{
lean_object* x_213; lean_object* x_214; 
x_213 = lean_array_fget(x_203, x_201);
lean_dec_ref(x_203);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_213);
x_214 = lp_aesop_Aesop_ForwardState_applyGoalDiff(x_4, x_213, x_3, x_7, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_214) == 0)
{
lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; uint8_t x_219; 
x_215 = lean_ctor_get(x_214, 0);
lean_inc(x_215);
lean_dec_ref(x_214);
x_216 = lp_aesop_Aesop_runNormRuleTac___closed__8;
x_217 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(x_216, x_10);
x_218 = lean_ctor_get(x_217, 0);
lean_inc(x_218);
lean_dec_ref(x_217);
x_219 = lean_unbox(x_218);
lean_dec(x_218);
if (x_219 == 0)
{
lean_object* x_220; lean_object* x_221; 
lean_dec_ref(x_204);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_220 = lean_ctor_get(x_213, 1);
lean_inc(x_220);
x_221 = lean_ctor_get(x_213, 3);
lean_inc_ref(x_221);
lean_dec(x_213);
x_19 = x_215;
x_20 = x_205;
x_21 = x_221;
x_22 = x_78;
x_23 = x_220;
x_24 = lean_box(0);
goto block_31;
}
else
{
lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; 
x_222 = lean_ctor_get(x_213, 1);
lean_inc(x_222);
x_223 = lean_ctor_get(x_213, 3);
lean_inc_ref(x_223);
lean_dec(x_213);
x_224 = lean_box(x_208);
lean_inc(x_222);
x_225 = lean_alloc_closure((void*)(l_Lean_MVarId_getMVarDependencies___boxed), 7, 2);
lean_closure_set(x_225, 0, x_222);
lean_closure_set(x_225, 1, x_224);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
x_226 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_204, x_225, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_226) == 0)
{
lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; size_t x_234; size_t x_235; lean_object* x_236; lean_object* x_237; 
x_227 = lean_ctor_get(x_226, 0);
lean_inc(x_227);
lean_dec_ref(x_226);
x_228 = lean_ctor_get(x_2, 1);
x_229 = lean_ctor_get(x_227, 1);
x_230 = lp_aesop_Aesop_updateForwardState___redArg___closed__2;
x_231 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4(x_230, x_228);
x_232 = lean_box(0);
x_233 = lp_aesop_Aesop_runNormRuleTac___closed__3;
x_234 = lean_array_size(x_229);
x_235 = 0;
x_236 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__15(x_233, x_208, x_232, x_231, x_229, x_234, x_235, x_233);
x_237 = lean_ctor_get(x_236, 0);
lean_inc(x_237);
lean_dec_ref(x_236);
if (lean_obj_tag(x_237) == 0)
{
x_52 = lean_box(0);
x_53 = x_231;
x_54 = x_215;
x_55 = x_227;
x_56 = x_205;
x_57 = x_223;
x_58 = x_208;
x_59 = x_78;
x_60 = x_222;
goto block_70;
}
else
{
lean_object* x_238; uint8_t x_239; 
x_238 = lean_ctor_get(x_237, 0);
lean_inc(x_238);
lean_dec_ref(x_237);
x_239 = lean_unbox(x_238);
lean_dec(x_238);
if (x_239 == 0)
{
lean_dec_ref(x_231);
lean_dec(x_227);
x_32 = lean_box(0);
x_33 = x_215;
x_34 = x_205;
x_35 = x_223;
x_36 = x_78;
x_37 = x_222;
goto block_43;
}
else
{
x_52 = lean_box(0);
x_53 = x_231;
x_54 = x_215;
x_55 = x_227;
x_56 = x_205;
x_57 = x_223;
x_58 = x_208;
x_59 = x_78;
x_60 = x_222;
goto block_70;
}
}
}
else
{
lean_object* x_240; lean_object* x_241; lean_object* x_242; 
lean_dec_ref(x_223);
lean_dec(x_222);
lean_dec(x_215);
lean_dec(x_205);
lean_dec_ref(x_78);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_240 = lean_ctor_get(x_226, 0);
lean_inc(x_240);
if (lean_is_exclusive(x_226)) {
 lean_ctor_release(x_226, 0);
 x_241 = x_226;
} else {
 lean_dec_ref(x_226);
 x_241 = lean_box(0);
}
if (lean_is_scalar(x_241)) {
 x_242 = lean_alloc_ctor(1, 1, 0);
} else {
 x_242 = x_241;
}
lean_ctor_set(x_242, 0, x_240);
return x_242;
}
}
}
else
{
lean_object* x_243; lean_object* x_244; lean_object* x_245; 
lean_dec(x_213);
lean_dec(x_205);
lean_dec_ref(x_204);
lean_dec_ref(x_78);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_243 = lean_ctor_get(x_214, 0);
lean_inc(x_243);
if (lean_is_exclusive(x_214)) {
 lean_ctor_release(x_214, 0);
 x_244 = x_214;
} else {
 lean_dec_ref(x_214);
 x_244 = lean_box(0);
}
if (lean_is_scalar(x_244)) {
 x_245 = lean_alloc_ctor(1, 1, 0);
} else {
 x_245 = x_244;
}
lean_ctor_set(x_245, 0, x_243);
return x_245;
}
}
}
else
{
lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; 
lean_dec_ref(x_204);
lean_dec_ref(x_203);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_246 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_246, 0, x_205);
x_247 = lp_aesop_Aesop_updateForwardState___redArg___closed__2;
x_248 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_248, 0, x_3);
lean_ctor_set(x_248, 1, x_247);
x_249 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_249, 0, x_246);
lean_ctor_set(x_249, 1, x_248);
x_250 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_250, 0, x_249);
x_251 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_251, 0, x_250);
lean_ctor_set(x_251, 1, x_78);
if (lean_is_scalar(x_207)) {
 x_252 = lean_alloc_ctor(0, 1, 0);
} else {
 x_252 = x_207;
}
lean_ctor_set(x_252, 0, x_251);
return x_252;
}
}
else
{
lean_object* x_253; lean_object* x_254; lean_object* x_255; 
lean_dec(x_205);
lean_dec_ref(x_204);
lean_dec_ref(x_203);
lean_dec_ref(x_78);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_253 = lean_ctor_get(x_206, 0);
lean_inc(x_253);
if (lean_is_exclusive(x_206)) {
 lean_ctor_release(x_206, 0);
 x_254 = x_206;
} else {
 lean_dec_ref(x_206);
 x_254 = lean_box(0);
}
if (lean_is_scalar(x_254)) {
 x_255 = lean_alloc_ctor(1, 1, 0);
} else {
 x_255 = x_254;
}
lean_ctor_set(x_255, 0, x_253);
return x_255;
}
}
}
}
}
}
else
{
uint8_t x_260; 
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_260 = !lean_is_exclusive(x_76);
if (x_260 == 0)
{
return x_76;
}
else
{
lean_object* x_261; lean_object* x_262; 
x_261 = lean_ctor_get(x_76, 0);
lean_inc(x_261);
lean_dec(x_76);
x_262 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_262, 0, x_261);
return x_262;
}
}
}
else
{
uint8_t x_263; 
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_263 = !lean_is_exclusive(x_71);
if (x_263 == 0)
{
return x_71;
}
else
{
lean_object* x_264; lean_object* x_265; 
x_264 = lean_ctor_get(x_71, 0);
lean_inc(x_264);
lean_dec(x_71);
x_265 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_265, 0, x_264);
return x_265;
}
}
block_18:
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_15 = lean_box(0);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_13);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
block_31:
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_23);
lean_ctor_set(x_25, 1, x_20);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_19);
lean_ctor_set(x_26, 1, x_21);
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_25);
lean_ctor_set(x_27, 1, x_26);
x_28 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_28, 0, x_27);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_22);
x_30 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_30, 0, x_29);
return x_30;
}
block_43:
{
lean_object* x_38; lean_object* x_39; uint8_t x_40; 
lean_dec(x_37);
lean_dec_ref(x_36);
lean_dec_ref(x_35);
lean_dec(x_34);
lean_dec_ref(x_33);
x_38 = lp_aesop_Aesop_runNormRuleTac___closed__2;
x_39 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg(x_1, x_2, x_38, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
x_40 = !lean_is_exclusive(x_39);
if (x_40 == 0)
{
return x_39;
}
else
{
lean_object* x_41; lean_object* x_42; 
x_41 = lean_ctor_get(x_39, 0);
lean_inc(x_41);
lean_dec(x_39);
x_42 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_42, 0, x_41);
return x_42;
}
}
block_51:
{
if (x_48 == 0)
{
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_19 = x_45;
x_20 = x_46;
x_21 = x_47;
x_22 = x_49;
x_23 = x_50;
x_24 = lean_box(0);
goto block_31;
}
else
{
x_32 = lean_box(0);
x_33 = x_45;
x_34 = x_46;
x_35 = x_47;
x_36 = x_49;
x_37 = x_50;
goto block_43;
}
}
block_70:
{
lean_object* x_61; lean_object* x_62; lean_object* x_63; size_t x_64; size_t x_65; lean_object* x_66; lean_object* x_67; 
x_61 = lean_ctor_get(x_53, 1);
lean_inc_ref(x_61);
lean_dec_ref(x_53);
x_62 = lean_box(0);
x_63 = lp_aesop_Aesop_runNormRuleTac___closed__3;
x_64 = lean_array_size(x_61);
x_65 = 0;
x_66 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__13(x_63, x_58, x_62, x_55, x_61, x_64, x_65, x_63);
lean_dec_ref(x_61);
lean_dec_ref(x_55);
x_67 = lean_ctor_get(x_66, 0);
lean_inc(x_67);
lean_dec_ref(x_66);
if (lean_obj_tag(x_67) == 0)
{
x_44 = lean_box(0);
x_45 = x_54;
x_46 = x_56;
x_47 = x_57;
x_48 = x_58;
x_49 = x_59;
x_50 = x_60;
goto block_51;
}
else
{
lean_object* x_68; uint8_t x_69; 
x_68 = lean_ctor_get(x_67, 0);
lean_inc(x_68);
lean_dec_ref(x_67);
x_69 = lean_unbox(x_68);
lean_dec(x_68);
if (x_69 == 0)
{
x_32 = lean_box(0);
x_33 = x_54;
x_34 = x_56;
x_35 = x_57;
x_36 = x_59;
x_37 = x_60;
goto block_43;
}
else
{
x_44 = lean_box(0);
x_45 = x_54;
x_46 = x_56;
x_47 = x_57;
x_48 = x_58;
x_49 = x_59;
x_50 = x_60;
goto block_51;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(x_1, x_2);
lean_dec_ref(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__4_spec__4___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__9___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; size_t x_6; lean_object* x_7; 
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4_spec__9(x_1, x_5, x_6, x_4);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__13___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; size_t x_10; size_t x_11; lean_object* x_12; 
x_9 = lean_unbox(x_2);
x_10 = lean_unbox_usize(x_6);
lean_dec(x_6);
x_11 = lean_unbox_usize(x_7);
lean_dec(x_7);
x_12 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__13(x_1, x_9, x_3, x_4, x_5, x_10, x_11, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__15___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; size_t x_10; size_t x_11; lean_object* x_12; 
x_9 = lean_unbox(x_2);
x_10 = lean_unbox_usize(x_6);
lean_dec(x_6);
x_11 = lean_unbox_usize(x_7);
lean_dec(x_7);
x_12 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRuleTac_spec__15(x_1, x_9, x_3, x_4, x_5, x_10, x_11, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__12___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lean_unbox(x_2);
x_8 = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__12(x_1, x_7, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__14___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lean_unbox(x_2);
x_8 = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Aesop_runNormRuleTac_spec__14(x_1, x_7, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_runNormRuleTac_spec__11___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRuleTac___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_runNormRuleTac(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 2);
x_5 = lean_ctor_get(x_2, 13);
x_6 = l_Lean_checkTraceOption(x_5, x_4, x_1);
x_7 = lean_box(x_6);
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1___redArg(x_1, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3_spec__3(size_t x_1, size_t x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_lt(x_2, x_1);
if (x_4 == 0)
{
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; size_t x_9; size_t x_10; lean_object* x_11; 
x_5 = lean_array_uget(x_3, x_2);
x_6 = lean_ctor_get(x_5, 1);
lean_inc_ref(x_6);
lean_dec(x_5);
x_7 = lean_unsigned_to_nat(0u);
x_8 = lean_array_uset(x_3, x_2, x_7);
x_9 = 1;
x_10 = lean_usize_add(x_2, x_9);
x_11 = lean_array_uset(x_8, x_2, x_6);
x_2 = x_10;
x_3 = x_11;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_10; 
x_10 = !lean_is_exclusive(x_7);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; size_t x_17; size_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; uint8_t x_22; 
x_11 = lean_ctor_get(x_7, 5);
x_12 = lean_st_ref_get(x_8);
x_13 = lean_ctor_get(x_12, 4);
lean_inc_ref(x_13);
lean_dec(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc_ref(x_14);
lean_dec_ref(x_13);
x_15 = l_Lean_replaceRef(x_3, x_11);
lean_dec(x_11);
lean_ctor_set(x_7, 5, x_15);
x_16 = l_Lean_PersistentArray_toArray___redArg(x_14);
lean_dec_ref(x_14);
x_17 = lean_array_size(x_16);
x_18 = 0;
x_19 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3_spec__3(x_17, x_18, x_16);
x_20 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_20, 0, x_2);
lean_ctor_set(x_20, 1, x_4);
lean_ctor_set(x_20, 2, x_19);
x_21 = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0(x_20, x_5, x_6, x_7, x_8);
lean_dec_ref(x_7);
x_22 = !lean_is_exclusive(x_21);
if (x_22 == 0)
{
lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_23 = lean_ctor_get(x_21, 0);
x_24 = lean_st_ref_take(x_8);
x_25 = !lean_is_exclusive(x_24);
if (x_25 == 0)
{
lean_object* x_26; uint8_t x_27; 
x_26 = lean_ctor_get(x_24, 4);
x_27 = !lean_is_exclusive(x_26);
if (x_27 == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_28 = lean_ctor_get(x_26, 0);
lean_dec(x_28);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_3);
lean_ctor_set(x_29, 1, x_23);
x_30 = l_Lean_PersistentArray_push___redArg(x_1, x_29);
lean_ctor_set(x_26, 0, x_30);
x_31 = lean_st_ref_set(x_8, x_24);
x_32 = lean_box(0);
lean_ctor_set(x_21, 0, x_32);
return x_21;
}
else
{
uint64_t x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_33 = lean_ctor_get_uint64(x_26, sizeof(void*)*1);
lean_dec(x_26);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_3);
lean_ctor_set(x_34, 1, x_23);
x_35 = l_Lean_PersistentArray_push___redArg(x_1, x_34);
x_36 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_36, 0, x_35);
lean_ctor_set_uint64(x_36, sizeof(void*)*1, x_33);
lean_ctor_set(x_24, 4, x_36);
x_37 = lean_st_ref_set(x_8, x_24);
x_38 = lean_box(0);
lean_ctor_set(x_21, 0, x_38);
return x_21;
}
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; uint64_t x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; 
x_39 = lean_ctor_get(x_24, 4);
x_40 = lean_ctor_get(x_24, 0);
x_41 = lean_ctor_get(x_24, 1);
x_42 = lean_ctor_get(x_24, 2);
x_43 = lean_ctor_get(x_24, 3);
x_44 = lean_ctor_get(x_24, 5);
x_45 = lean_ctor_get(x_24, 6);
x_46 = lean_ctor_get(x_24, 7);
x_47 = lean_ctor_get(x_24, 8);
lean_inc(x_47);
lean_inc(x_46);
lean_inc(x_45);
lean_inc(x_44);
lean_inc(x_39);
lean_inc(x_43);
lean_inc(x_42);
lean_inc(x_41);
lean_inc(x_40);
lean_dec(x_24);
x_48 = lean_ctor_get_uint64(x_39, sizeof(void*)*1);
if (lean_is_exclusive(x_39)) {
 lean_ctor_release(x_39, 0);
 x_49 = x_39;
} else {
 lean_dec_ref(x_39);
 x_49 = lean_box(0);
}
x_50 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_50, 0, x_3);
lean_ctor_set(x_50, 1, x_23);
x_51 = l_Lean_PersistentArray_push___redArg(x_1, x_50);
if (lean_is_scalar(x_49)) {
 x_52 = lean_alloc_ctor(0, 1, 8);
} else {
 x_52 = x_49;
}
lean_ctor_set(x_52, 0, x_51);
lean_ctor_set_uint64(x_52, sizeof(void*)*1, x_48);
x_53 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_53, 0, x_40);
lean_ctor_set(x_53, 1, x_41);
lean_ctor_set(x_53, 2, x_42);
lean_ctor_set(x_53, 3, x_43);
lean_ctor_set(x_53, 4, x_52);
lean_ctor_set(x_53, 5, x_44);
lean_ctor_set(x_53, 6, x_45);
lean_ctor_set(x_53, 7, x_46);
lean_ctor_set(x_53, 8, x_47);
x_54 = lean_st_ref_set(x_8, x_53);
x_55 = lean_box(0);
lean_ctor_set(x_21, 0, x_55);
return x_21;
}
}
else
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; uint64_t x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_56 = lean_ctor_get(x_21, 0);
lean_inc(x_56);
lean_dec(x_21);
x_57 = lean_st_ref_take(x_8);
x_58 = lean_ctor_get(x_57, 4);
lean_inc_ref(x_58);
x_59 = lean_ctor_get(x_57, 0);
lean_inc_ref(x_59);
x_60 = lean_ctor_get(x_57, 1);
lean_inc(x_60);
x_61 = lean_ctor_get(x_57, 2);
lean_inc_ref(x_61);
x_62 = lean_ctor_get(x_57, 3);
lean_inc_ref(x_62);
x_63 = lean_ctor_get(x_57, 5);
lean_inc_ref(x_63);
x_64 = lean_ctor_get(x_57, 6);
lean_inc_ref(x_64);
x_65 = lean_ctor_get(x_57, 7);
lean_inc_ref(x_65);
x_66 = lean_ctor_get(x_57, 8);
lean_inc_ref(x_66);
if (lean_is_exclusive(x_57)) {
 lean_ctor_release(x_57, 0);
 lean_ctor_release(x_57, 1);
 lean_ctor_release(x_57, 2);
 lean_ctor_release(x_57, 3);
 lean_ctor_release(x_57, 4);
 lean_ctor_release(x_57, 5);
 lean_ctor_release(x_57, 6);
 lean_ctor_release(x_57, 7);
 lean_ctor_release(x_57, 8);
 x_67 = x_57;
} else {
 lean_dec_ref(x_57);
 x_67 = lean_box(0);
}
x_68 = lean_ctor_get_uint64(x_58, sizeof(void*)*1);
if (lean_is_exclusive(x_58)) {
 lean_ctor_release(x_58, 0);
 x_69 = x_58;
} else {
 lean_dec_ref(x_58);
 x_69 = lean_box(0);
}
x_70 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_70, 0, x_3);
lean_ctor_set(x_70, 1, x_56);
x_71 = l_Lean_PersistentArray_push___redArg(x_1, x_70);
if (lean_is_scalar(x_69)) {
 x_72 = lean_alloc_ctor(0, 1, 8);
} else {
 x_72 = x_69;
}
lean_ctor_set(x_72, 0, x_71);
lean_ctor_set_uint64(x_72, sizeof(void*)*1, x_68);
if (lean_is_scalar(x_67)) {
 x_73 = lean_alloc_ctor(0, 9, 0);
} else {
 x_73 = x_67;
}
lean_ctor_set(x_73, 0, x_59);
lean_ctor_set(x_73, 1, x_60);
lean_ctor_set(x_73, 2, x_61);
lean_ctor_set(x_73, 3, x_62);
lean_ctor_set(x_73, 4, x_72);
lean_ctor_set(x_73, 5, x_63);
lean_ctor_set(x_73, 6, x_64);
lean_ctor_set(x_73, 7, x_65);
lean_ctor_set(x_73, 8, x_66);
x_74 = lean_st_ref_set(x_8, x_73);
x_75 = lean_box(0);
x_76 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_76, 0, x_75);
return x_76;
}
}
else
{
lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; uint8_t x_89; lean_object* x_90; uint8_t x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; size_t x_99; size_t x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; uint64_t x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; 
x_77 = lean_ctor_get(x_7, 0);
x_78 = lean_ctor_get(x_7, 1);
x_79 = lean_ctor_get(x_7, 2);
x_80 = lean_ctor_get(x_7, 3);
x_81 = lean_ctor_get(x_7, 4);
x_82 = lean_ctor_get(x_7, 5);
x_83 = lean_ctor_get(x_7, 6);
x_84 = lean_ctor_get(x_7, 7);
x_85 = lean_ctor_get(x_7, 8);
x_86 = lean_ctor_get(x_7, 9);
x_87 = lean_ctor_get(x_7, 10);
x_88 = lean_ctor_get(x_7, 11);
x_89 = lean_ctor_get_uint8(x_7, sizeof(void*)*14);
x_90 = lean_ctor_get(x_7, 12);
x_91 = lean_ctor_get_uint8(x_7, sizeof(void*)*14 + 1);
x_92 = lean_ctor_get(x_7, 13);
lean_inc(x_92);
lean_inc(x_90);
lean_inc(x_88);
lean_inc(x_87);
lean_inc(x_86);
lean_inc(x_85);
lean_inc(x_84);
lean_inc(x_83);
lean_inc(x_82);
lean_inc(x_81);
lean_inc(x_80);
lean_inc(x_79);
lean_inc(x_78);
lean_inc(x_77);
lean_dec(x_7);
x_93 = lean_st_ref_get(x_8);
x_94 = lean_ctor_get(x_93, 4);
lean_inc_ref(x_94);
lean_dec(x_93);
x_95 = lean_ctor_get(x_94, 0);
lean_inc_ref(x_95);
lean_dec_ref(x_94);
x_96 = l_Lean_replaceRef(x_3, x_82);
lean_dec(x_82);
x_97 = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(x_97, 0, x_77);
lean_ctor_set(x_97, 1, x_78);
lean_ctor_set(x_97, 2, x_79);
lean_ctor_set(x_97, 3, x_80);
lean_ctor_set(x_97, 4, x_81);
lean_ctor_set(x_97, 5, x_96);
lean_ctor_set(x_97, 6, x_83);
lean_ctor_set(x_97, 7, x_84);
lean_ctor_set(x_97, 8, x_85);
lean_ctor_set(x_97, 9, x_86);
lean_ctor_set(x_97, 10, x_87);
lean_ctor_set(x_97, 11, x_88);
lean_ctor_set(x_97, 12, x_90);
lean_ctor_set(x_97, 13, x_92);
lean_ctor_set_uint8(x_97, sizeof(void*)*14, x_89);
lean_ctor_set_uint8(x_97, sizeof(void*)*14 + 1, x_91);
x_98 = l_Lean_PersistentArray_toArray___redArg(x_95);
lean_dec_ref(x_95);
x_99 = lean_array_size(x_98);
x_100 = 0;
x_101 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3_spec__3(x_99, x_100, x_98);
x_102 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_102, 0, x_2);
lean_ctor_set(x_102, 1, x_4);
lean_ctor_set(x_102, 2, x_101);
x_103 = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0(x_102, x_5, x_6, x_97, x_8);
lean_dec_ref(x_97);
x_104 = lean_ctor_get(x_103, 0);
lean_inc(x_104);
if (lean_is_exclusive(x_103)) {
 lean_ctor_release(x_103, 0);
 x_105 = x_103;
} else {
 lean_dec_ref(x_103);
 x_105 = lean_box(0);
}
x_106 = lean_st_ref_take(x_8);
x_107 = lean_ctor_get(x_106, 4);
lean_inc_ref(x_107);
x_108 = lean_ctor_get(x_106, 0);
lean_inc_ref(x_108);
x_109 = lean_ctor_get(x_106, 1);
lean_inc(x_109);
x_110 = lean_ctor_get(x_106, 2);
lean_inc_ref(x_110);
x_111 = lean_ctor_get(x_106, 3);
lean_inc_ref(x_111);
x_112 = lean_ctor_get(x_106, 5);
lean_inc_ref(x_112);
x_113 = lean_ctor_get(x_106, 6);
lean_inc_ref(x_113);
x_114 = lean_ctor_get(x_106, 7);
lean_inc_ref(x_114);
x_115 = lean_ctor_get(x_106, 8);
lean_inc_ref(x_115);
if (lean_is_exclusive(x_106)) {
 lean_ctor_release(x_106, 0);
 lean_ctor_release(x_106, 1);
 lean_ctor_release(x_106, 2);
 lean_ctor_release(x_106, 3);
 lean_ctor_release(x_106, 4);
 lean_ctor_release(x_106, 5);
 lean_ctor_release(x_106, 6);
 lean_ctor_release(x_106, 7);
 lean_ctor_release(x_106, 8);
 x_116 = x_106;
} else {
 lean_dec_ref(x_106);
 x_116 = lean_box(0);
}
x_117 = lean_ctor_get_uint64(x_107, sizeof(void*)*1);
if (lean_is_exclusive(x_107)) {
 lean_ctor_release(x_107, 0);
 x_118 = x_107;
} else {
 lean_dec_ref(x_107);
 x_118 = lean_box(0);
}
x_119 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_119, 0, x_3);
lean_ctor_set(x_119, 1, x_104);
x_120 = l_Lean_PersistentArray_push___redArg(x_1, x_119);
if (lean_is_scalar(x_118)) {
 x_121 = lean_alloc_ctor(0, 1, 8);
} else {
 x_121 = x_118;
}
lean_ctor_set(x_121, 0, x_120);
lean_ctor_set_uint64(x_121, sizeof(void*)*1, x_117);
if (lean_is_scalar(x_116)) {
 x_122 = lean_alloc_ctor(0, 9, 0);
} else {
 x_122 = x_116;
}
lean_ctor_set(x_122, 0, x_108);
lean_ctor_set(x_122, 1, x_109);
lean_ctor_set(x_122, 2, x_110);
lean_ctor_set(x_122, 3, x_111);
lean_ctor_set(x_122, 4, x_121);
lean_ctor_set(x_122, 5, x_112);
lean_ctor_set(x_122, 6, x_113);
lean_ctor_set(x_122, 7, x_114);
lean_ctor_set(x_122, 8, x_115);
x_123 = lean_st_ref_set(x_8, x_122);
x_124 = lean_box(0);
if (lean_is_scalar(x_105)) {
 x_125 = lean_alloc_ctor(0, 1, 0);
} else {
 x_125 = x_105;
}
lean_ctor_set(x_125, 0, x_124);
return x_125;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_ctor_set_tag(x_1, 1);
return x_1;
}
else
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec(x_1);
x_5 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_1);
if (x_6 == 0)
{
lean_ctor_set_tag(x_1, 0);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec(x_1);
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
}
}
static lean_object* _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("<exception thrown while producing trace node message>", 53, 53);
return x_1;
}
}
static lean_object* _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static double _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__2() {
_start:
{
lean_object* x_1; double x_2; 
x_1 = lean_unsigned_to_nat(1000000000u);
x_2 = lean_float_of_nat(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_trace_profiler;
return x_1;
}
}
static lean_object* _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_trace_profiler_threshold;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__6(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = l_Lean_KVMap_find(x_1, x_3);
if (lean_obj_tag(x_5) == 0)
{
lean_inc(x_4);
return x_4;
}
else
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
if (lean_obj_tag(x_6) == 3)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
return x_7;
}
else
{
lean_dec(x_6);
lean_inc(x_4);
return x_4;
}
}
}
}
static double _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__5() {
_start:
{
lean_object* x_1; double x_2; 
x_1 = lean_unsigned_to_nat(1000u);
x_2 = lean_float_of_nat(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(32u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__0;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__2() {
_start:
{
size_t x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = 5;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__0;
x_4 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__1;
x_5 = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_2);
lean_ctor_set_usize(x_5, 4, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lean_st_ref_get(x_1);
x_4 = lean_ctor_get(x_3, 4);
lean_inc_ref(x_4);
lean_dec(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lean_st_ref_take(x_1);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; uint8_t x_9; 
x_8 = lean_ctor_get(x_6, 4);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_ctor_get(x_8, 0);
lean_dec(x_10);
x_11 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__2;
lean_ctor_set(x_8, 0, x_11);
x_12 = lean_st_ref_set(x_1, x_6);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_5);
return x_13;
}
else
{
uint64_t x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_14 = lean_ctor_get_uint64(x_8, sizeof(void*)*1);
lean_dec(x_8);
x_15 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__2;
x_16 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set_uint64(x_16, sizeof(void*)*1, x_14);
lean_ctor_set(x_6, 4, x_16);
x_17 = lean_st_ref_set(x_1, x_6);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_5);
return x_18;
}
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; uint64_t x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_19 = lean_ctor_get(x_6, 4);
x_20 = lean_ctor_get(x_6, 0);
x_21 = lean_ctor_get(x_6, 1);
x_22 = lean_ctor_get(x_6, 2);
x_23 = lean_ctor_get(x_6, 3);
x_24 = lean_ctor_get(x_6, 5);
x_25 = lean_ctor_get(x_6, 6);
x_26 = lean_ctor_get(x_6, 7);
x_27 = lean_ctor_get(x_6, 8);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_19);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_6);
x_28 = lean_ctor_get_uint64(x_19, sizeof(void*)*1);
if (lean_is_exclusive(x_19)) {
 lean_ctor_release(x_19, 0);
 x_29 = x_19;
} else {
 lean_dec_ref(x_19);
 x_29 = lean_box(0);
}
x_30 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__2;
if (lean_is_scalar(x_29)) {
 x_31 = lean_alloc_ctor(0, 1, 8);
} else {
 x_31 = x_29;
}
lean_ctor_set(x_31, 0, x_30);
lean_ctor_set_uint64(x_31, sizeof(void*)*1, x_28);
x_32 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_32, 0, x_20);
lean_ctor_set(x_32, 1, x_21);
lean_ctor_set(x_32, 2, x_22);
lean_ctor_set(x_32, 3, x_23);
lean_ctor_set(x_32, 4, x_31);
lean_ctor_set(x_32, 5, x_24);
lean_ctor_set(x_32, 6, x_25);
lean_ctor_set(x_32, 7, x_26);
lean_ctor_set(x_32, 8, x_27);
x_33 = lean_st_ref_set(x_1, x_32);
x_34 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_34, 0, x_5);
return x_34;
}
}
}
static lean_object* _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_trace_profiler_useHeartbeats;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_34; double x_35; uint8_t x_36; lean_object* x_37; double x_38; lean_object* x_39; lean_object* x_40; lean_object* x_45; lean_object* x_46; double x_47; uint8_t x_48; lean_object* x_49; double x_50; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; double x_73; lean_object* x_74; double x_75; lean_object* x_76; uint8_t x_77; lean_object* x_78; lean_object* x_79; lean_object* x_84; double x_85; lean_object* x_86; double x_87; lean_object* x_88; uint8_t x_89; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; double x_98; uint8_t x_99; lean_object* x_100; double x_101; uint8_t x_102; lean_object* x_136; lean_object* x_137; double x_138; uint8_t x_139; lean_object* x_140; double x_141; double x_142; uint8_t x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; double x_168; lean_object* x_169; double x_170; lean_object* x_171; uint8_t x_172; lean_object* x_173; uint8_t x_174; lean_object* x_208; double x_209; lean_object* x_210; double x_211; lean_object* x_212; uint8_t x_213; double x_214; uint8_t x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; uint8_t x_258; 
x_14 = lean_ctor_get(x_11, 2);
x_15 = lean_ctor_get(x_11, 5);
lean_inc(x_1);
x_94 = lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1___redArg(x_1, x_11);
x_95 = lean_ctor_get(x_94, 0);
lean_inc(x_95);
lean_dec_ref(x_94);
x_258 = lean_unbox(x_95);
if (x_258 == 0)
{
lean_object* x_259; uint8_t x_260; 
x_259 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__3;
x_260 = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(x_14, x_259);
if (x_260 == 0)
{
lean_object* x_261; 
lean_dec(x_95);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
lean_dec(x_1);
x_261 = lean_apply_8(x_3, x_6, x_7, x_8, x_9, x_10, x_11, x_12, lean_box(0));
return x_261;
}
else
{
lean_inc(x_15);
goto block_257;
}
}
else
{
lean_inc(x_15);
goto block_257;
}
block_33:
{
lean_object* x_28; 
lean_dec(x_22);
lean_dec(x_21);
lean_dec_ref(x_20);
x_28 = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___redArg(x_17, x_19, x_15, x_18, x_23, x_24, x_25, x_26);
lean_dec(x_26);
lean_dec(x_24);
lean_dec_ref(x_23);
if (lean_obj_tag(x_28) == 0)
{
lean_object* x_29; 
lean_dec_ref(x_28);
x_29 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_16);
return x_29;
}
else
{
uint8_t x_30; 
lean_dec_ref(x_16);
x_30 = !lean_is_exclusive(x_28);
if (x_30 == 0)
{
return x_28;
}
else
{
lean_object* x_31; lean_object* x_32; 
x_31 = lean_ctor_get(x_28, 0);
lean_inc(x_31);
lean_dec(x_28);
x_32 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_32, 0, x_31);
return x_32;
}
}
}
block_44:
{
if (x_36 == 0)
{
double x_41; lean_object* x_42; 
x_41 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0;
x_42 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_42, 0, x_1);
lean_ctor_set(x_42, 1, x_5);
lean_ctor_set_float(x_42, sizeof(void*)*2, x_41);
lean_ctor_set_float(x_42, sizeof(void*)*2 + 8, x_41);
lean_ctor_set_uint8(x_42, sizeof(void*)*2 + 16, x_4);
x_16 = x_34;
x_17 = x_37;
x_18 = x_39;
x_19 = x_42;
x_20 = x_6;
x_21 = x_7;
x_22 = x_8;
x_23 = x_9;
x_24 = x_10;
x_25 = x_11;
x_26 = x_12;
x_27 = lean_box(0);
goto block_33;
}
else
{
lean_object* x_43; 
x_43 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_43, 0, x_1);
lean_ctor_set(x_43, 1, x_5);
lean_ctor_set_float(x_43, sizeof(void*)*2, x_38);
lean_ctor_set_float(x_43, sizeof(void*)*2 + 8, x_35);
lean_ctor_set_uint8(x_43, sizeof(void*)*2 + 16, x_4);
x_16 = x_34;
x_17 = x_37;
x_18 = x_39;
x_19 = x_43;
x_20 = x_6;
x_21 = x_7;
x_22 = x_8;
x_23 = x_9;
x_24 = x_10;
x_25 = x_11;
x_26 = x_12;
x_27 = lean_box(0);
goto block_33;
}
}
block_54:
{
lean_object* x_51; 
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_46);
x_51 = lean_apply_9(x_2, x_46, x_6, x_7, x_8, x_9, x_10, x_11, x_12, lean_box(0));
if (lean_obj_tag(x_51) == 0)
{
lean_object* x_52; 
x_52 = lean_ctor_get(x_51, 0);
lean_inc(x_52);
lean_dec_ref(x_51);
x_34 = x_46;
x_35 = x_47;
x_36 = x_48;
x_37 = x_49;
x_38 = x_50;
x_39 = x_52;
x_40 = lean_box(0);
goto block_44;
}
else
{
lean_object* x_53; 
lean_dec_ref(x_51);
x_53 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__1;
x_34 = x_46;
x_35 = x_47;
x_36 = x_48;
x_37 = x_49;
x_38 = x_50;
x_39 = x_53;
x_40 = lean_box(0);
goto block_44;
}
}
block_72:
{
lean_object* x_67; 
lean_dec(x_61);
lean_dec(x_60);
lean_dec_ref(x_59);
x_67 = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___redArg(x_56, x_58, x_15, x_57, x_62, x_63, x_64, x_65);
lean_dec(x_65);
lean_dec(x_63);
lean_dec_ref(x_62);
if (lean_obj_tag(x_67) == 0)
{
lean_object* x_68; 
lean_dec_ref(x_67);
x_68 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_55);
return x_68;
}
else
{
uint8_t x_69; 
lean_dec_ref(x_55);
x_69 = !lean_is_exclusive(x_67);
if (x_69 == 0)
{
return x_67;
}
else
{
lean_object* x_70; lean_object* x_71; 
x_70 = lean_ctor_get(x_67, 0);
lean_inc(x_70);
lean_dec(x_67);
x_71 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_71, 0, x_70);
return x_71;
}
}
}
block_83:
{
if (x_77 == 0)
{
double x_80; lean_object* x_81; 
x_80 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0;
x_81 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_81, 0, x_1);
lean_ctor_set(x_81, 1, x_5);
lean_ctor_set_float(x_81, sizeof(void*)*2, x_80);
lean_ctor_set_float(x_81, sizeof(void*)*2 + 8, x_80);
lean_ctor_set_uint8(x_81, sizeof(void*)*2 + 16, x_4);
x_55 = x_74;
x_56 = x_76;
x_57 = x_78;
x_58 = x_81;
x_59 = x_6;
x_60 = x_7;
x_61 = x_8;
x_62 = x_9;
x_63 = x_10;
x_64 = x_11;
x_65 = x_12;
x_66 = lean_box(0);
goto block_72;
}
else
{
lean_object* x_82; 
x_82 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_82, 0, x_1);
lean_ctor_set(x_82, 1, x_5);
lean_ctor_set_float(x_82, sizeof(void*)*2, x_73);
lean_ctor_set_float(x_82, sizeof(void*)*2 + 8, x_75);
lean_ctor_set_uint8(x_82, sizeof(void*)*2 + 16, x_4);
x_55 = x_74;
x_56 = x_76;
x_57 = x_78;
x_58 = x_82;
x_59 = x_6;
x_60 = x_7;
x_61 = x_8;
x_62 = x_9;
x_63 = x_10;
x_64 = x_11;
x_65 = x_12;
x_66 = lean_box(0);
goto block_72;
}
}
block_93:
{
lean_object* x_90; 
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_84);
x_90 = lean_apply_9(x_2, x_84, x_6, x_7, x_8, x_9, x_10, x_11, x_12, lean_box(0));
if (lean_obj_tag(x_90) == 0)
{
lean_object* x_91; 
x_91 = lean_ctor_get(x_90, 0);
lean_inc(x_91);
lean_dec_ref(x_90);
x_73 = x_85;
x_74 = x_84;
x_75 = x_87;
x_76 = x_86;
x_77 = x_89;
x_78 = x_91;
x_79 = lean_box(0);
goto block_83;
}
else
{
lean_object* x_92; 
lean_dec_ref(x_90);
x_92 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__1;
x_73 = x_85;
x_74 = x_84;
x_75 = x_87;
x_76 = x_86;
x_77 = x_89;
x_78 = x_92;
x_79 = lean_box(0);
goto block_83;
}
}
block_135:
{
uint8_t x_103; 
x_103 = lean_unbox(x_95);
lean_dec(x_95);
if (x_103 == 0)
{
if (x_102 == 0)
{
lean_object* x_104; uint8_t x_105; 
lean_dec(x_15);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
lean_dec(x_1);
x_104 = lean_st_ref_take(x_12);
x_105 = !lean_is_exclusive(x_104);
if (x_105 == 0)
{
lean_object* x_106; uint8_t x_107; 
x_106 = lean_ctor_get(x_104, 4);
x_107 = !lean_is_exclusive(x_106);
if (x_107 == 0)
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; 
x_108 = lean_ctor_get(x_106, 0);
x_109 = l_Lean_PersistentArray_append___redArg(x_100, x_108);
lean_dec_ref(x_108);
lean_ctor_set(x_106, 0, x_109);
x_110 = lean_st_ref_set(x_12, x_104);
lean_dec(x_12);
x_111 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_97);
return x_111;
}
else
{
uint64_t x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; 
x_112 = lean_ctor_get_uint64(x_106, sizeof(void*)*1);
x_113 = lean_ctor_get(x_106, 0);
lean_inc(x_113);
lean_dec(x_106);
x_114 = l_Lean_PersistentArray_append___redArg(x_100, x_113);
lean_dec_ref(x_113);
x_115 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_115, 0, x_114);
lean_ctor_set_uint64(x_115, sizeof(void*)*1, x_112);
lean_ctor_set(x_104, 4, x_115);
x_116 = lean_st_ref_set(x_12, x_104);
lean_dec(x_12);
x_117 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_97);
return x_117;
}
}
else
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; uint64_t x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; 
x_118 = lean_ctor_get(x_104, 4);
x_119 = lean_ctor_get(x_104, 0);
x_120 = lean_ctor_get(x_104, 1);
x_121 = lean_ctor_get(x_104, 2);
x_122 = lean_ctor_get(x_104, 3);
x_123 = lean_ctor_get(x_104, 5);
x_124 = lean_ctor_get(x_104, 6);
x_125 = lean_ctor_get(x_104, 7);
x_126 = lean_ctor_get(x_104, 8);
lean_inc(x_126);
lean_inc(x_125);
lean_inc(x_124);
lean_inc(x_123);
lean_inc(x_118);
lean_inc(x_122);
lean_inc(x_121);
lean_inc(x_120);
lean_inc(x_119);
lean_dec(x_104);
x_127 = lean_ctor_get_uint64(x_118, sizeof(void*)*1);
x_128 = lean_ctor_get(x_118, 0);
lean_inc_ref(x_128);
if (lean_is_exclusive(x_118)) {
 lean_ctor_release(x_118, 0);
 x_129 = x_118;
} else {
 lean_dec_ref(x_118);
 x_129 = lean_box(0);
}
x_130 = l_Lean_PersistentArray_append___redArg(x_100, x_128);
lean_dec_ref(x_128);
if (lean_is_scalar(x_129)) {
 x_131 = lean_alloc_ctor(0, 1, 8);
} else {
 x_131 = x_129;
}
lean_ctor_set(x_131, 0, x_130);
lean_ctor_set_uint64(x_131, sizeof(void*)*1, x_127);
x_132 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_132, 0, x_119);
lean_ctor_set(x_132, 1, x_120);
lean_ctor_set(x_132, 2, x_121);
lean_ctor_set(x_132, 3, x_122);
lean_ctor_set(x_132, 4, x_131);
lean_ctor_set(x_132, 5, x_123);
lean_ctor_set(x_132, 6, x_124);
lean_ctor_set(x_132, 7, x_125);
lean_ctor_set(x_132, 8, x_126);
x_133 = lean_st_ref_set(x_12, x_132);
lean_dec(x_12);
x_134 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_97);
return x_134;
}
}
else
{
x_45 = lean_box(0);
x_46 = x_97;
x_47 = x_98;
x_48 = x_99;
x_49 = x_100;
x_50 = x_101;
goto block_54;
}
}
else
{
x_45 = lean_box(0);
x_46 = x_97;
x_47 = x_98;
x_48 = x_99;
x_49 = x_100;
x_50 = x_101;
goto block_54;
}
}
block_145:
{
double x_143; uint8_t x_144; 
x_143 = lean_float_sub(x_138, x_141);
x_144 = lean_float_decLt(x_142, x_143);
x_96 = lean_box(0);
x_97 = x_137;
x_98 = x_138;
x_99 = x_139;
x_100 = x_140;
x_101 = x_141;
x_102 = x_144;
goto block_135;
}
block_167:
{
lean_object* x_151; double x_152; double x_153; double x_154; double x_155; double x_156; lean_object* x_157; uint8_t x_158; 
x_151 = lean_io_mono_nanos_now();
x_152 = lean_float_of_nat(x_148);
x_153 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__2;
x_154 = lean_float_div(x_152, x_153);
x_155 = lean_float_of_nat(x_151);
x_156 = lean_float_div(x_155, x_153);
x_157 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__3;
x_158 = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(x_14, x_157);
if (x_158 == 0)
{
x_96 = lean_box(0);
x_97 = x_149;
x_98 = x_156;
x_99 = x_158;
x_100 = x_147;
x_101 = x_154;
x_102 = x_158;
goto block_135;
}
else
{
if (x_146 == 0)
{
lean_object* x_159; lean_object* x_160; double x_161; double x_162; double x_163; 
x_159 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__4;
x_160 = lp_aesop_Lean_Option_get___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__6(x_14, x_159);
x_161 = lean_float_of_nat(x_160);
x_162 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__5;
x_163 = lean_float_div(x_161, x_162);
x_136 = lean_box(0);
x_137 = x_149;
x_138 = x_156;
x_139 = x_158;
x_140 = x_147;
x_141 = x_154;
x_142 = x_163;
goto block_145;
}
else
{
lean_object* x_164; lean_object* x_165; double x_166; 
x_164 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__4;
x_165 = lp_aesop_Lean_Option_get___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__6(x_14, x_164);
x_166 = lean_float_of_nat(x_165);
x_136 = lean_box(0);
x_137 = x_149;
x_138 = x_156;
x_139 = x_158;
x_140 = x_147;
x_141 = x_154;
x_142 = x_166;
goto block_145;
}
}
}
block_207:
{
uint8_t x_175; 
x_175 = lean_unbox(x_95);
lean_dec(x_95);
if (x_175 == 0)
{
if (x_174 == 0)
{
lean_object* x_176; uint8_t x_177; 
lean_dec(x_15);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
lean_dec(x_1);
x_176 = lean_st_ref_take(x_12);
x_177 = !lean_is_exclusive(x_176);
if (x_177 == 0)
{
lean_object* x_178; uint8_t x_179; 
x_178 = lean_ctor_get(x_176, 4);
x_179 = !lean_is_exclusive(x_178);
if (x_179 == 0)
{
lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; 
x_180 = lean_ctor_get(x_178, 0);
x_181 = l_Lean_PersistentArray_append___redArg(x_171, x_180);
lean_dec_ref(x_180);
lean_ctor_set(x_178, 0, x_181);
x_182 = lean_st_ref_set(x_12, x_176);
lean_dec(x_12);
x_183 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_169);
return x_183;
}
else
{
uint64_t x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; 
x_184 = lean_ctor_get_uint64(x_178, sizeof(void*)*1);
x_185 = lean_ctor_get(x_178, 0);
lean_inc(x_185);
lean_dec(x_178);
x_186 = l_Lean_PersistentArray_append___redArg(x_171, x_185);
lean_dec_ref(x_185);
x_187 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_187, 0, x_186);
lean_ctor_set_uint64(x_187, sizeof(void*)*1, x_184);
lean_ctor_set(x_176, 4, x_187);
x_188 = lean_st_ref_set(x_12, x_176);
lean_dec(x_12);
x_189 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_169);
return x_189;
}
}
else
{
lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; uint64_t x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; 
x_190 = lean_ctor_get(x_176, 4);
x_191 = lean_ctor_get(x_176, 0);
x_192 = lean_ctor_get(x_176, 1);
x_193 = lean_ctor_get(x_176, 2);
x_194 = lean_ctor_get(x_176, 3);
x_195 = lean_ctor_get(x_176, 5);
x_196 = lean_ctor_get(x_176, 6);
x_197 = lean_ctor_get(x_176, 7);
x_198 = lean_ctor_get(x_176, 8);
lean_inc(x_198);
lean_inc(x_197);
lean_inc(x_196);
lean_inc(x_195);
lean_inc(x_190);
lean_inc(x_194);
lean_inc(x_193);
lean_inc(x_192);
lean_inc(x_191);
lean_dec(x_176);
x_199 = lean_ctor_get_uint64(x_190, sizeof(void*)*1);
x_200 = lean_ctor_get(x_190, 0);
lean_inc_ref(x_200);
if (lean_is_exclusive(x_190)) {
 lean_ctor_release(x_190, 0);
 x_201 = x_190;
} else {
 lean_dec_ref(x_190);
 x_201 = lean_box(0);
}
x_202 = l_Lean_PersistentArray_append___redArg(x_171, x_200);
lean_dec_ref(x_200);
if (lean_is_scalar(x_201)) {
 x_203 = lean_alloc_ctor(0, 1, 8);
} else {
 x_203 = x_201;
}
lean_ctor_set(x_203, 0, x_202);
lean_ctor_set_uint64(x_203, sizeof(void*)*1, x_199);
x_204 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_204, 0, x_191);
lean_ctor_set(x_204, 1, x_192);
lean_ctor_set(x_204, 2, x_193);
lean_ctor_set(x_204, 3, x_194);
lean_ctor_set(x_204, 4, x_203);
lean_ctor_set(x_204, 5, x_195);
lean_ctor_set(x_204, 6, x_196);
lean_ctor_set(x_204, 7, x_197);
lean_ctor_set(x_204, 8, x_198);
x_205 = lean_st_ref_set(x_12, x_204);
lean_dec(x_12);
x_206 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_169);
return x_206;
}
}
else
{
x_84 = x_169;
x_85 = x_168;
x_86 = x_171;
x_87 = x_170;
x_88 = lean_box(0);
x_89 = x_172;
goto block_93;
}
}
else
{
x_84 = x_169;
x_85 = x_168;
x_86 = x_171;
x_87 = x_170;
x_88 = lean_box(0);
x_89 = x_172;
goto block_93;
}
}
block_217:
{
double x_215; uint8_t x_216; 
x_215 = lean_float_sub(x_211, x_209);
x_216 = lean_float_decLt(x_214, x_215);
x_168 = x_209;
x_169 = x_208;
x_170 = x_211;
x_171 = x_210;
x_172 = x_213;
x_173 = lean_box(0);
x_174 = x_216;
goto block_207;
}
block_236:
{
lean_object* x_223; double x_224; double x_225; lean_object* x_226; uint8_t x_227; 
x_223 = lean_io_get_num_heartbeats();
x_224 = lean_float_of_nat(x_219);
x_225 = lean_float_of_nat(x_223);
x_226 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__3;
x_227 = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(x_14, x_226);
if (x_227 == 0)
{
x_168 = x_224;
x_169 = x_221;
x_170 = x_225;
x_171 = x_220;
x_172 = x_227;
x_173 = lean_box(0);
x_174 = x_227;
goto block_207;
}
else
{
if (x_218 == 0)
{
lean_object* x_228; lean_object* x_229; double x_230; double x_231; double x_232; 
x_228 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__4;
x_229 = lp_aesop_Lean_Option_get___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__6(x_14, x_228);
x_230 = lean_float_of_nat(x_229);
x_231 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__5;
x_232 = lean_float_div(x_230, x_231);
x_208 = x_221;
x_209 = x_224;
x_210 = x_220;
x_211 = x_225;
x_212 = lean_box(0);
x_213 = x_227;
x_214 = x_232;
goto block_217;
}
else
{
lean_object* x_233; lean_object* x_234; double x_235; 
x_233 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__4;
x_234 = lp_aesop_Lean_Option_get___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__6(x_14, x_233);
x_235 = lean_float_of_nat(x_234);
x_208 = x_221;
x_209 = x_224;
x_210 = x_220;
x_211 = x_225;
x_212 = lean_box(0);
x_213 = x_227;
x_214 = x_235;
goto block_217;
}
}
}
block_257:
{
lean_object* x_237; lean_object* x_238; lean_object* x_239; uint8_t x_240; 
x_237 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg(x_12);
x_238 = lean_ctor_get(x_237, 0);
lean_inc(x_238);
lean_dec_ref(x_237);
x_239 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__6;
x_240 = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(x_14, x_239);
if (x_240 == 0)
{
lean_object* x_241; lean_object* x_242; 
x_241 = lean_io_mono_nanos_now();
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_242 = lean_apply_8(x_3, x_6, x_7, x_8, x_9, x_10, x_11, x_12, lean_box(0));
if (lean_obj_tag(x_242) == 0)
{
uint8_t x_243; 
x_243 = !lean_is_exclusive(x_242);
if (x_243 == 0)
{
lean_ctor_set_tag(x_242, 1);
x_146 = x_240;
x_147 = x_238;
x_148 = x_241;
x_149 = x_242;
x_150 = lean_box(0);
goto block_167;
}
else
{
lean_object* x_244; lean_object* x_245; 
x_244 = lean_ctor_get(x_242, 0);
lean_inc(x_244);
lean_dec(x_242);
x_245 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_245, 0, x_244);
x_146 = x_240;
x_147 = x_238;
x_148 = x_241;
x_149 = x_245;
x_150 = lean_box(0);
goto block_167;
}
}
else
{
uint8_t x_246; 
x_246 = !lean_is_exclusive(x_242);
if (x_246 == 0)
{
lean_ctor_set_tag(x_242, 0);
x_146 = x_240;
x_147 = x_238;
x_148 = x_241;
x_149 = x_242;
x_150 = lean_box(0);
goto block_167;
}
else
{
lean_object* x_247; lean_object* x_248; 
x_247 = lean_ctor_get(x_242, 0);
lean_inc(x_247);
lean_dec(x_242);
x_248 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_248, 0, x_247);
x_146 = x_240;
x_147 = x_238;
x_148 = x_241;
x_149 = x_248;
x_150 = lean_box(0);
goto block_167;
}
}
}
else
{
lean_object* x_249; lean_object* x_250; 
x_249 = lean_io_get_num_heartbeats();
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_250 = lean_apply_8(x_3, x_6, x_7, x_8, x_9, x_10, x_11, x_12, lean_box(0));
if (lean_obj_tag(x_250) == 0)
{
uint8_t x_251; 
x_251 = !lean_is_exclusive(x_250);
if (x_251 == 0)
{
lean_ctor_set_tag(x_250, 1);
x_218 = x_240;
x_219 = x_249;
x_220 = x_238;
x_221 = x_250;
x_222 = lean_box(0);
goto block_236;
}
else
{
lean_object* x_252; lean_object* x_253; 
x_252 = lean_ctor_get(x_250, 0);
lean_inc(x_252);
lean_dec(x_250);
x_253 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_253, 0, x_252);
x_218 = x_240;
x_219 = x_249;
x_220 = x_238;
x_221 = x_253;
x_222 = lean_box(0);
goto block_236;
}
}
else
{
uint8_t x_254; 
x_254 = !lean_is_exclusive(x_250);
if (x_254 == 0)
{
lean_ctor_set_tag(x_250, 0);
x_218 = x_240;
x_219 = x_249;
x_220 = x_238;
x_221 = x_250;
x_222 = lean_box(0);
goto block_236;
}
else
{
lean_object* x_255; lean_object* x_256; 
x_255 = lean_ctor_get(x_250, 0);
lean_inc(x_255);
lean_dec(x_250);
x_256 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_256, 0, x_255);
x_218 = x_240;
x_219 = x_249;
x_220 = x_238;
x_221 = x_256;
x_222 = lean_box(0);
goto block_236;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_15; 
x_15 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___redArg(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_8; 
x_8 = lean_usize_dec_lt(x_4, x_3);
if (x_8 == 0)
{
lean_object* x_9; 
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_5);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_array_uget(x_2, x_4);
x_11 = lp_aesop_Aesop_eraseForwardRuleMatch___redArg(x_10, x_6);
lean_dec(x_10);
if (lean_obj_tag(x_11) == 0)
{
size_t x_12; size_t x_13; 
lean_dec_ref(x_11);
x_12 = 1;
x_13 = lean_usize_add(x_4, x_12);
{
size_t _tmp_3 = x_13;
lean_object* _tmp_4 = x_1;
x_4 = _tmp_3;
x_5 = _tmp_4;
}
goto _start;
}
else
{
return x_11;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_7);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___redArg(x_1, x_2, x_3, x_4, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg(x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_runNormRule_spec__8(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = l_Lean_KVMap_find(x_1, x_3);
if (lean_obj_tag(x_5) == 0)
{
lean_inc(x_4);
return x_4;
}
else
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
return x_7;
}
else
{
lean_dec(x_6);
lean_inc(x_4);
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lp_aesop_Aesop_getForwardState___redArg(x_5);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_14);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
x_15 = lp_aesop_Aesop_runNormRuleTac(x_1, x_2, x_13, x_14, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_4);
if (lean_obj_tag(x_15) == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; size_t x_20; size_t x_21; lean_object* x_22; 
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
x_18 = lean_ctor_get(x_16, 1);
lean_inc(x_18);
lean_dec(x_16);
x_19 = lean_box(0);
x_20 = lean_array_size(x_18);
x_21 = 0;
x_22 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___redArg(x_19, x_18, x_20, x_21, x_19, x_5);
lean_dec(x_18);
if (lean_obj_tag(x_22) == 0)
{
uint8_t x_23; 
x_23 = !lean_is_exclusive(x_22);
if (x_23 == 0)
{
lean_object* x_24; 
x_24 = lean_ctor_get(x_22, 0);
lean_dec(x_24);
if (lean_obj_tag(x_17) == 1)
{
uint8_t x_25; 
lean_free_object(x_22);
x_25 = !lean_is_exclusive(x_17);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_26 = lean_ctor_get(x_17, 0);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 0);
lean_inc(x_28);
lean_dec(x_26);
x_29 = lean_ctor_get(x_27, 0);
lean_inc(x_29);
x_30 = lean_ctor_get(x_27, 1);
lean_inc(x_30);
lean_dec(x_27);
x_31 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_32 = lp_aesop_Aesop_modifyForwardState___redArg(x_29, x_31, x_30, x_5);
x_33 = !lean_is_exclusive(x_32);
if (x_33 == 0)
{
lean_object* x_34; lean_object* x_35; 
x_34 = lean_ctor_get(x_32, 0);
lean_dec(x_34);
lean_inc(x_28);
lean_ctor_set(x_17, 0, x_28);
x_35 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_28);
lean_dec(x_28);
if (lean_obj_tag(x_35) == 1)
{
uint8_t x_36; 
lean_free_object(x_32);
x_36 = !lean_is_exclusive(x_35);
if (x_36 == 0)
{
lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_37 = lean_ctor_get(x_35, 0);
x_38 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_39 = !lean_is_exclusive(x_38);
if (x_39 == 0)
{
lean_object* x_40; uint8_t x_41; 
x_40 = lean_ctor_get(x_38, 0);
x_41 = lean_unbox(x_40);
lean_dec(x_40);
if (x_41 == 0)
{
lean_free_object(x_35);
lean_dec(x_37);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
lean_ctor_set(x_38, 0, x_17);
return x_38;
}
else
{
lean_object* x_42; lean_object* x_43; 
lean_free_object(x_38);
x_42 = lean_ctor_get(x_3, 0);
lean_inc(x_42);
lean_dec_ref(x_3);
x_43 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_42, x_35, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_43) == 0)
{
uint8_t x_44; 
x_44 = !lean_is_exclusive(x_43);
if (x_44 == 0)
{
lean_object* x_45; 
x_45 = lean_ctor_get(x_43, 0);
lean_dec(x_45);
lean_ctor_set(x_43, 0, x_17);
return x_43;
}
else
{
lean_object* x_46; 
lean_dec(x_43);
x_46 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_46, 0, x_17);
return x_46;
}
}
else
{
uint8_t x_47; 
lean_dec_ref(x_17);
x_47 = !lean_is_exclusive(x_43);
if (x_47 == 0)
{
return x_43;
}
else
{
lean_object* x_48; lean_object* x_49; 
x_48 = lean_ctor_get(x_43, 0);
lean_inc(x_48);
lean_dec(x_43);
x_49 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_49, 0, x_48);
return x_49;
}
}
}
}
else
{
lean_object* x_50; uint8_t x_51; 
x_50 = lean_ctor_get(x_38, 0);
lean_inc(x_50);
lean_dec(x_38);
x_51 = lean_unbox(x_50);
lean_dec(x_50);
if (x_51 == 0)
{
lean_object* x_52; 
lean_free_object(x_35);
lean_dec(x_37);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_52 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_52, 0, x_17);
return x_52;
}
else
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_ctor_get(x_3, 0);
lean_inc(x_53);
lean_dec_ref(x_3);
x_54 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_53, x_35, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_54) == 0)
{
lean_object* x_55; lean_object* x_56; 
if (lean_is_exclusive(x_54)) {
 lean_ctor_release(x_54, 0);
 x_55 = x_54;
} else {
 lean_dec_ref(x_54);
 x_55 = lean_box(0);
}
if (lean_is_scalar(x_55)) {
 x_56 = lean_alloc_ctor(0, 1, 0);
} else {
 x_56 = x_55;
}
lean_ctor_set(x_56, 0, x_17);
return x_56;
}
else
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; 
lean_dec_ref(x_17);
x_57 = lean_ctor_get(x_54, 0);
lean_inc(x_57);
if (lean_is_exclusive(x_54)) {
 lean_ctor_release(x_54, 0);
 x_58 = x_54;
} else {
 lean_dec_ref(x_54);
 x_58 = lean_box(0);
}
if (lean_is_scalar(x_58)) {
 x_59 = lean_alloc_ctor(1, 1, 0);
} else {
 x_59 = x_58;
}
lean_ctor_set(x_59, 0, x_57);
return x_59;
}
}
}
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; uint8_t x_64; 
x_60 = lean_ctor_get(x_35, 0);
lean_inc(x_60);
lean_dec(x_35);
x_61 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_62 = lean_ctor_get(x_61, 0);
lean_inc(x_62);
if (lean_is_exclusive(x_61)) {
 lean_ctor_release(x_61, 0);
 x_63 = x_61;
} else {
 lean_dec_ref(x_61);
 x_63 = lean_box(0);
}
x_64 = lean_unbox(x_62);
lean_dec(x_62);
if (x_64 == 0)
{
lean_object* x_65; 
lean_dec(x_60);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_63)) {
 x_65 = lean_alloc_ctor(0, 1, 0);
} else {
 x_65 = x_63;
}
lean_ctor_set(x_65, 0, x_17);
return x_65;
}
else
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; 
lean_dec(x_63);
x_66 = lean_ctor_get(x_3, 0);
lean_inc(x_66);
lean_dec_ref(x_3);
x_67 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_67, 0, x_60);
x_68 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_66, x_67, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_68) == 0)
{
lean_object* x_69; lean_object* x_70; 
if (lean_is_exclusive(x_68)) {
 lean_ctor_release(x_68, 0);
 x_69 = x_68;
} else {
 lean_dec_ref(x_68);
 x_69 = lean_box(0);
}
if (lean_is_scalar(x_69)) {
 x_70 = lean_alloc_ctor(0, 1, 0);
} else {
 x_70 = x_69;
}
lean_ctor_set(x_70, 0, x_17);
return x_70;
}
else
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; 
lean_dec_ref(x_17);
x_71 = lean_ctor_get(x_68, 0);
lean_inc(x_71);
if (lean_is_exclusive(x_68)) {
 lean_ctor_release(x_68, 0);
 x_72 = x_68;
} else {
 lean_dec_ref(x_68);
 x_72 = lean_box(0);
}
if (lean_is_scalar(x_72)) {
 x_73 = lean_alloc_ctor(1, 1, 0);
} else {
 x_73 = x_72;
}
lean_ctor_set(x_73, 0, x_71);
return x_73;
}
}
}
}
else
{
lean_dec(x_35);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
lean_ctor_set(x_32, 0, x_17);
return x_32;
}
}
else
{
lean_object* x_74; 
lean_dec(x_32);
lean_inc(x_28);
lean_ctor_set(x_17, 0, x_28);
x_74 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_28);
lean_dec(x_28);
if (lean_obj_tag(x_74) == 1)
{
lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; uint8_t x_80; 
x_75 = lean_ctor_get(x_74, 0);
lean_inc(x_75);
if (lean_is_exclusive(x_74)) {
 lean_ctor_release(x_74, 0);
 x_76 = x_74;
} else {
 lean_dec_ref(x_74);
 x_76 = lean_box(0);
}
x_77 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_78 = lean_ctor_get(x_77, 0);
lean_inc(x_78);
if (lean_is_exclusive(x_77)) {
 lean_ctor_release(x_77, 0);
 x_79 = x_77;
} else {
 lean_dec_ref(x_77);
 x_79 = lean_box(0);
}
x_80 = lean_unbox(x_78);
lean_dec(x_78);
if (x_80 == 0)
{
lean_object* x_81; 
lean_dec(x_76);
lean_dec(x_75);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_79)) {
 x_81 = lean_alloc_ctor(0, 1, 0);
} else {
 x_81 = x_79;
}
lean_ctor_set(x_81, 0, x_17);
return x_81;
}
else
{
lean_object* x_82; lean_object* x_83; lean_object* x_84; 
lean_dec(x_79);
x_82 = lean_ctor_get(x_3, 0);
lean_inc(x_82);
lean_dec_ref(x_3);
if (lean_is_scalar(x_76)) {
 x_83 = lean_alloc_ctor(1, 1, 0);
} else {
 x_83 = x_76;
}
lean_ctor_set(x_83, 0, x_75);
x_84 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_82, x_83, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_84) == 0)
{
lean_object* x_85; lean_object* x_86; 
if (lean_is_exclusive(x_84)) {
 lean_ctor_release(x_84, 0);
 x_85 = x_84;
} else {
 lean_dec_ref(x_84);
 x_85 = lean_box(0);
}
if (lean_is_scalar(x_85)) {
 x_86 = lean_alloc_ctor(0, 1, 0);
} else {
 x_86 = x_85;
}
lean_ctor_set(x_86, 0, x_17);
return x_86;
}
else
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; 
lean_dec_ref(x_17);
x_87 = lean_ctor_get(x_84, 0);
lean_inc(x_87);
if (lean_is_exclusive(x_84)) {
 lean_ctor_release(x_84, 0);
 x_88 = x_84;
} else {
 lean_dec_ref(x_84);
 x_88 = lean_box(0);
}
if (lean_is_scalar(x_88)) {
 x_89 = lean_alloc_ctor(1, 1, 0);
} else {
 x_89 = x_88;
}
lean_ctor_set(x_89, 0, x_87);
return x_89;
}
}
}
else
{
lean_object* x_90; 
lean_dec(x_74);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_90 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_90, 0, x_17);
return x_90;
}
}
}
else
{
lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; 
x_91 = lean_ctor_get(x_17, 0);
lean_inc(x_91);
lean_dec(x_17);
x_92 = lean_ctor_get(x_91, 1);
lean_inc(x_92);
x_93 = lean_ctor_get(x_91, 0);
lean_inc(x_93);
lean_dec(x_91);
x_94 = lean_ctor_get(x_92, 0);
lean_inc(x_94);
x_95 = lean_ctor_get(x_92, 1);
lean_inc(x_95);
lean_dec(x_92);
x_96 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_97 = lp_aesop_Aesop_modifyForwardState___redArg(x_94, x_96, x_95, x_5);
if (lean_is_exclusive(x_97)) {
 lean_ctor_release(x_97, 0);
 x_98 = x_97;
} else {
 lean_dec_ref(x_97);
 x_98 = lean_box(0);
}
lean_inc(x_93);
x_99 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_99, 0, x_93);
x_100 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_93);
lean_dec(x_93);
if (lean_obj_tag(x_100) == 1)
{
lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; uint8_t x_106; 
lean_dec(x_98);
x_101 = lean_ctor_get(x_100, 0);
lean_inc(x_101);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 x_102 = x_100;
} else {
 lean_dec_ref(x_100);
 x_102 = lean_box(0);
}
x_103 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_104 = lean_ctor_get(x_103, 0);
lean_inc(x_104);
if (lean_is_exclusive(x_103)) {
 lean_ctor_release(x_103, 0);
 x_105 = x_103;
} else {
 lean_dec_ref(x_103);
 x_105 = lean_box(0);
}
x_106 = lean_unbox(x_104);
lean_dec(x_104);
if (x_106 == 0)
{
lean_object* x_107; 
lean_dec(x_102);
lean_dec(x_101);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_105)) {
 x_107 = lean_alloc_ctor(0, 1, 0);
} else {
 x_107 = x_105;
}
lean_ctor_set(x_107, 0, x_99);
return x_107;
}
else
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; 
lean_dec(x_105);
x_108 = lean_ctor_get(x_3, 0);
lean_inc(x_108);
lean_dec_ref(x_3);
if (lean_is_scalar(x_102)) {
 x_109 = lean_alloc_ctor(1, 1, 0);
} else {
 x_109 = x_102;
}
lean_ctor_set(x_109, 0, x_101);
x_110 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_108, x_109, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_110) == 0)
{
lean_object* x_111; lean_object* x_112; 
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_111 = x_110;
} else {
 lean_dec_ref(x_110);
 x_111 = lean_box(0);
}
if (lean_is_scalar(x_111)) {
 x_112 = lean_alloc_ctor(0, 1, 0);
} else {
 x_112 = x_111;
}
lean_ctor_set(x_112, 0, x_99);
return x_112;
}
else
{
lean_object* x_113; lean_object* x_114; lean_object* x_115; 
lean_dec_ref(x_99);
x_113 = lean_ctor_get(x_110, 0);
lean_inc(x_113);
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_114 = x_110;
} else {
 lean_dec_ref(x_110);
 x_114 = lean_box(0);
}
if (lean_is_scalar(x_114)) {
 x_115 = lean_alloc_ctor(1, 1, 0);
} else {
 x_115 = x_114;
}
lean_ctor_set(x_115, 0, x_113);
return x_115;
}
}
}
else
{
lean_object* x_116; 
lean_dec(x_100);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_98)) {
 x_116 = lean_alloc_ctor(0, 1, 0);
} else {
 x_116 = x_98;
}
lean_ctor_set(x_116, 0, x_99);
return x_116;
}
}
}
else
{
lean_object* x_117; 
lean_dec(x_17);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_117 = lean_box(0);
lean_ctor_set(x_22, 0, x_117);
return x_22;
}
}
else
{
lean_dec(x_22);
if (lean_obj_tag(x_17) == 1)
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; 
x_118 = lean_ctor_get(x_17, 0);
lean_inc(x_118);
if (lean_is_exclusive(x_17)) {
 lean_ctor_release(x_17, 0);
 x_119 = x_17;
} else {
 lean_dec_ref(x_17);
 x_119 = lean_box(0);
}
x_120 = lean_ctor_get(x_118, 1);
lean_inc(x_120);
x_121 = lean_ctor_get(x_118, 0);
lean_inc(x_121);
lean_dec(x_118);
x_122 = lean_ctor_get(x_120, 0);
lean_inc(x_122);
x_123 = lean_ctor_get(x_120, 1);
lean_inc(x_123);
lean_dec(x_120);
x_124 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_125 = lp_aesop_Aesop_modifyForwardState___redArg(x_122, x_124, x_123, x_5);
if (lean_is_exclusive(x_125)) {
 lean_ctor_release(x_125, 0);
 x_126 = x_125;
} else {
 lean_dec_ref(x_125);
 x_126 = lean_box(0);
}
lean_inc(x_121);
if (lean_is_scalar(x_119)) {
 x_127 = lean_alloc_ctor(1, 1, 0);
} else {
 x_127 = x_119;
}
lean_ctor_set(x_127, 0, x_121);
x_128 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_121);
lean_dec(x_121);
if (lean_obj_tag(x_128) == 1)
{
lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; uint8_t x_134; 
lean_dec(x_126);
x_129 = lean_ctor_get(x_128, 0);
lean_inc(x_129);
if (lean_is_exclusive(x_128)) {
 lean_ctor_release(x_128, 0);
 x_130 = x_128;
} else {
 lean_dec_ref(x_128);
 x_130 = lean_box(0);
}
x_131 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_132 = lean_ctor_get(x_131, 0);
lean_inc(x_132);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 x_133 = x_131;
} else {
 lean_dec_ref(x_131);
 x_133 = lean_box(0);
}
x_134 = lean_unbox(x_132);
lean_dec(x_132);
if (x_134 == 0)
{
lean_object* x_135; 
lean_dec(x_130);
lean_dec(x_129);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_133)) {
 x_135 = lean_alloc_ctor(0, 1, 0);
} else {
 x_135 = x_133;
}
lean_ctor_set(x_135, 0, x_127);
return x_135;
}
else
{
lean_object* x_136; lean_object* x_137; lean_object* x_138; 
lean_dec(x_133);
x_136 = lean_ctor_get(x_3, 0);
lean_inc(x_136);
lean_dec_ref(x_3);
if (lean_is_scalar(x_130)) {
 x_137 = lean_alloc_ctor(1, 1, 0);
} else {
 x_137 = x_130;
}
lean_ctor_set(x_137, 0, x_129);
x_138 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_136, x_137, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_138) == 0)
{
lean_object* x_139; lean_object* x_140; 
if (lean_is_exclusive(x_138)) {
 lean_ctor_release(x_138, 0);
 x_139 = x_138;
} else {
 lean_dec_ref(x_138);
 x_139 = lean_box(0);
}
if (lean_is_scalar(x_139)) {
 x_140 = lean_alloc_ctor(0, 1, 0);
} else {
 x_140 = x_139;
}
lean_ctor_set(x_140, 0, x_127);
return x_140;
}
else
{
lean_object* x_141; lean_object* x_142; lean_object* x_143; 
lean_dec_ref(x_127);
x_141 = lean_ctor_get(x_138, 0);
lean_inc(x_141);
if (lean_is_exclusive(x_138)) {
 lean_ctor_release(x_138, 0);
 x_142 = x_138;
} else {
 lean_dec_ref(x_138);
 x_142 = lean_box(0);
}
if (lean_is_scalar(x_142)) {
 x_143 = lean_alloc_ctor(1, 1, 0);
} else {
 x_143 = x_142;
}
lean_ctor_set(x_143, 0, x_141);
return x_143;
}
}
}
else
{
lean_object* x_144; 
lean_dec(x_128);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_126)) {
 x_144 = lean_alloc_ctor(0, 1, 0);
} else {
 x_144 = x_126;
}
lean_ctor_set(x_144, 0, x_127);
return x_144;
}
}
else
{
lean_object* x_145; lean_object* x_146; 
lean_dec(x_17);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_145 = lean_box(0);
x_146 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_146, 0, x_145);
return x_146;
}
}
}
else
{
uint8_t x_147; 
lean_dec(x_17);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_147 = !lean_is_exclusive(x_22);
if (x_147 == 0)
{
return x_22;
}
else
{
lean_object* x_148; lean_object* x_149; 
x_148 = lean_ctor_get(x_22, 0);
lean_inc(x_148);
lean_dec(x_22);
x_149 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_149, 0, x_148);
return x_149;
}
}
}
else
{
uint8_t x_150; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_150 = !lean_is_exclusive(x_15);
if (x_150 == 0)
{
return x_15;
}
else
{
lean_object* x_151; lean_object* x_152; 
x_151 = lean_ctor_get(x_15, 0);
lean_inc(x_151);
lean_dec(x_15);
x_152 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_152, 0, x_151);
return x_152;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lp_aesop_Aesop_getForwardState___redArg(x_5);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_14);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
x_15 = lp_aesop_Aesop_runNormRuleTac(x_1, x_2, x_13, x_14, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_4);
if (lean_obj_tag(x_15) == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; size_t x_20; size_t x_21; lean_object* x_22; 
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
x_18 = lean_ctor_get(x_16, 1);
lean_inc(x_18);
lean_dec(x_16);
x_19 = lean_box(0);
x_20 = lean_array_size(x_18);
x_21 = 0;
x_22 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___redArg(x_19, x_18, x_20, x_21, x_19, x_5);
lean_dec(x_18);
if (lean_obj_tag(x_22) == 0)
{
uint8_t x_23; 
x_23 = !lean_is_exclusive(x_22);
if (x_23 == 0)
{
lean_object* x_24; 
x_24 = lean_ctor_get(x_22, 0);
lean_dec(x_24);
if (lean_obj_tag(x_17) == 1)
{
uint8_t x_25; 
lean_free_object(x_22);
x_25 = !lean_is_exclusive(x_17);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_26 = lean_ctor_get(x_17, 0);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 0);
lean_inc(x_28);
lean_dec(x_26);
x_29 = lean_ctor_get(x_27, 0);
lean_inc(x_29);
x_30 = lean_ctor_get(x_27, 1);
lean_inc(x_30);
lean_dec(x_27);
x_31 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_32 = lp_aesop_Aesop_modifyForwardState___redArg(x_29, x_31, x_30, x_5);
x_33 = !lean_is_exclusive(x_32);
if (x_33 == 0)
{
lean_object* x_34; lean_object* x_35; 
x_34 = lean_ctor_get(x_32, 0);
lean_dec(x_34);
lean_inc(x_28);
lean_ctor_set(x_17, 0, x_28);
x_35 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_28);
lean_dec(x_28);
if (lean_obj_tag(x_35) == 1)
{
uint8_t x_36; 
lean_free_object(x_32);
x_36 = !lean_is_exclusive(x_35);
if (x_36 == 0)
{
lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_37 = lean_ctor_get(x_35, 0);
x_38 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_39 = !lean_is_exclusive(x_38);
if (x_39 == 0)
{
lean_object* x_40; uint8_t x_41; 
x_40 = lean_ctor_get(x_38, 0);
x_41 = lean_unbox(x_40);
lean_dec(x_40);
if (x_41 == 0)
{
lean_free_object(x_35);
lean_dec(x_37);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
lean_ctor_set(x_38, 0, x_17);
return x_38;
}
else
{
lean_object* x_42; lean_object* x_43; 
lean_free_object(x_38);
x_42 = lean_ctor_get(x_3, 0);
lean_inc(x_42);
lean_dec_ref(x_3);
x_43 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_42, x_35, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_43) == 0)
{
uint8_t x_44; 
x_44 = !lean_is_exclusive(x_43);
if (x_44 == 0)
{
lean_object* x_45; 
x_45 = lean_ctor_get(x_43, 0);
lean_dec(x_45);
lean_ctor_set(x_43, 0, x_17);
return x_43;
}
else
{
lean_object* x_46; 
lean_dec(x_43);
x_46 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_46, 0, x_17);
return x_46;
}
}
else
{
uint8_t x_47; 
lean_dec_ref(x_17);
x_47 = !lean_is_exclusive(x_43);
if (x_47 == 0)
{
return x_43;
}
else
{
lean_object* x_48; lean_object* x_49; 
x_48 = lean_ctor_get(x_43, 0);
lean_inc(x_48);
lean_dec(x_43);
x_49 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_49, 0, x_48);
return x_49;
}
}
}
}
else
{
lean_object* x_50; uint8_t x_51; 
x_50 = lean_ctor_get(x_38, 0);
lean_inc(x_50);
lean_dec(x_38);
x_51 = lean_unbox(x_50);
lean_dec(x_50);
if (x_51 == 0)
{
lean_object* x_52; 
lean_free_object(x_35);
lean_dec(x_37);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_52 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_52, 0, x_17);
return x_52;
}
else
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_ctor_get(x_3, 0);
lean_inc(x_53);
lean_dec_ref(x_3);
x_54 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_53, x_35, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_54) == 0)
{
lean_object* x_55; lean_object* x_56; 
if (lean_is_exclusive(x_54)) {
 lean_ctor_release(x_54, 0);
 x_55 = x_54;
} else {
 lean_dec_ref(x_54);
 x_55 = lean_box(0);
}
if (lean_is_scalar(x_55)) {
 x_56 = lean_alloc_ctor(0, 1, 0);
} else {
 x_56 = x_55;
}
lean_ctor_set(x_56, 0, x_17);
return x_56;
}
else
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; 
lean_dec_ref(x_17);
x_57 = lean_ctor_get(x_54, 0);
lean_inc(x_57);
if (lean_is_exclusive(x_54)) {
 lean_ctor_release(x_54, 0);
 x_58 = x_54;
} else {
 lean_dec_ref(x_54);
 x_58 = lean_box(0);
}
if (lean_is_scalar(x_58)) {
 x_59 = lean_alloc_ctor(1, 1, 0);
} else {
 x_59 = x_58;
}
lean_ctor_set(x_59, 0, x_57);
return x_59;
}
}
}
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; uint8_t x_64; 
x_60 = lean_ctor_get(x_35, 0);
lean_inc(x_60);
lean_dec(x_35);
x_61 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_62 = lean_ctor_get(x_61, 0);
lean_inc(x_62);
if (lean_is_exclusive(x_61)) {
 lean_ctor_release(x_61, 0);
 x_63 = x_61;
} else {
 lean_dec_ref(x_61);
 x_63 = lean_box(0);
}
x_64 = lean_unbox(x_62);
lean_dec(x_62);
if (x_64 == 0)
{
lean_object* x_65; 
lean_dec(x_60);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_63)) {
 x_65 = lean_alloc_ctor(0, 1, 0);
} else {
 x_65 = x_63;
}
lean_ctor_set(x_65, 0, x_17);
return x_65;
}
else
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; 
lean_dec(x_63);
x_66 = lean_ctor_get(x_3, 0);
lean_inc(x_66);
lean_dec_ref(x_3);
x_67 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_67, 0, x_60);
x_68 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_66, x_67, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_68) == 0)
{
lean_object* x_69; lean_object* x_70; 
if (lean_is_exclusive(x_68)) {
 lean_ctor_release(x_68, 0);
 x_69 = x_68;
} else {
 lean_dec_ref(x_68);
 x_69 = lean_box(0);
}
if (lean_is_scalar(x_69)) {
 x_70 = lean_alloc_ctor(0, 1, 0);
} else {
 x_70 = x_69;
}
lean_ctor_set(x_70, 0, x_17);
return x_70;
}
else
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; 
lean_dec_ref(x_17);
x_71 = lean_ctor_get(x_68, 0);
lean_inc(x_71);
if (lean_is_exclusive(x_68)) {
 lean_ctor_release(x_68, 0);
 x_72 = x_68;
} else {
 lean_dec_ref(x_68);
 x_72 = lean_box(0);
}
if (lean_is_scalar(x_72)) {
 x_73 = lean_alloc_ctor(1, 1, 0);
} else {
 x_73 = x_72;
}
lean_ctor_set(x_73, 0, x_71);
return x_73;
}
}
}
}
else
{
lean_dec(x_35);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
lean_ctor_set(x_32, 0, x_17);
return x_32;
}
}
else
{
lean_object* x_74; 
lean_dec(x_32);
lean_inc(x_28);
lean_ctor_set(x_17, 0, x_28);
x_74 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_28);
lean_dec(x_28);
if (lean_obj_tag(x_74) == 1)
{
lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; uint8_t x_80; 
x_75 = lean_ctor_get(x_74, 0);
lean_inc(x_75);
if (lean_is_exclusive(x_74)) {
 lean_ctor_release(x_74, 0);
 x_76 = x_74;
} else {
 lean_dec_ref(x_74);
 x_76 = lean_box(0);
}
x_77 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_78 = lean_ctor_get(x_77, 0);
lean_inc(x_78);
if (lean_is_exclusive(x_77)) {
 lean_ctor_release(x_77, 0);
 x_79 = x_77;
} else {
 lean_dec_ref(x_77);
 x_79 = lean_box(0);
}
x_80 = lean_unbox(x_78);
lean_dec(x_78);
if (x_80 == 0)
{
lean_object* x_81; 
lean_dec(x_76);
lean_dec(x_75);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_79)) {
 x_81 = lean_alloc_ctor(0, 1, 0);
} else {
 x_81 = x_79;
}
lean_ctor_set(x_81, 0, x_17);
return x_81;
}
else
{
lean_object* x_82; lean_object* x_83; lean_object* x_84; 
lean_dec(x_79);
x_82 = lean_ctor_get(x_3, 0);
lean_inc(x_82);
lean_dec_ref(x_3);
if (lean_is_scalar(x_76)) {
 x_83 = lean_alloc_ctor(1, 1, 0);
} else {
 x_83 = x_76;
}
lean_ctor_set(x_83, 0, x_75);
x_84 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_82, x_83, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_84) == 0)
{
lean_object* x_85; lean_object* x_86; 
if (lean_is_exclusive(x_84)) {
 lean_ctor_release(x_84, 0);
 x_85 = x_84;
} else {
 lean_dec_ref(x_84);
 x_85 = lean_box(0);
}
if (lean_is_scalar(x_85)) {
 x_86 = lean_alloc_ctor(0, 1, 0);
} else {
 x_86 = x_85;
}
lean_ctor_set(x_86, 0, x_17);
return x_86;
}
else
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; 
lean_dec_ref(x_17);
x_87 = lean_ctor_get(x_84, 0);
lean_inc(x_87);
if (lean_is_exclusive(x_84)) {
 lean_ctor_release(x_84, 0);
 x_88 = x_84;
} else {
 lean_dec_ref(x_84);
 x_88 = lean_box(0);
}
if (lean_is_scalar(x_88)) {
 x_89 = lean_alloc_ctor(1, 1, 0);
} else {
 x_89 = x_88;
}
lean_ctor_set(x_89, 0, x_87);
return x_89;
}
}
}
else
{
lean_object* x_90; 
lean_dec(x_74);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_90 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_90, 0, x_17);
return x_90;
}
}
}
else
{
lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; 
x_91 = lean_ctor_get(x_17, 0);
lean_inc(x_91);
lean_dec(x_17);
x_92 = lean_ctor_get(x_91, 1);
lean_inc(x_92);
x_93 = lean_ctor_get(x_91, 0);
lean_inc(x_93);
lean_dec(x_91);
x_94 = lean_ctor_get(x_92, 0);
lean_inc(x_94);
x_95 = lean_ctor_get(x_92, 1);
lean_inc(x_95);
lean_dec(x_92);
x_96 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_97 = lp_aesop_Aesop_modifyForwardState___redArg(x_94, x_96, x_95, x_5);
if (lean_is_exclusive(x_97)) {
 lean_ctor_release(x_97, 0);
 x_98 = x_97;
} else {
 lean_dec_ref(x_97);
 x_98 = lean_box(0);
}
lean_inc(x_93);
x_99 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_99, 0, x_93);
x_100 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_93);
lean_dec(x_93);
if (lean_obj_tag(x_100) == 1)
{
lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; uint8_t x_106; 
lean_dec(x_98);
x_101 = lean_ctor_get(x_100, 0);
lean_inc(x_101);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 x_102 = x_100;
} else {
 lean_dec_ref(x_100);
 x_102 = lean_box(0);
}
x_103 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_104 = lean_ctor_get(x_103, 0);
lean_inc(x_104);
if (lean_is_exclusive(x_103)) {
 lean_ctor_release(x_103, 0);
 x_105 = x_103;
} else {
 lean_dec_ref(x_103);
 x_105 = lean_box(0);
}
x_106 = lean_unbox(x_104);
lean_dec(x_104);
if (x_106 == 0)
{
lean_object* x_107; 
lean_dec(x_102);
lean_dec(x_101);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_105)) {
 x_107 = lean_alloc_ctor(0, 1, 0);
} else {
 x_107 = x_105;
}
lean_ctor_set(x_107, 0, x_99);
return x_107;
}
else
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; 
lean_dec(x_105);
x_108 = lean_ctor_get(x_3, 0);
lean_inc(x_108);
lean_dec_ref(x_3);
if (lean_is_scalar(x_102)) {
 x_109 = lean_alloc_ctor(1, 1, 0);
} else {
 x_109 = x_102;
}
lean_ctor_set(x_109, 0, x_101);
x_110 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_108, x_109, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_110) == 0)
{
lean_object* x_111; lean_object* x_112; 
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_111 = x_110;
} else {
 lean_dec_ref(x_110);
 x_111 = lean_box(0);
}
if (lean_is_scalar(x_111)) {
 x_112 = lean_alloc_ctor(0, 1, 0);
} else {
 x_112 = x_111;
}
lean_ctor_set(x_112, 0, x_99);
return x_112;
}
else
{
lean_object* x_113; lean_object* x_114; lean_object* x_115; 
lean_dec_ref(x_99);
x_113 = lean_ctor_get(x_110, 0);
lean_inc(x_113);
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_114 = x_110;
} else {
 lean_dec_ref(x_110);
 x_114 = lean_box(0);
}
if (lean_is_scalar(x_114)) {
 x_115 = lean_alloc_ctor(1, 1, 0);
} else {
 x_115 = x_114;
}
lean_ctor_set(x_115, 0, x_113);
return x_115;
}
}
}
else
{
lean_object* x_116; 
lean_dec(x_100);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_98)) {
 x_116 = lean_alloc_ctor(0, 1, 0);
} else {
 x_116 = x_98;
}
lean_ctor_set(x_116, 0, x_99);
return x_116;
}
}
}
else
{
lean_object* x_117; 
lean_dec(x_17);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_117 = lean_box(0);
lean_ctor_set(x_22, 0, x_117);
return x_22;
}
}
else
{
lean_dec(x_22);
if (lean_obj_tag(x_17) == 1)
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; 
x_118 = lean_ctor_get(x_17, 0);
lean_inc(x_118);
if (lean_is_exclusive(x_17)) {
 lean_ctor_release(x_17, 0);
 x_119 = x_17;
} else {
 lean_dec_ref(x_17);
 x_119 = lean_box(0);
}
x_120 = lean_ctor_get(x_118, 1);
lean_inc(x_120);
x_121 = lean_ctor_get(x_118, 0);
lean_inc(x_121);
lean_dec(x_118);
x_122 = lean_ctor_get(x_120, 0);
lean_inc(x_122);
x_123 = lean_ctor_get(x_120, 1);
lean_inc(x_123);
lean_dec(x_120);
x_124 = lp_aesop_Aesop_modifyForwardState___redArg___closed__0;
x_125 = lp_aesop_Aesop_modifyForwardState___redArg(x_122, x_124, x_123, x_5);
if (lean_is_exclusive(x_125)) {
 lean_ctor_release(x_125, 0);
 x_126 = x_125;
} else {
 lean_dec_ref(x_125);
 x_126 = lean_box(0);
}
lean_inc(x_121);
if (lean_is_scalar(x_119)) {
 x_127 = lean_alloc_ctor(1, 1, 0);
} else {
 x_127 = x_119;
}
lean_ctor_set(x_127, 0, x_121);
x_128 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_121);
lean_dec(x_121);
if (lean_obj_tag(x_128) == 1)
{
lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; uint8_t x_134; 
lean_dec(x_126);
x_129 = lean_ctor_get(x_128, 0);
lean_inc(x_129);
if (lean_is_exclusive(x_128)) {
 lean_ctor_release(x_128, 0);
 x_130 = x_128;
} else {
 lean_dec_ref(x_128);
 x_130 = lean_box(0);
}
x_131 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_3, x_9);
x_132 = lean_ctor_get(x_131, 0);
lean_inc(x_132);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 x_133 = x_131;
} else {
 lean_dec_ref(x_131);
 x_133 = lean_box(0);
}
x_134 = lean_unbox(x_132);
lean_dec(x_132);
if (x_134 == 0)
{
lean_object* x_135; 
lean_dec(x_130);
lean_dec(x_129);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_133)) {
 x_135 = lean_alloc_ctor(0, 1, 0);
} else {
 x_135 = x_133;
}
lean_ctor_set(x_135, 0, x_127);
return x_135;
}
else
{
lean_object* x_136; lean_object* x_137; lean_object* x_138; 
lean_dec(x_133);
x_136 = lean_ctor_get(x_3, 0);
lean_inc(x_136);
lean_dec_ref(x_3);
if (lean_is_scalar(x_130)) {
 x_137 = lean_alloc_ctor(1, 1, 0);
} else {
 x_137 = x_130;
}
lean_ctor_set(x_137, 0, x_129);
x_138 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_136, x_137, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
if (lean_obj_tag(x_138) == 0)
{
lean_object* x_139; lean_object* x_140; 
if (lean_is_exclusive(x_138)) {
 lean_ctor_release(x_138, 0);
 x_139 = x_138;
} else {
 lean_dec_ref(x_138);
 x_139 = lean_box(0);
}
if (lean_is_scalar(x_139)) {
 x_140 = lean_alloc_ctor(0, 1, 0);
} else {
 x_140 = x_139;
}
lean_ctor_set(x_140, 0, x_127);
return x_140;
}
else
{
lean_object* x_141; lean_object* x_142; lean_object* x_143; 
lean_dec_ref(x_127);
x_141 = lean_ctor_get(x_138, 0);
lean_inc(x_141);
if (lean_is_exclusive(x_138)) {
 lean_ctor_release(x_138, 0);
 x_142 = x_138;
} else {
 lean_dec_ref(x_138);
 x_142 = lean_box(0);
}
if (lean_is_scalar(x_142)) {
 x_143 = lean_alloc_ctor(1, 1, 0);
} else {
 x_143 = x_142;
}
lean_ctor_set(x_143, 0, x_141);
return x_143;
}
}
}
else
{
lean_object* x_144; 
lean_dec(x_128);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
if (lean_is_scalar(x_126)) {
 x_144 = lean_alloc_ctor(0, 1, 0);
} else {
 x_144 = x_126;
}
lean_ctor_set(x_144, 0, x_127);
return x_144;
}
}
else
{
lean_object* x_145; lean_object* x_146; 
lean_dec(x_17);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_145 = lean_box(0);
x_146 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_146, 0, x_145);
return x_146;
}
}
}
else
{
uint8_t x_147; 
lean_dec(x_17);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_147 = !lean_is_exclusive(x_22);
if (x_147 == 0)
{
return x_22;
}
else
{
lean_object* x_148; lean_object* x_149; 
x_148 = lean_ctor_get(x_22, 0);
lean_inc(x_148);
lean_dec(x_22);
x_149 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_149, 0, x_148);
return x_149;
}
}
}
else
{
uint8_t x_150; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
x_150 = !lean_is_exclusive(x_15);
if (x_150 == 0)
{
return x_15;
}
else
{
lean_object* x_151; lean_object* x_152; 
x_151 = lean_ctor_get(x_15, 0);
lean_inc(x_151);
lean_dec(x_15);
x_152 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_152, 0, x_151);
return x_152;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRule___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_aesop_collectStats;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRule___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_TraceOption_stats;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormRule___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_aesop_stats_file;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_runNormRule___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_runNormRule___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; lean_object* x_59; lean_object* x_78; lean_object* x_82; uint8_t x_83; 
x_12 = lean_ctor_get(x_3, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_13);
x_14 = lean_ctor_get(x_3, 2);
lean_inc(x_14);
lean_dec_ref(x_3);
x_15 = lean_ctor_get(x_12, 0);
x_16 = lean_ctor_get(x_9, 2);
lean_inc_ref(x_15);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_15);
x_82 = lp_aesop_Aesop_runNormRule___closed__0;
x_83 = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(x_16, x_82);
if (x_83 == 0)
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; uint8_t x_87; 
x_84 = lp_aesop_Aesop_runNormRule___closed__1;
x_85 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_84, x_9);
x_86 = lean_ctor_get(x_85, 0);
lean_inc(x_86);
x_87 = lean_unbox(x_86);
lean_dec(x_86);
if (x_87 == 0)
{
lean_object* x_88; lean_object* x_89; lean_object* x_90; uint8_t x_91; 
lean_dec_ref(x_85);
x_88 = lp_aesop_Aesop_runNormRule___closed__2;
x_89 = lp_aesop_Lean_Option_get___at___00Aesop_runNormRule_spec__8(x_16, x_88);
x_90 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_91 = lean_string_dec_eq(x_89, x_90);
lean_dec_ref(x_89);
if (x_91 == 0)
{
x_59 = lean_box(0);
goto block_77;
}
else
{
x_18 = lean_box(0);
goto block_28;
}
}
else
{
x_78 = x_85;
goto block_81;
}
}
else
{
x_59 = lean_box(0);
goto block_77;
}
block_28:
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; 
x_19 = lean_ctor_get(x_4, 0);
x_20 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_inc_ref(x_19);
x_22 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_22, 0, x_1);
lean_ctor_set(x_22, 1, x_2);
lean_ctor_set(x_22, 2, x_13);
lean_ctor_set(x_22, 3, x_14);
lean_ctor_set(x_22, 4, x_19);
x_23 = lean_alloc_closure((void*)(lp_aesop_Aesop_runNormRule___lam__1___boxed), 11, 3);
lean_closure_set(x_23, 0, x_12);
lean_closure_set(x_23, 1, x_22);
lean_closure_set(x_23, 2, x_20);
x_24 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed), 10, 1);
lean_closure_set(x_24, 0, x_17);
x_25 = 1;
x_26 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_27 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_21, x_24, x_23, x_25, x_26, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_27;
}
block_58:
{
uint8_t x_35; 
x_35 = !lean_is_exclusive(x_33);
if (x_35 == 0)
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_36 = lean_ctor_get(x_33, 8);
x_37 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_37, 0, x_17);
lean_ctor_set(x_37, 1, x_31);
lean_ctor_set_uint8(x_37, sizeof(void*)*2, x_34);
x_38 = lean_array_push(x_36, x_37);
lean_ctor_set(x_33, 8, x_38);
x_39 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_39, 0, x_30);
lean_ctor_set(x_39, 1, x_33);
x_40 = lean_st_ref_set(x_6, x_39);
lean_dec(x_6);
x_41 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_41, 0, x_29);
return x_41;
}
else
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_42 = lean_ctor_get(x_33, 0);
x_43 = lean_ctor_get(x_33, 1);
x_44 = lean_ctor_get(x_33, 2);
x_45 = lean_ctor_get(x_33, 3);
x_46 = lean_ctor_get(x_33, 4);
x_47 = lean_ctor_get(x_33, 5);
x_48 = lean_ctor_get(x_33, 6);
x_49 = lean_ctor_get(x_33, 7);
x_50 = lean_ctor_get(x_33, 8);
x_51 = lean_ctor_get(x_33, 9);
lean_inc(x_51);
lean_inc(x_50);
lean_inc(x_49);
lean_inc(x_48);
lean_inc(x_47);
lean_inc(x_46);
lean_inc(x_45);
lean_inc(x_44);
lean_inc(x_43);
lean_inc(x_42);
lean_dec(x_33);
x_52 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_52, 0, x_17);
lean_ctor_set(x_52, 1, x_31);
lean_ctor_set_uint8(x_52, sizeof(void*)*2, x_34);
x_53 = lean_array_push(x_50, x_52);
x_54 = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(x_54, 0, x_42);
lean_ctor_set(x_54, 1, x_43);
lean_ctor_set(x_54, 2, x_44);
lean_ctor_set(x_54, 3, x_45);
lean_ctor_set(x_54, 4, x_46);
lean_ctor_set(x_54, 5, x_47);
lean_ctor_set(x_54, 6, x_48);
lean_ctor_set(x_54, 7, x_49);
lean_ctor_set(x_54, 8, x_53);
lean_ctor_set(x_54, 9, x_51);
x_55 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_55, 0, x_30);
lean_ctor_set(x_55, 1, x_54);
x_56 = lean_st_ref_set(x_6, x_55);
lean_dec(x_6);
x_57 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_57, 0, x_29);
return x_57;
}
}
block_77:
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; uint8_t x_67; lean_object* x_68; lean_object* x_69; 
x_60 = lean_io_mono_nanos_now();
x_61 = lean_ctor_get(x_4, 0);
x_62 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_63 = lean_ctor_get(x_62, 0);
lean_inc(x_63);
lean_inc_ref(x_61);
x_64 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_64, 0, x_1);
lean_ctor_set(x_64, 1, x_2);
lean_ctor_set(x_64, 2, x_13);
lean_ctor_set(x_64, 3, x_14);
lean_ctor_set(x_64, 4, x_61);
x_65 = lean_alloc_closure((void*)(lp_aesop_Aesop_runNormRule___lam__0___boxed), 11, 3);
lean_closure_set(x_65, 0, x_12);
lean_closure_set(x_65, 1, x_64);
lean_closure_set(x_65, 2, x_62);
lean_inc_ref(x_17);
x_66 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed), 10, 1);
lean_closure_set(x_66, 0, x_17);
x_67 = 1;
x_68 = lp_aesop_Aesop_withNormTraceNode___closed__43;
lean_inc(x_6);
x_69 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_63, x_66, x_65, x_67, x_68, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_69) == 0)
{
lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
x_70 = lean_ctor_get(x_69, 0);
lean_inc(x_70);
lean_dec_ref(x_69);
x_71 = lean_io_mono_nanos_now();
x_72 = lean_st_ref_take(x_6);
x_73 = lean_ctor_get(x_72, 0);
lean_inc_ref(x_73);
x_74 = lean_ctor_get(x_72, 1);
lean_inc_ref(x_74);
lean_dec(x_72);
x_75 = lean_nat_sub(x_71, x_60);
lean_dec(x_60);
lean_dec(x_71);
if (lean_obj_tag(x_70) == 0)
{
uint8_t x_76; 
x_76 = 0;
x_29 = x_70;
x_30 = x_73;
x_31 = x_75;
x_32 = lean_box(0);
x_33 = x_74;
x_34 = x_76;
goto block_58;
}
else
{
x_29 = x_70;
x_30 = x_73;
x_31 = x_75;
x_32 = lean_box(0);
x_33 = x_74;
x_34 = x_67;
goto block_58;
}
}
else
{
lean_dec(x_60);
lean_dec_ref(x_17);
lean_dec(x_6);
return x_69;
}
}
block_81:
{
lean_object* x_79; uint8_t x_80; 
x_79 = lean_ctor_get(x_78, 0);
lean_inc(x_79);
lean_dec_ref(x_78);
x_80 = lean_unbox(x_79);
lean_dec(x_79);
if (x_80 == 0)
{
x_18 = lean_box(0);
goto block_28;
}
else
{
x_59 = lean_box(0);
goto block_77;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
uint8_t x_15; lean_object* x_16; 
x_15 = lean_unbox(x_5);
x_16 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1(x_1, x_2, x_3, x_4, x_15, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
size_t x_14; size_t x_15; lean_object* x_16; 
x_14 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_15 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_16 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0(x_1, x_2, x_14, x_15, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_2);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Lean_isTracingEnabledFor___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__1___redArg(x_1, x_2);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_MonadExcept_ofExcept___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__5___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_runNormRule_spec__8___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Lean_Option_get___at___00Aesop_runNormRule_spec__8(x_1, x_2);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__6___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Lean_Option_get___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__6(x_1, x_2);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
size_t x_8; size_t x_9; lean_object* x_10; 
x_8 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_9 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_10 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormRule_spec__0___redArg(x_1, x_2, x_8, x_9, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; lean_object* x_6; 
x_4 = lean_unbox_usize(x_1);
lean_dec(x_1);
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3_spec__3(x_4, x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__3___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormRule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_runNormRule(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_14; lean_object* x_15; 
x_14 = lean_unbox(x_4);
x_15 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_1, x_2, x_3, x_14, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_15;
}
}
static lean_object* _init_lp_aesop_Aesop_runFirstNormRule___closed__0() {
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
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runFirstNormRule_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, size_t x_6, size_t x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_17; 
x_17 = lean_usize_dec_lt(x_7, x_6);
if (x_17 == 0)
{
lean_object* x_18; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_8);
return x_18;
}
else
{
lean_object* x_19; lean_object* x_20; 
lean_dec_ref(x_8);
x_19 = lean_array_uget(x_5, x_7);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_19);
lean_inc_ref(x_2);
lean_inc(x_1);
x_20 = lp_aesop_Aesop_runNormRule(x_1, x_2, x_19, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_20) == 0)
{
uint8_t x_21; 
x_21 = !lean_is_exclusive(x_20);
if (x_21 == 0)
{
lean_object* x_22; 
x_22 = lean_ctor_get(x_20, 0);
if (lean_obj_tag(x_22) == 1)
{
lean_object* x_23; uint8_t x_24; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
x_23 = lean_ctor_get(x_19, 0);
lean_inc(x_23);
lean_dec(x_19);
x_24 = !lean_is_exclusive(x_22);
if (x_24 == 0)
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_25 = lean_ctor_get(x_22, 0);
x_26 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_26);
lean_dec(x_23);
x_27 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_27, 0, x_26);
x_28 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set(x_28, 1, x_25);
lean_ctor_set(x_22, 0, x_28);
x_29 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_29, 0, x_22);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_29);
lean_ctor_set(x_30, 1, x_3);
lean_ctor_set(x_20, 0, x_30);
return x_20;
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_31 = lean_ctor_get(x_22, 0);
lean_inc(x_31);
lean_dec(x_22);
x_32 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_32);
lean_dec(x_23);
x_33 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_33, 0, x_32);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_31);
x_35 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_35, 0, x_34);
x_36 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_36, 0, x_35);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_36);
lean_ctor_set(x_37, 1, x_3);
lean_ctor_set(x_20, 0, x_37);
return x_20;
}
}
else
{
size_t x_38; size_t x_39; 
lean_free_object(x_20);
lean_dec(x_22);
lean_dec(x_19);
x_38 = 1;
x_39 = lean_usize_add(x_7, x_38);
lean_inc_ref(x_4);
{
size_t _tmp_6 = x_39;
lean_object* _tmp_7 = x_4;
x_7 = _tmp_6;
x_8 = _tmp_7;
}
goto _start;
}
}
else
{
lean_object* x_41; 
x_41 = lean_ctor_get(x_20, 0);
lean_inc(x_41);
lean_dec(x_20);
if (lean_obj_tag(x_41) == 1)
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
x_42 = lean_ctor_get(x_19, 0);
lean_inc(x_42);
lean_dec(x_19);
x_43 = lean_ctor_get(x_41, 0);
lean_inc(x_43);
if (lean_is_exclusive(x_41)) {
 lean_ctor_release(x_41, 0);
 x_44 = x_41;
} else {
 lean_dec_ref(x_41);
 x_44 = lean_box(0);
}
x_45 = lean_ctor_get(x_42, 0);
lean_inc_ref(x_45);
lean_dec(x_42);
x_46 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_46, 0, x_45);
x_47 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_47, 0, x_46);
lean_ctor_set(x_47, 1, x_43);
if (lean_is_scalar(x_44)) {
 x_48 = lean_alloc_ctor(1, 1, 0);
} else {
 x_48 = x_44;
}
lean_ctor_set(x_48, 0, x_47);
x_49 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_49, 0, x_48);
x_50 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_50, 0, x_49);
lean_ctor_set(x_50, 1, x_3);
x_51 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_51, 0, x_50);
return x_51;
}
else
{
size_t x_52; size_t x_53; 
lean_dec(x_41);
lean_dec(x_19);
x_52 = 1;
x_53 = lean_usize_add(x_7, x_52);
lean_inc_ref(x_4);
{
size_t _tmp_6 = x_53;
lean_object* _tmp_7 = x_4;
x_7 = _tmp_6;
x_8 = _tmp_7;
}
goto _start;
}
}
}
else
{
uint8_t x_55; 
lean_dec(x_19);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
x_55 = !lean_is_exclusive(x_20);
if (x_55 == 0)
{
return x_20;
}
else
{
lean_object* x_56; lean_object* x_57; 
x_56 = lean_ctor_get(x_20, 0);
lean_inc(x_56);
lean_dec(x_20);
x_57 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_57, 0, x_56);
return x_57;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstNormRule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; size_t x_15; size_t x_16; lean_object* x_17; 
x_12 = lean_box(0);
x_13 = lean_box(0);
x_14 = lp_aesop_Aesop_runFirstNormRule___closed__0;
x_15 = lean_array_size(x_3);
x_16 = 0;
x_17 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runFirstNormRule_spec__0(x_1, x_2, x_13, x_14, x_3, x_15, x_16, x_14, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_17) == 0)
{
uint8_t x_18; 
x_18 = !lean_is_exclusive(x_17);
if (x_18 == 0)
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_17, 0);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec(x_19);
if (lean_obj_tag(x_20) == 0)
{
lean_ctor_set(x_17, 0, x_12);
return x_17;
}
else
{
lean_object* x_21; 
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_ctor_set(x_17, 0, x_21);
return x_17;
}
}
else
{
lean_object* x_22; lean_object* x_23; 
x_22 = lean_ctor_get(x_17, 0);
lean_inc(x_22);
lean_dec(x_17);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec(x_22);
if (lean_obj_tag(x_23) == 0)
{
lean_object* x_24; 
x_24 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_24, 0, x_12);
return x_24;
}
else
{
lean_object* x_25; lean_object* x_26; 
x_25 = lean_ctor_get(x_23, 0);
lean_inc(x_25);
lean_dec_ref(x_23);
x_26 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_26, 0, x_25);
return x_26;
}
}
}
else
{
uint8_t x_27; 
x_27 = !lean_is_exclusive(x_17);
if (x_27 == 0)
{
return x_17;
}
else
{
lean_object* x_28; lean_object* x_29; 
x_28 = lean_ctor_get(x_17, 0);
lean_inc(x_28);
lean_dec(x_17);
x_29 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_29, 0, x_28);
return x_29;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstNormRule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_runFirstNormRule(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runFirstNormRule_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
size_t x_17; size_t x_18; lean_object* x_19; 
x_17 = lean_unbox_usize(x_6);
lean_dec(x_6);
x_18 = lean_unbox_usize(x_7);
lean_dec(x_7);
x_19 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runFirstNormRule_spec__0(x_1, x_2, x_3, x_4, x_5, x_17, x_18, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_5);
return x_19;
}
}
static lean_object* _init_lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_13; lean_object* x_19; lean_object* x_20; lean_object* x_21; uint8_t x_22; lean_object* x_23; uint8_t x_24; lean_object* x_25; lean_object* x_26; 
x_19 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_19);
x_20 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_20);
lean_dec_ref(x_6);
x_21 = lean_ctor_get(x_20, 0);
lean_inc_ref(x_21);
lean_dec_ref(x_20);
x_22 = lean_ctor_get_uint8(x_19, sizeof(void*)*3 + 1);
x_23 = lean_ctor_get(x_19, 1);
lean_inc(x_23);
lean_dec_ref(x_19);
x_24 = lean_ctor_get_uint8(x_21, sizeof(void*)*6 + 9);
lean_dec_ref(x_21);
x_25 = lean_box(x_22);
lean_inc_ref(x_5);
lean_inc(x_23);
lean_inc(x_1);
x_26 = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnly___boxed), 9, 4);
lean_closure_set(x_26, 0, x_25);
lean_closure_set(x_26, 1, x_1);
lean_closure_set(x_26, 2, x_23);
lean_closure_set(x_26, 3, x_5);
if (x_24 == 0)
{
lean_object* x_27; lean_object* x_28; 
lean_dec(x_23);
lean_dec_ref(x_5);
x_27 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__2;
x_28 = lean_array_push(x_27, x_26);
x_13 = x_28;
goto block_18;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_29 = lean_box(x_22);
lean_inc(x_1);
x_30 = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___boxed), 9, 4);
lean_closure_set(x_30, 0, x_29);
lean_closure_set(x_30, 1, x_1);
lean_closure_set(x_30, 2, x_23);
lean_closure_set(x_30, 3, x_5);
x_31 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__3;
x_32 = lean_array_push(x_31, x_26);
x_33 = lean_array_push(x_32, x_30);
x_13 = x_33;
goto block_18;
}
block_12:
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_10, 0, x_3);
lean_ctor_set(x_10, 1, x_1);
lean_ctor_set(x_10, 2, x_8);
lean_ctor_set(x_10, 3, x_4);
lean_ctor_set(x_10, 4, x_9);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
block_18:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_8 = x_13;
x_9 = x_14;
goto block_12;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_15 = lean_ctor_get(x_2, 0);
lean_inc(x_15);
lean_dec_ref(x_2);
x_16 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__1;
x_17 = lean_array_push(x_16, x_15);
x_8 = x_13;
x_9 = x_17;
goto block_12;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkNormSimpScriptStep(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_mkNormSimpScriptStep(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec(x_7);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkNormSimpScriptStep___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, size_t x_4, size_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_18; uint8_t x_21; 
x_21 = lean_usize_dec_eq(x_4, x_5);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_22 = lean_ctor_get(x_6, 0);
x_23 = lean_ctor_get(x_6, 1);
x_24 = lean_array_uget(x_3, x_4);
x_25 = lean_ctor_get(x_24, 1);
lean_inc(x_25);
lean_dec(x_24);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_1);
x_26 = lp_aesop_Aesop_elabRuleTermForSimpMetaM(x_1, x_25, x_22, x_23, x_2, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_26) == 0)
{
lean_dec_ref(x_6);
x_18 = x_26;
goto block_20;
}
else
{
lean_object* x_27; uint8_t x_28; uint8_t x_30; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
x_30 = l_Lean_Exception_isInterrupt(x_27);
if (x_30 == 0)
{
uint8_t x_31; 
x_31 = l_Lean_Exception_isRuntime(x_27);
x_28 = x_31;
goto block_29;
}
else
{
lean_dec(x_27);
x_28 = x_30;
goto block_29;
}
block_29:
{
if (x_28 == 0)
{
lean_dec_ref(x_26);
x_12 = x_6;
x_13 = lean_box(0);
goto block_17;
}
else
{
lean_dec_ref(x_6);
x_18 = x_26;
goto block_20;
}
}
}
}
else
{
lean_object* x_32; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_1);
x_32 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_32, 0, x_6);
return x_32;
}
block_17:
{
size_t x_14; size_t x_15; 
x_14 = 1;
x_15 = lean_usize_add(x_4, x_14);
x_4 = x_15;
x_6 = x_12;
goto _start;
}
block_20:
{
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_12 = x_19;
x_13 = lean_box(0);
goto block_17;
}
else
{
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_1);
return x_18;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, size_t x_4, size_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_15; 
x_15 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_10, x_11, x_12, x_13);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_3);
lean_ctor_set(x_14, 1, x_4);
x_15 = lean_unsigned_to_nat(0u);
x_16 = lean_array_get_size(x_2);
x_17 = lean_nat_dec_lt(x_15, x_16);
if (x_17 == 0)
{
lean_object* x_18; 
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_1);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_14);
return x_18;
}
else
{
uint8_t x_19; 
x_19 = lean_nat_dec_le(x_16, x_16);
if (x_19 == 0)
{
lean_object* x_20; 
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_1);
x_20 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_20, 0, x_14);
return x_20;
}
else
{
size_t x_21; size_t x_22; lean_object* x_23; 
x_21 = 0;
x_22 = lean_usize_of_nat(x_16);
x_23 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0___redArg(x_1, x_5, x_2, x_21, x_22, x_14, x_9, x_10, x_11, x_12);
return x_23;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
uint8_t x_15; size_t x_16; size_t x_17; lean_object* x_18; 
x_15 = lean_unbox(x_2);
x_16 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_17 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_18 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0(x_1, x_15, x_3, x_16, x_17, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_14; lean_object* x_15; 
x_14 = lean_unbox(x_5);
x_15 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules(x_1, x_2, x_3, x_4, x_14, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_2);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; size_t x_13; size_t x_14; lean_object* x_15; 
x_12 = lean_unbox(x_2);
x_13 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_14 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_15 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules_spec__0___redArg(x_1, x_12, x_3, x_13, x_14, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_3);
return x_15;
}
}
static size_t _init_lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__0() {
_start:
{
size_t x_1; size_t x_2; size_t x_3; 
x_1 = 5;
x_2 = 1;
x_3 = lean_usize_shift_left(x_2, x_1);
return x_3;
}
}
static size_t _init_lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__1() {
_start:
{
size_t x_1; size_t x_2; size_t x_3; 
x_1 = 1;
x_2 = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__0;
x_3 = lean_usize_sub(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_array_get_size(x_1);
x_5 = lean_nat_dec_lt(x_2, x_4);
if (x_5 == 0)
{
lean_dec(x_2);
return x_5;
}
else
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_array_fget_borrowed(x_1, x_2);
x_7 = l_Lean_instBEqMVarId_beq(x_3, x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_unsigned_to_nat(1u);
x_9 = lean_nat_add(x_2, x_8);
lean_dec(x_2);
x_2 = x_9;
goto _start;
}
else
{
lean_dec(x_2);
return x_7;
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg(lean_object* x_1, size_t x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; size_t x_6; size_t x_7; size_t x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_box(2);
x_6 = 5;
x_7 = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__1;
x_8 = lean_usize_land(x_2, x_7);
x_9 = lean_usize_to_nat(x_8);
x_10 = lean_array_get(x_5, x_4, x_9);
lean_dec(x_9);
lean_dec_ref(x_4);
switch (lean_obj_tag(x_10)) {
case 0:
{
lean_object* x_11; uint8_t x_12; 
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = l_Lean_instBEqMVarId_beq(x_3, x_11);
lean_dec(x_11);
return x_12;
}
case 1:
{
lean_object* x_13; size_t x_14; 
x_13 = lean_ctor_get(x_10, 0);
lean_inc(x_13);
lean_dec_ref(x_10);
x_14 = lean_usize_shift_right(x_2, x_6);
x_1 = x_13;
x_2 = x_14;
goto _start;
}
default: 
{
uint8_t x_16; 
x_16 = 0;
return x_16;
}
}
}
else
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_17 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_17);
lean_dec_ref(x_1);
x_18 = lean_unsigned_to_nat(0u);
x_19 = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0___redArg(x_17, x_18, x_3);
lean_dec_ref(x_17);
return x_19;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint64_t x_3; size_t x_4; uint8_t x_5; 
x_3 = l_Lean_instHashableMVarId_hash(x_2);
x_4 = lean_uint64_to_usize(x_3);
x_5 = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg(x_1, x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_4 = lean_st_ref_get(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec(x_4);
x_6 = lean_ctor_get(x_5, 7);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_5, 8);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___redArg(x_6, x_1);
if (x_8 == 0)
{
uint8_t x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___redArg(x_7, x_1);
x_10 = lean_box(x_9);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; 
lean_dec_ref(x_7);
x_12 = lean_box(x_8);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0___redArg(x_1, x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lean_apply_8(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, lean_box(0));
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg___lam__0___boxed), 9, 4);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_3);
lean_closure_set(x_11, 2, x_4);
lean_closure_set(x_11, 3, x_5);
x_12 = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), x_1, x_11, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_12) == 0)
{
return x_12;
}
else
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
return x_12;
}
else
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_12, 0);
lean_inc(x_14);
lean_dec(x_12);
x_15 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; 
x_7 = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0___redArg(x_2, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, size_t x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_13; lean_object* x_14; 
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_21; lean_object* x_22; 
lean_dec_ref(x_1);
x_21 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_21, 0, x_4);
x_22 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; uint8_t x_27; 
lean_dec_ref(x_4);
x_23 = lean_ctor_get(x_3, 0);
x_24 = lean_ctor_get(x_3, 2);
x_25 = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0___redArg(x_23, x_9);
x_26 = lean_ctor_get(x_25, 0);
lean_inc(x_26);
lean_dec_ref(x_25);
x_27 = lean_unbox(x_26);
lean_dec(x_26);
if (x_27 == 0)
{
uint8_t x_28; 
lean_dec_ref(x_1);
x_28 = 1;
x_13 = x_28;
x_14 = lean_box(0);
goto block_20;
}
else
{
lean_inc_ref(x_1);
{
lean_object* _tmp_2 = x_24;
lean_object* _tmp_3 = x_1;
x_3 = _tmp_2;
x_4 = _tmp_3;
}
goto _start;
}
}
block_20:
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_box(x_13);
x_16 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_16, 0, x_15);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_2);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
x_19 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, size_t x_4, size_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_15; 
x_15 = lean_usize_dec_lt(x_5, x_4);
if (x_15 == 0)
{
lean_object* x_16; 
lean_dec_ref(x_1);
x_16 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_16, 0, x_6);
return x_16;
}
else
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_array_uget(x_3, x_5);
lean_inc_ref(x_1);
x_18 = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__4(x_1, x_2, x_17, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_17);
if (lean_obj_tag(x_18) == 0)
{
uint8_t x_19; 
x_19 = !lean_is_exclusive(x_18);
if (x_19 == 0)
{
lean_object* x_20; 
x_20 = lean_ctor_get(x_18, 0);
if (lean_obj_tag(x_20) == 0)
{
lean_object* x_21; 
lean_dec_ref(x_1);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_ctor_set(x_18, 0, x_21);
return x_18;
}
else
{
lean_object* x_22; size_t x_23; size_t x_24; 
lean_free_object(x_18);
x_22 = lean_ctor_get(x_20, 0);
lean_inc(x_22);
lean_dec_ref(x_20);
x_23 = 1;
x_24 = lean_usize_add(x_5, x_23);
x_5 = x_24;
x_6 = x_22;
goto _start;
}
}
else
{
lean_object* x_26; 
x_26 = lean_ctor_get(x_18, 0);
lean_inc(x_26);
lean_dec(x_18);
if (lean_obj_tag(x_26) == 0)
{
lean_object* x_27; lean_object* x_28; 
lean_dec_ref(x_1);
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
x_28 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_28, 0, x_27);
return x_28;
}
else
{
lean_object* x_29; size_t x_30; size_t x_31; 
x_29 = lean_ctor_get(x_26, 0);
lean_inc(x_29);
lean_dec_ref(x_26);
x_30 = 1;
x_31 = lean_usize_add(x_5, x_30);
x_5 = x_31;
x_6 = x_29;
goto _start;
}
}
}
else
{
uint8_t x_33; 
lean_dec_ref(x_1);
x_33 = !lean_is_exclusive(x_18);
if (x_33 == 0)
{
return x_18;
}
else
{
lean_object* x_34; lean_object* x_35; 
x_34 = lean_ctor_get(x_18, 0);
lean_inc(x_34);
lean_dec(x_18);
x_35 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_35, 0, x_34);
return x_35;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; size_t x_13; size_t x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_1, 1);
x_11 = lean_box(0);
x_12 = lp_aesop_Aesop_runNormRuleTac___closed__3;
x_13 = lean_array_size(x_10);
x_14 = 0;
x_15 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__5(x_12, x_11, x_10, x_13, x_14, x_12, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_15) == 0)
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec(x_17);
if (lean_obj_tag(x_18) == 0)
{
uint8_t x_19; lean_object* x_20; 
x_19 = 0;
x_20 = lean_box(x_19);
lean_ctor_set(x_15, 0, x_20);
return x_15;
}
else
{
lean_object* x_21; 
x_21 = lean_ctor_get(x_18, 0);
lean_inc(x_21);
lean_dec_ref(x_18);
lean_ctor_set(x_15, 0, x_21);
return x_15;
}
}
else
{
lean_object* x_22; lean_object* x_23; 
x_22 = lean_ctor_get(x_15, 0);
lean_inc(x_22);
lean_dec(x_15);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec(x_22);
if (lean_obj_tag(x_23) == 0)
{
uint8_t x_24; lean_object* x_25; lean_object* x_26; 
x_24 = 0;
x_25 = lean_box(x_24);
x_26 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_26, 0, x_25);
return x_26;
}
else
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_23, 0);
lean_inc(x_27);
lean_dec_ref(x_23);
x_28 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_28, 0, x_27);
return x_28;
}
}
}
else
{
uint8_t x_29; 
x_29 = !lean_is_exclusive(x_15);
if (x_29 == 0)
{
return x_15;
}
else
{
lean_object* x_30; lean_object* x_31; 
x_30 = lean_ctor_get(x_15, 0);
lean_inc(x_30);
lean_dec(x_15);
x_31 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_31, 0, x_30);
return x_31;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Normalisation simp solved the goal but dropped some metavariables. Skipping normalisation simp.", 95, 95);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normSimpCore___lam__0___closed__1;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("norm simp left the goal unchanged", 33, 33);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normSimpCore___lam__0___closed__3;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(32u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normSimpCore___lam__0___closed__8;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__10() {
_start:
{
size_t x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = 5;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_aesop_Aesop_normSimpCore___lam__0___closed__8;
x_4 = lp_aesop_Aesop_normSimpCore___lam__0___closed__9;
x_5 = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_2);
lean_ctor_set_usize(x_5, 4, x_1);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normSimpCore___lam__0___closed__5;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_normSimpCore___lam__0___closed__10;
x_2 = lp_aesop_Aesop_normSimpCore___lam__0___closed__6;
x_3 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_2);
lean_ctor_set(x_3, 2, x_2);
lean_ctor_set(x_3, 3, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_aesop_Aesop_normSimpCore___lam__0___closed__6;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_normSimpCore___lam__0___closed__11;
x_2 = lp_aesop_Aesop_normSimpCore___lam__0___closed__7;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimpCore___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = l_Lean_Meta_saveState___redArg(x_8, x_10);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_175; lean_object* x_176; lean_object* x_177; uint8_t x_178; lean_object* x_179; uint8_t x_180; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_175 = lean_ctor_get(x_4, 1);
x_176 = lean_ctor_get(x_175, 3);
x_177 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_177);
x_178 = lean_ctor_get_uint8(x_1, sizeof(void*)*3 + 1);
x_179 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_179);
lean_dec_ref(x_1);
x_180 = 1;
if (x_178 == 0)
{
lean_object* x_181; 
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_2);
x_181 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules(x_2, x_176, x_177, x_179, x_178, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_181) == 0)
{
lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; 
x_182 = lean_ctor_get(x_181, 0);
lean_inc(x_182);
lean_dec_ref(x_181);
x_183 = lean_ctor_get(x_182, 0);
lean_inc(x_183);
x_184 = lean_ctor_get(x_182, 1);
lean_inc(x_184);
lean_dec(x_182);
x_185 = lean_box(0);
x_186 = lp_aesop_Aesop_normSimpCore___lam__0___closed__12;
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_2);
x_187 = lp_aesop_Aesop_simpGoalWithAllHypotheses(x_2, x_183, x_184, x_185, x_180, x_186, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_187) == 0)
{
lean_object* x_188; 
x_188 = lean_ctor_get(x_187, 0);
lean_inc(x_188);
lean_dec_ref(x_187);
x_139 = x_188;
x_140 = x_4;
x_141 = x_5;
x_142 = x_6;
x_143 = x_7;
x_144 = x_8;
x_145 = x_9;
x_146 = x_10;
x_147 = lean_box(0);
goto block_174;
}
else
{
uint8_t x_189; 
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_4);
lean_dec(x_2);
x_189 = !lean_is_exclusive(x_187);
if (x_189 == 0)
{
return x_187;
}
else
{
lean_object* x_190; lean_object* x_191; 
x_190 = lean_ctor_get(x_187, 0);
lean_inc(x_190);
lean_dec(x_187);
x_191 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_191, 0, x_190);
return x_191;
}
}
}
else
{
uint8_t x_192; 
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_4);
lean_dec(x_2);
x_192 = !lean_is_exclusive(x_181);
if (x_192 == 0)
{
return x_181;
}
else
{
lean_object* x_193; lean_object* x_194; 
x_193 = lean_ctor_get(x_181, 0);
lean_inc(x_193);
lean_dec(x_181);
x_194 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_194, 0, x_193);
return x_194;
}
}
}
else
{
lean_object* x_195; 
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_2);
x_195 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_normSimpCore_addLocalRules(x_2, x_176, x_177, x_179, x_180, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_195) == 0)
{
lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; 
x_196 = lean_ctor_get(x_195, 0);
lean_inc(x_196);
lean_dec_ref(x_195);
x_197 = lean_ctor_get(x_196, 0);
lean_inc(x_197);
x_198 = lean_ctor_get(x_196, 1);
lean_inc(x_198);
lean_dec(x_196);
x_199 = lp_aesop_Aesop_normSimpCore___lam__0___closed__12;
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_2);
x_200 = lp_aesop_Aesop_simpAll(x_2, x_197, x_198, x_199, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_200) == 0)
{
lean_object* x_201; 
x_201 = lean_ctor_get(x_200, 0);
lean_inc(x_201);
lean_dec_ref(x_200);
x_139 = x_201;
x_140 = x_4;
x_141 = x_5;
x_142 = x_6;
x_143 = x_7;
x_144 = x_8;
x_145 = x_9;
x_146 = x_10;
x_147 = lean_box(0);
goto block_174;
}
else
{
uint8_t x_202; 
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_4);
lean_dec(x_2);
x_202 = !lean_is_exclusive(x_200);
if (x_202 == 0)
{
return x_200;
}
else
{
lean_object* x_203; lean_object* x_204; 
x_203 = lean_ctor_get(x_200, 0);
lean_inc(x_203);
lean_dec(x_200);
x_204 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_204, 0, x_203);
return x_204;
}
}
}
else
{
uint8_t x_205; 
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_4);
lean_dec(x_2);
x_205 = !lean_is_exclusive(x_195);
if (x_205 == 0)
{
return x_195;
}
else
{
lean_object* x_206; lean_object* x_207; 
x_206 = lean_ctor_get(x_195, 0);
lean_inc(x_206);
lean_dec(x_195);
x_207 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_207, 0, x_206);
return x_207;
}
}
}
block_124:
{
lean_object* x_23; 
x_23 = l_Lean_Meta_saveState___redArg(x_19, x_21);
if (lean_obj_tag(x_23) == 0)
{
switch (lean_obj_tag(x_14)) {
case 0:
{
lean_object* x_24; uint8_t x_25; 
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec(x_17);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
lean_dec_ref(x_23);
x_25 = !lean_is_exclusive(x_14);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; 
x_26 = lean_ctor_get(x_14, 0);
x_27 = lean_box(0);
x_28 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg(x_2, x_27, x_13, x_24, x_26, x_15);
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_30 = lean_ctor_get(x_28, 0);
x_31 = lp_aesop_Aesop_normSimpCore___lam__0___closed__0;
x_32 = lean_array_push(x_31, x_30);
lean_ctor_set_tag(x_14, 1);
lean_ctor_set(x_14, 0, x_32);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_14);
x_34 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_28, 0, x_34);
return x_28;
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_35 = lean_ctor_get(x_28, 0);
lean_inc(x_35);
lean_dec(x_28);
x_36 = lp_aesop_Aesop_normSimpCore___lam__0___closed__0;
x_37 = lean_array_push(x_36, x_35);
lean_ctor_set_tag(x_14, 1);
lean_ctor_set(x_14, 0, x_37);
x_38 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_38, 0, x_14);
x_39 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_39, 0, x_38);
x_40 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_40, 0, x_39);
return x_40;
}
}
else
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; 
x_41 = lean_ctor_get(x_14, 0);
lean_inc(x_41);
lean_dec(x_14);
x_42 = lean_box(0);
x_43 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg(x_2, x_42, x_13, x_24, x_41, x_15);
x_44 = lean_ctor_get(x_43, 0);
lean_inc(x_44);
if (lean_is_exclusive(x_43)) {
 lean_ctor_release(x_43, 0);
 x_45 = x_43;
} else {
 lean_dec_ref(x_43);
 x_45 = lean_box(0);
}
x_46 = lp_aesop_Aesop_normSimpCore___lam__0___closed__0;
x_47 = lean_array_push(x_46, x_44);
x_48 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_48, 0, x_47);
x_49 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_49, 0, x_48);
x_50 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_50, 0, x_49);
if (lean_is_scalar(x_45)) {
 x_51 = lean_alloc_ctor(0, 1, 0);
} else {
 x_51 = x_45;
}
lean_ctor_set(x_51, 0, x_50);
return x_51;
}
}
case 1:
{
uint8_t x_52; 
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec(x_17);
lean_dec_ref(x_15);
lean_dec(x_13);
lean_dec(x_2);
x_52 = !lean_is_exclusive(x_23);
if (x_52 == 0)
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_ctor_get(x_23, 0);
lean_dec(x_53);
x_54 = lean_box(0);
lean_ctor_set(x_23, 0, x_54);
return x_23;
}
else
{
lean_object* x_55; lean_object* x_56; 
lean_dec(x_23);
x_55 = lean_box(0);
x_56 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_56, 0, x_55);
return x_56;
}
}
default: 
{
lean_object* x_57; uint8_t x_58; 
x_57 = lean_ctor_get(x_23, 0);
lean_inc(x_57);
lean_dec_ref(x_23);
x_58 = !lean_is_exclusive(x_14);
if (x_58 == 0)
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; uint8_t x_63; 
x_59 = lean_ctor_get(x_14, 0);
x_60 = lean_ctor_get(x_14, 1);
lean_inc(x_59);
x_61 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_61, 0, x_59);
lean_inc_ref(x_15);
lean_inc(x_2);
x_62 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg(x_2, x_61, x_13, x_57, x_60, x_15);
x_63 = !lean_is_exclusive(x_62);
if (x_63 == 0)
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_62, 0);
lean_inc(x_21);
lean_inc_ref(x_20);
lean_inc(x_19);
lean_inc_ref(x_18);
lean_inc(x_17);
lean_inc(x_59);
x_65 = lp_aesop_Aesop_diffGoals(x_2, x_59, x_17, x_18, x_19, x_20, x_21);
if (lean_obj_tag(x_65) == 0)
{
lean_object* x_66; lean_object* x_67; 
x_66 = lean_ctor_get(x_65, 0);
lean_inc(x_66);
lean_dec_ref(x_65);
x_67 = lp_aesop_Aesop_applyDiffToForwardState(x_66, x_15, x_16, x_17, x_18, x_19, x_20, x_21);
if (lean_obj_tag(x_67) == 0)
{
uint8_t x_68; 
x_68 = !lean_is_exclusive(x_67);
if (x_68 == 0)
{
lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; 
x_69 = lean_ctor_get(x_67, 0);
lean_dec(x_69);
x_70 = lp_aesop_Aesop_normSimpCore___lam__0___closed__0;
x_71 = lean_array_push(x_70, x_64);
lean_ctor_set_tag(x_62, 1);
lean_ctor_set(x_62, 0, x_71);
lean_ctor_set_tag(x_14, 0);
lean_ctor_set(x_14, 1, x_62);
x_72 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_72, 0, x_14);
lean_ctor_set(x_67, 0, x_72);
return x_67;
}
else
{
lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
lean_dec(x_67);
x_73 = lp_aesop_Aesop_normSimpCore___lam__0___closed__0;
x_74 = lean_array_push(x_73, x_64);
lean_ctor_set_tag(x_62, 1);
lean_ctor_set(x_62, 0, x_74);
lean_ctor_set_tag(x_14, 0);
lean_ctor_set(x_14, 1, x_62);
x_75 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_75, 0, x_14);
x_76 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_76, 0, x_75);
return x_76;
}
}
else
{
uint8_t x_77; 
lean_free_object(x_62);
lean_dec(x_64);
lean_free_object(x_14);
lean_dec(x_59);
x_77 = !lean_is_exclusive(x_67);
if (x_77 == 0)
{
return x_67;
}
else
{
lean_object* x_78; lean_object* x_79; 
x_78 = lean_ctor_get(x_67, 0);
lean_inc(x_78);
lean_dec(x_67);
x_79 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_79, 0, x_78);
return x_79;
}
}
}
else
{
uint8_t x_80; 
lean_free_object(x_62);
lean_dec(x_64);
lean_free_object(x_14);
lean_dec(x_59);
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec(x_17);
lean_dec_ref(x_15);
x_80 = !lean_is_exclusive(x_65);
if (x_80 == 0)
{
return x_65;
}
else
{
lean_object* x_81; lean_object* x_82; 
x_81 = lean_ctor_get(x_65, 0);
lean_inc(x_81);
lean_dec(x_65);
x_82 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_82, 0, x_81);
return x_82;
}
}
}
else
{
lean_object* x_83; lean_object* x_84; 
x_83 = lean_ctor_get(x_62, 0);
lean_inc(x_83);
lean_dec(x_62);
lean_inc(x_21);
lean_inc_ref(x_20);
lean_inc(x_19);
lean_inc_ref(x_18);
lean_inc(x_17);
lean_inc(x_59);
x_84 = lp_aesop_Aesop_diffGoals(x_2, x_59, x_17, x_18, x_19, x_20, x_21);
if (lean_obj_tag(x_84) == 0)
{
lean_object* x_85; lean_object* x_86; 
x_85 = lean_ctor_get(x_84, 0);
lean_inc(x_85);
lean_dec_ref(x_84);
x_86 = lp_aesop_Aesop_applyDiffToForwardState(x_85, x_15, x_16, x_17, x_18, x_19, x_20, x_21);
if (lean_obj_tag(x_86) == 0)
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; 
if (lean_is_exclusive(x_86)) {
 lean_ctor_release(x_86, 0);
 x_87 = x_86;
} else {
 lean_dec_ref(x_86);
 x_87 = lean_box(0);
}
x_88 = lp_aesop_Aesop_normSimpCore___lam__0___closed__0;
x_89 = lean_array_push(x_88, x_83);
x_90 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_90, 0, x_89);
lean_ctor_set_tag(x_14, 0);
lean_ctor_set(x_14, 1, x_90);
x_91 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_91, 0, x_14);
if (lean_is_scalar(x_87)) {
 x_92 = lean_alloc_ctor(0, 1, 0);
} else {
 x_92 = x_87;
}
lean_ctor_set(x_92, 0, x_91);
return x_92;
}
else
{
lean_object* x_93; lean_object* x_94; lean_object* x_95; 
lean_dec(x_83);
lean_free_object(x_14);
lean_dec(x_59);
x_93 = lean_ctor_get(x_86, 0);
lean_inc(x_93);
if (lean_is_exclusive(x_86)) {
 lean_ctor_release(x_86, 0);
 x_94 = x_86;
} else {
 lean_dec_ref(x_86);
 x_94 = lean_box(0);
}
if (lean_is_scalar(x_94)) {
 x_95 = lean_alloc_ctor(1, 1, 0);
} else {
 x_95 = x_94;
}
lean_ctor_set(x_95, 0, x_93);
return x_95;
}
}
else
{
lean_object* x_96; lean_object* x_97; lean_object* x_98; 
lean_dec(x_83);
lean_free_object(x_14);
lean_dec(x_59);
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec(x_17);
lean_dec_ref(x_15);
x_96 = lean_ctor_get(x_84, 0);
lean_inc(x_96);
if (lean_is_exclusive(x_84)) {
 lean_ctor_release(x_84, 0);
 x_97 = x_84;
} else {
 lean_dec_ref(x_84);
 x_97 = lean_box(0);
}
if (lean_is_scalar(x_97)) {
 x_98 = lean_alloc_ctor(1, 1, 0);
} else {
 x_98 = x_97;
}
lean_ctor_set(x_98, 0, x_96);
return x_98;
}
}
}
else
{
lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; 
x_99 = lean_ctor_get(x_14, 0);
x_100 = lean_ctor_get(x_14, 1);
lean_inc(x_100);
lean_inc(x_99);
lean_dec(x_14);
lean_inc(x_99);
x_101 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_101, 0, x_99);
lean_inc_ref(x_15);
lean_inc(x_2);
x_102 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg(x_2, x_101, x_13, x_57, x_100, x_15);
x_103 = lean_ctor_get(x_102, 0);
lean_inc(x_103);
if (lean_is_exclusive(x_102)) {
 lean_ctor_release(x_102, 0);
 x_104 = x_102;
} else {
 lean_dec_ref(x_102);
 x_104 = lean_box(0);
}
lean_inc(x_21);
lean_inc_ref(x_20);
lean_inc(x_19);
lean_inc_ref(x_18);
lean_inc(x_17);
lean_inc(x_99);
x_105 = lp_aesop_Aesop_diffGoals(x_2, x_99, x_17, x_18, x_19, x_20, x_21);
if (lean_obj_tag(x_105) == 0)
{
lean_object* x_106; lean_object* x_107; 
x_106 = lean_ctor_get(x_105, 0);
lean_inc(x_106);
lean_dec_ref(x_105);
x_107 = lp_aesop_Aesop_applyDiffToForwardState(x_106, x_15, x_16, x_17, x_18, x_19, x_20, x_21);
if (lean_obj_tag(x_107) == 0)
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; 
if (lean_is_exclusive(x_107)) {
 lean_ctor_release(x_107, 0);
 x_108 = x_107;
} else {
 lean_dec_ref(x_107);
 x_108 = lean_box(0);
}
x_109 = lp_aesop_Aesop_normSimpCore___lam__0___closed__0;
x_110 = lean_array_push(x_109, x_103);
if (lean_is_scalar(x_104)) {
 x_111 = lean_alloc_ctor(1, 1, 0);
} else {
 x_111 = x_104;
 lean_ctor_set_tag(x_111, 1);
}
lean_ctor_set(x_111, 0, x_110);
x_112 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_112, 0, x_99);
lean_ctor_set(x_112, 1, x_111);
x_113 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_113, 0, x_112);
if (lean_is_scalar(x_108)) {
 x_114 = lean_alloc_ctor(0, 1, 0);
} else {
 x_114 = x_108;
}
lean_ctor_set(x_114, 0, x_113);
return x_114;
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; 
lean_dec(x_104);
lean_dec(x_103);
lean_dec(x_99);
x_115 = lean_ctor_get(x_107, 0);
lean_inc(x_115);
if (lean_is_exclusive(x_107)) {
 lean_ctor_release(x_107, 0);
 x_116 = x_107;
} else {
 lean_dec_ref(x_107);
 x_116 = lean_box(0);
}
if (lean_is_scalar(x_116)) {
 x_117 = lean_alloc_ctor(1, 1, 0);
} else {
 x_117 = x_116;
}
lean_ctor_set(x_117, 0, x_115);
return x_117;
}
}
else
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; 
lean_dec(x_104);
lean_dec(x_103);
lean_dec(x_99);
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec(x_17);
lean_dec_ref(x_15);
x_118 = lean_ctor_get(x_105, 0);
lean_inc(x_118);
if (lean_is_exclusive(x_105)) {
 lean_ctor_release(x_105, 0);
 x_119 = x_105;
} else {
 lean_dec_ref(x_105);
 x_119 = lean_box(0);
}
if (lean_is_scalar(x_119)) {
 x_120 = lean_alloc_ctor(1, 1, 0);
} else {
 x_120 = x_119;
}
lean_ctor_set(x_120, 0, x_118);
return x_120;
}
}
}
}
}
else
{
uint8_t x_121; 
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec(x_17);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_2);
x_121 = !lean_is_exclusive(x_23);
if (x_121 == 0)
{
return x_23;
}
else
{
lean_object* x_122; lean_object* x_123; 
x_122 = lean_ctor_get(x_23, 0);
lean_inc(x_122);
lean_dec(x_23);
x_123 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_123, 0, x_122);
return x_123;
}
}
}
block_138:
{
lean_object* x_133; 
x_133 = l_Lean_Meta_SavedState_restore___redArg(x_13, x_129, x_131);
if (lean_obj_tag(x_133) == 0)
{
lean_object* x_134; 
lean_dec_ref(x_133);
x_134 = lean_box(1);
x_14 = x_134;
x_15 = x_125;
x_16 = x_126;
x_17 = x_127;
x_18 = x_128;
x_19 = x_129;
x_20 = x_130;
x_21 = x_131;
x_22 = lean_box(0);
goto block_124;
}
else
{
uint8_t x_135; 
lean_dec(x_131);
lean_dec_ref(x_130);
lean_dec(x_129);
lean_dec_ref(x_128);
lean_dec(x_127);
lean_dec_ref(x_125);
lean_dec(x_13);
lean_dec(x_2);
x_135 = !lean_is_exclusive(x_133);
if (x_135 == 0)
{
return x_133;
}
else
{
lean_object* x_136; lean_object* x_137; 
x_136 = lean_ctor_get(x_133, 0);
lean_inc(x_136);
lean_dec(x_133);
x_137 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_137, 0, x_136);
return x_137;
}
}
}
block_174:
{
switch (lean_obj_tag(x_139)) {
case 0:
{
lean_object* x_148; 
x_148 = lp_aesop_Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4(x_3, x_140, x_141, x_142, x_143, x_144, x_145, x_146);
if (lean_obj_tag(x_148) == 0)
{
lean_object* x_149; uint8_t x_150; 
x_149 = lean_ctor_get(x_148, 0);
lean_inc(x_149);
lean_dec_ref(x_148);
x_150 = lean_unbox(x_149);
lean_dec(x_149);
if (x_150 == 0)
{
x_14 = x_139;
x_15 = x_140;
x_16 = x_141;
x_17 = x_142;
x_18 = x_143;
x_19 = x_144;
x_20 = x_145;
x_21 = x_146;
x_22 = lean_box(0);
goto block_124;
}
else
{
lean_object* x_151; lean_object* x_152; lean_object* x_153; uint8_t x_154; 
lean_dec_ref(x_139);
x_151 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_152 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_151, x_145);
x_153 = lean_ctor_get(x_152, 0);
lean_inc(x_153);
lean_dec_ref(x_152);
x_154 = lean_unbox(x_153);
lean_dec(x_153);
if (x_154 == 0)
{
x_125 = x_140;
x_126 = x_141;
x_127 = x_142;
x_128 = x_143;
x_129 = x_144;
x_130 = x_145;
x_131 = x_146;
x_132 = lean_box(0);
goto block_138;
}
else
{
lean_object* x_155; lean_object* x_156; lean_object* x_157; 
x_155 = lean_ctor_get(x_151, 0);
lean_inc(x_155);
x_156 = lp_aesop_Aesop_normSimpCore___lam__0___closed__2;
x_157 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_155, x_156, x_143, x_144, x_145, x_146);
if (lean_obj_tag(x_157) == 0)
{
lean_dec_ref(x_157);
x_125 = x_140;
x_126 = x_141;
x_127 = x_142;
x_128 = x_143;
x_129 = x_144;
x_130 = x_145;
x_131 = x_146;
x_132 = lean_box(0);
goto block_138;
}
else
{
uint8_t x_158; 
lean_dec(x_146);
lean_dec_ref(x_145);
lean_dec(x_144);
lean_dec_ref(x_143);
lean_dec(x_142);
lean_dec_ref(x_140);
lean_dec(x_13);
lean_dec(x_2);
x_158 = !lean_is_exclusive(x_157);
if (x_158 == 0)
{
return x_157;
}
else
{
lean_object* x_159; lean_object* x_160; 
x_159 = lean_ctor_get(x_157, 0);
lean_inc(x_159);
lean_dec(x_157);
x_160 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_160, 0, x_159);
return x_160;
}
}
}
}
}
else
{
uint8_t x_161; 
lean_dec(x_146);
lean_dec_ref(x_145);
lean_dec(x_144);
lean_dec_ref(x_143);
lean_dec(x_142);
lean_dec_ref(x_140);
lean_dec_ref(x_139);
lean_dec(x_13);
lean_dec(x_2);
x_161 = !lean_is_exclusive(x_148);
if (x_161 == 0)
{
return x_148;
}
else
{
lean_object* x_162; lean_object* x_163; 
x_162 = lean_ctor_get(x_148, 0);
lean_inc(x_162);
lean_dec(x_148);
x_163 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_163, 0, x_162);
return x_163;
}
}
}
case 1:
{
lean_object* x_164; lean_object* x_165; lean_object* x_166; uint8_t x_167; 
x_164 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_165 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_164, x_145);
x_166 = lean_ctor_get(x_165, 0);
lean_inc(x_166);
lean_dec_ref(x_165);
x_167 = lean_unbox(x_166);
lean_dec(x_166);
if (x_167 == 0)
{
x_14 = x_139;
x_15 = x_140;
x_16 = x_141;
x_17 = x_142;
x_18 = x_143;
x_19 = x_144;
x_20 = x_145;
x_21 = x_146;
x_22 = lean_box(0);
goto block_124;
}
else
{
lean_object* x_168; lean_object* x_169; lean_object* x_170; 
x_168 = lean_ctor_get(x_164, 0);
lean_inc(x_168);
x_169 = lp_aesop_Aesop_normSimpCore___lam__0___closed__4;
x_170 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_168, x_169, x_143, x_144, x_145, x_146);
if (lean_obj_tag(x_170) == 0)
{
lean_dec_ref(x_170);
x_14 = x_139;
x_15 = x_140;
x_16 = x_141;
x_17 = x_142;
x_18 = x_143;
x_19 = x_144;
x_20 = x_145;
x_21 = x_146;
x_22 = lean_box(0);
goto block_124;
}
else
{
uint8_t x_171; 
lean_dec(x_146);
lean_dec_ref(x_145);
lean_dec(x_144);
lean_dec_ref(x_143);
lean_dec(x_142);
lean_dec_ref(x_140);
lean_dec(x_13);
lean_dec(x_2);
x_171 = !lean_is_exclusive(x_170);
if (x_171 == 0)
{
return x_170;
}
else
{
lean_object* x_172; lean_object* x_173; 
x_172 = lean_ctor_get(x_170, 0);
lean_inc(x_172);
lean_dec(x_170);
x_173 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_173, 0, x_172);
return x_173;
}
}
}
}
default: 
{
x_14 = x_139;
x_15 = x_140;
x_16 = x_141;
x_17 = x_142;
x_18 = x_143;
x_19 = x_144;
x_20 = x_145;
x_21 = x_146;
x_22 = lean_box(0);
goto block_124;
}
}
}
}
else
{
uint8_t x_208; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_208 = !lean_is_exclusive(x_12);
if (x_208 == 0)
{
return x_12;
}
else
{
lean_object* x_209; lean_object* x_210; 
x_209 = lean_ctor_get(x_12, 0);
lean_inc(x_209);
lean_dec(x_12);
x_210 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_210, 0, x_209);
return x_210;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimpCore___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_normSimpCore___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimpCore(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_ctor_get(x_3, 2);
lean_inc(x_1);
lean_inc_ref(x_11);
x_12 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimpCore___lam__0___boxed), 11, 3);
lean_closure_set(x_12, 0, x_11);
lean_closure_set(x_12, 1, x_1);
lean_closure_set(x_12, 2, x_2);
x_13 = lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg(x_1, x_12, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0(x_1, x_2, x_3);
lean_dec(x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; uint8_t x_6; lean_object* x_7; 
x_5 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_6 = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0(x_1, x_2, x_5, x_4);
lean_dec(x_4);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___redArg(x_1, x_2);
lean_dec(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimpCore___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_normSimpCore(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Lean_MVarId_withContext___at___00Aesop_normSimpCore_spec__7___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0_spec__0___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_1);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
size_t x_15; size_t x_16; lean_object* x_17; 
x_15 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_16 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_17 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__5(x_1, x_2, x_3, x_15, x_16, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; uint8_t x_5; lean_object* x_6; 
x_4 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_5 = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg(x_1, x_4, x_3);
lean_dec(x_3);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_HashSet_anyM___at___00Aesop_normSimpCore_spec__4_spec__4(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = l_Lean_instBEqMVarId_beq(x_3, x_1);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_array_push(x_2, x_3);
return x_5;
}
else
{
lean_dec(x_3);
return x_2;
}
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_instBEqMVarId_beq___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lp_aesop_Aesop_checkSimp___lam__1___closed__0;
lean_inc(x_3);
x_5 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_5, 0, x_3);
x_6 = l_Option_instBEq_beq___redArg(x_4, x_5, x_1);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_array_push(x_2, x_3);
return x_7;
}
else
{
lean_dec(x_3);
return x_2;
}
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = l_Lean_instMonadExceptOfExceptionCoreM;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = l_Lean_instMonadExceptOfExceptionCoreM;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_checkSimp___closed__1;
x_2 = lp_aesop_Aesop_checkSimp___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__2;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__2;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_checkSimp___closed__4;
x_2 = lp_aesop_Aesop_checkSimp___closed__3;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__5;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__5;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_checkSimp___closed__7;
x_2 = lp_aesop_Aesop_checkSimp___closed__6;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__8;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__8;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_checkSimp___closed__10;
x_2 = lp_aesop_Aesop_checkSimp___closed__9;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__11;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__11;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_checkSimp___closed__13;
x_2 = lp_aesop_Aesop_checkSimp___closed__12;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_runNormRuleTac___closed__8;
x_2 = lp_aesop_Aesop_Check_name(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__15;
x_2 = l_Lean_MessageData_ofName(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_2 = lp_aesop_Aesop_checkSimp___closed__16;
x_3 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" solved the goal", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__18;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__20() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" assigned mvars:", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__20;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__22() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__23() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__1___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__24() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__2___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__25() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__3), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__26() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__4___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__27() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__5___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__28() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__6), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__29() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_checkSimp___closed__23;
x_2 = lp_aesop_Aesop_checkSimp___closed__22;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__30() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_aesop_Aesop_checkSimp___closed__27;
x_2 = lp_aesop_Aesop_checkSimp___closed__26;
x_3 = lp_aesop_Aesop_checkSimp___closed__25;
x_4 = lp_aesop_Aesop_checkSimp___closed__24;
x_5 = lp_aesop_Aesop_checkSimp___closed__29;
x_6 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_3);
lean_ctor_set(x_6, 3, x_2);
lean_ctor_set(x_6, 4, x_1);
return x_6;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__31() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_checkSimp___closed__28;
x_2 = lp_aesop_Aesop_checkSimp___closed__30;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__32() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_MessageData_ofName), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__33() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" introduced mvars:", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkSimp___closed__34() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkSimp___closed__33;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_checkSimp___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_checkSimp___lam__2(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; uint8_t x_14; 
x_13 = lp_aesop_Aesop_withNormTraceNode___closed__1;
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_15 = lean_ctor_get(x_13, 0);
x_16 = lean_ctor_get(x_13, 1);
lean_dec(x_16);
x_17 = !lean_is_exclusive(x_15);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_18 = lean_ctor_get(x_15, 0);
x_19 = lean_ctor_get(x_15, 2);
x_20 = lean_ctor_get(x_15, 3);
x_21 = lean_ctor_get(x_15, 4);
x_22 = lean_ctor_get(x_15, 1);
lean_dec(x_22);
x_23 = lp_aesop_Aesop_withNormTraceNode___closed__2;
x_24 = lp_aesop_Aesop_withNormTraceNode___closed__3;
lean_inc_ref(x_18);
x_25 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_25, 0, x_18);
x_26 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_26, 0, x_18);
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_25);
lean_ctor_set(x_27, 1, x_26);
x_28 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_28, 0, x_21);
x_29 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_29, 0, x_20);
x_30 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_30, 0, x_19);
lean_ctor_set(x_15, 4, x_28);
lean_ctor_set(x_15, 3, x_29);
lean_ctor_set(x_15, 2, x_30);
lean_ctor_set(x_15, 1, x_23);
lean_ctor_set(x_15, 0, x_27);
lean_ctor_set(x_13, 1, x_24);
x_31 = l_ReaderT_instMonad___redArg(x_13);
x_32 = !lean_is_exclusive(x_31);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; uint8_t x_35; 
x_33 = lean_ctor_get(x_31, 0);
x_34 = lean_ctor_get(x_31, 1);
lean_dec(x_34);
x_35 = !lean_is_exclusive(x_33);
if (x_35 == 0)
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_36 = lean_ctor_get(x_33, 0);
x_37 = lean_ctor_get(x_33, 2);
x_38 = lean_ctor_get(x_33, 3);
x_39 = lean_ctor_get(x_33, 4);
x_40 = lean_ctor_get(x_33, 1);
lean_dec(x_40);
x_41 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_42 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_36);
x_43 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_43, 0, x_36);
x_44 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_44, 0, x_36);
x_45 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_45, 0, x_43);
lean_ctor_set(x_45, 1, x_44);
x_46 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_46, 0, x_39);
x_47 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_47, 0, x_38);
x_48 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_48, 0, x_37);
lean_ctor_set(x_33, 4, x_46);
lean_ctor_set(x_33, 3, x_47);
lean_ctor_set(x_33, 2, x_48);
lean_ctor_set(x_33, 1, x_41);
lean_ctor_set(x_33, 0, x_45);
lean_ctor_set(x_31, 1, x_42);
x_49 = l_ReaderT_instMonad___redArg(x_31);
x_50 = l_ReaderT_instMonad___redArg(x_49);
x_51 = l_ReaderT_instMonad___redArg(x_50);
x_52 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_53 = lean_ctor_get(x_52, 0);
lean_inc(x_53);
x_54 = lp_aesop_Aesop_runNormRuleTac___closed__8;
lean_inc_ref(x_51);
x_55 = lp_aesop_Aesop_Check_isEnabled___redArg(x_51, x_53, x_54);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_56 = lean_apply_8(x_55, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_56) == 0)
{
lean_object* x_57; uint8_t x_58; 
x_57 = lean_ctor_get(x_56, 0);
lean_inc(x_57);
lean_dec_ref(x_56);
x_58 = lean_unbox(x_57);
lean_dec(x_57);
if (x_58 == 0)
{
lean_object* x_59; 
lean_dec_ref(x_51);
lean_dec(x_3);
lean_dec_ref(x_1);
x_59 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
return x_59;
}
else
{
lean_object* x_60; 
x_60 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_60) == 0)
{
lean_object* x_61; lean_object* x_62; 
x_61 = lean_ctor_get(x_60, 0);
lean_inc(x_61);
lean_dec_ref(x_60);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_62 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_62) == 0)
{
lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; lean_object* x_273; 
x_63 = lean_ctor_get(x_62, 0);
lean_inc(x_63);
if (lean_is_exclusive(x_62)) {
 lean_ctor_release(x_62, 0);
 x_64 = x_62;
} else {
 lean_dec_ref(x_62);
 x_64 = lean_box(0);
}
x_65 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__0___boxed), 1, 0);
lean_inc(x_3);
x_66 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__2___boxed), 3, 1);
lean_closure_set(x_66, 0, x_3);
x_67 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_68 = lp_aesop_Aesop_withNormTraceNode___closed__7;
if (lean_obj_tag(x_63) == 0)
{
lean_object* x_295; 
x_295 = lean_box(0);
x_273 = x_295;
goto block_294;
}
else
{
lean_object* x_296; lean_object* x_297; 
x_296 = lean_ctor_get(x_63, 0);
x_297 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_296);
x_273 = x_297;
goto block_294;
}
block_174:
{
if (x_2 == 0)
{
if (lean_obj_tag(x_69) == 0)
{
lean_object* x_78; uint8_t x_79; 
lean_dec(x_64);
x_78 = l_Lean_Meta_instMonadMCtxMetaM;
x_79 = !lean_is_exclusive(x_78);
if (x_79 == 0)
{
lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; 
x_80 = lean_ctor_get(x_78, 0);
x_81 = lean_ctor_get(x_78, 1);
x_82 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_82, 0, x_81);
lean_closure_set(x_82, 1, x_68);
x_83 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_83, 0, lean_box(0));
lean_closure_set(x_83, 1, lean_box(0));
lean_closure_set(x_83, 2, lean_box(0));
lean_closure_set(x_83, 3, lean_box(0));
lean_closure_set(x_83, 4, x_80);
x_84 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_84, 0, x_82);
lean_closure_set(x_84, 1, x_68);
x_85 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_85, 0, lean_box(0));
lean_closure_set(x_85, 1, lean_box(0));
lean_closure_set(x_85, 2, lean_box(0));
lean_closure_set(x_85, 3, lean_box(0));
lean_closure_set(x_85, 4, x_83);
x_86 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_86, 0, x_84);
lean_closure_set(x_86, 1, x_67);
x_87 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_87, 0, lean_box(0));
lean_closure_set(x_87, 1, x_85);
lean_ctor_set(x_78, 1, x_86);
lean_ctor_set(x_78, 0, x_87);
lean_inc_ref(x_51);
x_88 = l_Lean_MVarId_isAssigned___redArg(x_51, x_78, x_3);
lean_inc(x_76);
lean_inc_ref(x_75);
lean_inc(x_74);
lean_inc_ref(x_73);
lean_inc(x_72);
lean_inc(x_71);
lean_inc_ref(x_70);
x_89 = lean_apply_8(x_88, x_70, x_71, x_72, x_73, x_74, x_75, x_76, lean_box(0));
if (lean_obj_tag(x_89) == 0)
{
uint8_t x_90; 
x_90 = !lean_is_exclusive(x_89);
if (x_90 == 0)
{
lean_object* x_91; uint8_t x_92; 
x_91 = lean_ctor_get(x_89, 0);
x_92 = lean_unbox(x_91);
lean_dec(x_91);
if (x_92 == 0)
{
lean_dec(x_76);
lean_dec_ref(x_75);
lean_dec(x_74);
lean_dec_ref(x_73);
lean_dec(x_72);
lean_dec(x_71);
lean_dec_ref(x_70);
lean_dec_ref(x_51);
lean_dec_ref(x_1);
lean_ctor_set(x_89, 0, x_63);
return x_89;
}
else
{
lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; 
lean_free_object(x_89);
x_93 = lp_aesop_Aesop_checkSimp___closed__14;
x_94 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_95 = lean_ctor_get(x_94, 0);
lean_inc_ref(x_95);
x_96 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_51);
x_97 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_96, x_51);
x_98 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_98, 0, x_93);
lean_ctor_set(x_98, 1, x_95);
lean_ctor_set(x_98, 2, x_97);
x_99 = lp_aesop_Aesop_checkSimp___closed__17;
x_100 = l_Lean_stringToMessageData(x_1);
x_101 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_101, 0, x_99);
lean_ctor_set(x_101, 1, x_100);
x_102 = lp_aesop_Aesop_checkSimp___closed__19;
x_103 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_103, 0, x_101);
lean_ctor_set(x_103, 1, x_102);
x_104 = l_Lean_throwError___redArg(x_51, x_98, x_103);
x_105 = lean_apply_8(x_104, x_70, x_71, x_72, x_73, x_74, x_75, x_76, lean_box(0));
if (lean_obj_tag(x_105) == 0)
{
uint8_t x_106; 
x_106 = !lean_is_exclusive(x_105);
if (x_106 == 0)
{
lean_object* x_107; 
x_107 = lean_ctor_get(x_105, 0);
lean_dec(x_107);
lean_ctor_set(x_105, 0, x_63);
return x_105;
}
else
{
lean_object* x_108; 
lean_dec(x_105);
x_108 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_108, 0, x_63);
return x_108;
}
}
else
{
uint8_t x_109; 
lean_dec(x_63);
x_109 = !lean_is_exclusive(x_105);
if (x_109 == 0)
{
return x_105;
}
else
{
lean_object* x_110; lean_object* x_111; 
x_110 = lean_ctor_get(x_105, 0);
lean_inc(x_110);
lean_dec(x_105);
x_111 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_111, 0, x_110);
return x_111;
}
}
}
}
else
{
lean_object* x_112; uint8_t x_113; 
x_112 = lean_ctor_get(x_89, 0);
lean_inc(x_112);
lean_dec(x_89);
x_113 = lean_unbox(x_112);
lean_dec(x_112);
if (x_113 == 0)
{
lean_object* x_114; 
lean_dec(x_76);
lean_dec_ref(x_75);
lean_dec(x_74);
lean_dec_ref(x_73);
lean_dec(x_72);
lean_dec(x_71);
lean_dec_ref(x_70);
lean_dec_ref(x_51);
lean_dec_ref(x_1);
x_114 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_114, 0, x_63);
return x_114;
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; 
x_115 = lp_aesop_Aesop_checkSimp___closed__14;
x_116 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_117 = lean_ctor_get(x_116, 0);
lean_inc_ref(x_117);
x_118 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_51);
x_119 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_118, x_51);
x_120 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_120, 0, x_115);
lean_ctor_set(x_120, 1, x_117);
lean_ctor_set(x_120, 2, x_119);
x_121 = lp_aesop_Aesop_checkSimp___closed__17;
x_122 = l_Lean_stringToMessageData(x_1);
x_123 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_123, 0, x_121);
lean_ctor_set(x_123, 1, x_122);
x_124 = lp_aesop_Aesop_checkSimp___closed__19;
x_125 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_125, 0, x_123);
lean_ctor_set(x_125, 1, x_124);
x_126 = l_Lean_throwError___redArg(x_51, x_120, x_125);
x_127 = lean_apply_8(x_126, x_70, x_71, x_72, x_73, x_74, x_75, x_76, lean_box(0));
if (lean_obj_tag(x_127) == 0)
{
lean_object* x_128; lean_object* x_129; 
if (lean_is_exclusive(x_127)) {
 lean_ctor_release(x_127, 0);
 x_128 = x_127;
} else {
 lean_dec_ref(x_127);
 x_128 = lean_box(0);
}
if (lean_is_scalar(x_128)) {
 x_129 = lean_alloc_ctor(0, 1, 0);
} else {
 x_129 = x_128;
}
lean_ctor_set(x_129, 0, x_63);
return x_129;
}
else
{
lean_object* x_130; lean_object* x_131; lean_object* x_132; 
lean_dec(x_63);
x_130 = lean_ctor_get(x_127, 0);
lean_inc(x_130);
if (lean_is_exclusive(x_127)) {
 lean_ctor_release(x_127, 0);
 x_131 = x_127;
} else {
 lean_dec_ref(x_127);
 x_131 = lean_box(0);
}
if (lean_is_scalar(x_131)) {
 x_132 = lean_alloc_ctor(1, 1, 0);
} else {
 x_132 = x_131;
}
lean_ctor_set(x_132, 0, x_130);
return x_132;
}
}
}
}
else
{
uint8_t x_133; 
lean_dec(x_76);
lean_dec_ref(x_75);
lean_dec(x_74);
lean_dec_ref(x_73);
lean_dec(x_72);
lean_dec(x_71);
lean_dec_ref(x_70);
lean_dec(x_63);
lean_dec_ref(x_51);
lean_dec_ref(x_1);
x_133 = !lean_is_exclusive(x_89);
if (x_133 == 0)
{
return x_89;
}
else
{
lean_object* x_134; lean_object* x_135; 
x_134 = lean_ctor_get(x_89, 0);
lean_inc(x_134);
lean_dec(x_89);
x_135 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_135, 0, x_134);
return x_135;
}
}
}
else
{
lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; 
x_136 = lean_ctor_get(x_78, 0);
x_137 = lean_ctor_get(x_78, 1);
lean_inc(x_137);
lean_inc(x_136);
lean_dec(x_78);
x_138 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_138, 0, x_137);
lean_closure_set(x_138, 1, x_68);
x_139 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_139, 0, lean_box(0));
lean_closure_set(x_139, 1, lean_box(0));
lean_closure_set(x_139, 2, lean_box(0));
lean_closure_set(x_139, 3, lean_box(0));
lean_closure_set(x_139, 4, x_136);
x_140 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_140, 0, x_138);
lean_closure_set(x_140, 1, x_68);
x_141 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_141, 0, lean_box(0));
lean_closure_set(x_141, 1, lean_box(0));
lean_closure_set(x_141, 2, lean_box(0));
lean_closure_set(x_141, 3, lean_box(0));
lean_closure_set(x_141, 4, x_139);
x_142 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_142, 0, x_140);
lean_closure_set(x_142, 1, x_67);
x_143 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_143, 0, lean_box(0));
lean_closure_set(x_143, 1, x_141);
x_144 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_144, 0, x_143);
lean_ctor_set(x_144, 1, x_142);
lean_inc_ref(x_51);
x_145 = l_Lean_MVarId_isAssigned___redArg(x_51, x_144, x_3);
lean_inc(x_76);
lean_inc_ref(x_75);
lean_inc(x_74);
lean_inc_ref(x_73);
lean_inc(x_72);
lean_inc(x_71);
lean_inc_ref(x_70);
x_146 = lean_apply_8(x_145, x_70, x_71, x_72, x_73, x_74, x_75, x_76, lean_box(0));
if (lean_obj_tag(x_146) == 0)
{
lean_object* x_147; lean_object* x_148; uint8_t x_149; 
x_147 = lean_ctor_get(x_146, 0);
lean_inc(x_147);
if (lean_is_exclusive(x_146)) {
 lean_ctor_release(x_146, 0);
 x_148 = x_146;
} else {
 lean_dec_ref(x_146);
 x_148 = lean_box(0);
}
x_149 = lean_unbox(x_147);
lean_dec(x_147);
if (x_149 == 0)
{
lean_object* x_150; 
lean_dec(x_76);
lean_dec_ref(x_75);
lean_dec(x_74);
lean_dec_ref(x_73);
lean_dec(x_72);
lean_dec(x_71);
lean_dec_ref(x_70);
lean_dec_ref(x_51);
lean_dec_ref(x_1);
if (lean_is_scalar(x_148)) {
 x_150 = lean_alloc_ctor(0, 1, 0);
} else {
 x_150 = x_148;
}
lean_ctor_set(x_150, 0, x_63);
return x_150;
}
else
{
lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; 
lean_dec(x_148);
x_151 = lp_aesop_Aesop_checkSimp___closed__14;
x_152 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_153 = lean_ctor_get(x_152, 0);
lean_inc_ref(x_153);
x_154 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_51);
x_155 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_154, x_51);
x_156 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_156, 0, x_151);
lean_ctor_set(x_156, 1, x_153);
lean_ctor_set(x_156, 2, x_155);
x_157 = lp_aesop_Aesop_checkSimp___closed__17;
x_158 = l_Lean_stringToMessageData(x_1);
x_159 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_159, 0, x_157);
lean_ctor_set(x_159, 1, x_158);
x_160 = lp_aesop_Aesop_checkSimp___closed__19;
x_161 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_161, 0, x_159);
lean_ctor_set(x_161, 1, x_160);
x_162 = l_Lean_throwError___redArg(x_51, x_156, x_161);
x_163 = lean_apply_8(x_162, x_70, x_71, x_72, x_73, x_74, x_75, x_76, lean_box(0));
if (lean_obj_tag(x_163) == 0)
{
lean_object* x_164; lean_object* x_165; 
if (lean_is_exclusive(x_163)) {
 lean_ctor_release(x_163, 0);
 x_164 = x_163;
} else {
 lean_dec_ref(x_163);
 x_164 = lean_box(0);
}
if (lean_is_scalar(x_164)) {
 x_165 = lean_alloc_ctor(0, 1, 0);
} else {
 x_165 = x_164;
}
lean_ctor_set(x_165, 0, x_63);
return x_165;
}
else
{
lean_object* x_166; lean_object* x_167; lean_object* x_168; 
lean_dec(x_63);
x_166 = lean_ctor_get(x_163, 0);
lean_inc(x_166);
if (lean_is_exclusive(x_163)) {
 lean_ctor_release(x_163, 0);
 x_167 = x_163;
} else {
 lean_dec_ref(x_163);
 x_167 = lean_box(0);
}
if (lean_is_scalar(x_167)) {
 x_168 = lean_alloc_ctor(1, 1, 0);
} else {
 x_168 = x_167;
}
lean_ctor_set(x_168, 0, x_166);
return x_168;
}
}
}
else
{
lean_object* x_169; lean_object* x_170; lean_object* x_171; 
lean_dec(x_76);
lean_dec_ref(x_75);
lean_dec(x_74);
lean_dec_ref(x_73);
lean_dec(x_72);
lean_dec(x_71);
lean_dec_ref(x_70);
lean_dec(x_63);
lean_dec_ref(x_51);
lean_dec_ref(x_1);
x_169 = lean_ctor_get(x_146, 0);
lean_inc(x_169);
if (lean_is_exclusive(x_146)) {
 lean_ctor_release(x_146, 0);
 x_170 = x_146;
} else {
 lean_dec_ref(x_146);
 x_170 = lean_box(0);
}
if (lean_is_scalar(x_170)) {
 x_171 = lean_alloc_ctor(1, 1, 0);
} else {
 x_171 = x_170;
}
lean_ctor_set(x_171, 0, x_169);
return x_171;
}
}
}
else
{
lean_object* x_172; 
lean_dec(x_76);
lean_dec_ref(x_75);
lean_dec(x_74);
lean_dec_ref(x_73);
lean_dec(x_72);
lean_dec(x_71);
lean_dec_ref(x_70);
lean_dec_ref(x_69);
lean_dec_ref(x_51);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_64)) {
 x_172 = lean_alloc_ctor(0, 1, 0);
} else {
 x_172 = x_64;
}
lean_ctor_set(x_172, 0, x_63);
return x_172;
}
}
else
{
lean_object* x_173; 
lean_dec(x_76);
lean_dec_ref(x_75);
lean_dec(x_74);
lean_dec_ref(x_73);
lean_dec(x_72);
lean_dec(x_71);
lean_dec_ref(x_70);
lean_dec(x_69);
lean_dec_ref(x_51);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_64)) {
 x_173 = lean_alloc_ctor(0, 1, 0);
} else {
 x_173 = x_64;
}
lean_ctor_set(x_173, 0, x_63);
return x_173;
}
}
block_213:
{
uint8_t x_186; 
x_186 = l_Array_isEmpty___redArg(x_185);
lean_dec_ref(x_185);
if (x_186 == 0)
{
lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; size_t x_199; size_t x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; 
x_187 = lp_aesop_Aesop_checkSimp___closed__14;
x_188 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_189 = lean_ctor_get(x_188, 0);
lean_inc_ref(x_189);
x_190 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_51);
x_191 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_190, x_51);
x_192 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_192, 0, x_187);
lean_ctor_set(x_192, 1, x_189);
lean_ctor_set(x_192, 2, x_191);
x_193 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_194 = l_Lean_stringToMessageData(x_1);
x_195 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_195, 0, x_193);
lean_ctor_set(x_195, 1, x_194);
x_196 = lp_aesop_Aesop_checkSimp___closed__21;
x_197 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_197, 0, x_195);
lean_ctor_set(x_197, 1, x_196);
x_198 = lp_aesop_Aesop_checkSimp___closed__31;
x_199 = lean_array_size(x_181);
x_200 = 0;
x_201 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_198, x_65, x_199, x_200, x_181);
x_202 = lean_array_to_list(x_201);
x_203 = lp_aesop_Aesop_checkSimp___closed__32;
x_204 = lean_box(0);
x_205 = l_List_mapTR_loop___redArg(x_203, x_202, x_204);
x_206 = l_Lean_MessageData_ofList(x_205);
x_207 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_207, 0, x_197);
lean_ctor_set(x_207, 1, x_206);
lean_inc_ref(x_51);
x_208 = l_Lean_throwError___redArg(x_51, x_192, x_207);
lean_inc(x_176);
lean_inc_ref(x_177);
lean_inc(x_178);
lean_inc_ref(x_182);
lean_inc(x_179);
lean_inc(x_175);
lean_inc_ref(x_183);
x_209 = lean_apply_8(x_208, x_183, x_175, x_179, x_182, x_178, x_177, x_176, lean_box(0));
if (lean_obj_tag(x_209) == 0)
{
lean_dec_ref(x_209);
x_69 = x_184;
x_70 = x_183;
x_71 = x_175;
x_72 = x_179;
x_73 = x_182;
x_74 = x_178;
x_75 = x_177;
x_76 = x_176;
x_77 = lean_box(0);
goto block_174;
}
else
{
uint8_t x_210; 
lean_dec(x_184);
lean_dec_ref(x_183);
lean_dec_ref(x_182);
lean_dec(x_179);
lean_dec(x_178);
lean_dec_ref(x_177);
lean_dec(x_176);
lean_dec(x_175);
lean_dec(x_64);
lean_dec(x_63);
lean_dec_ref(x_51);
lean_dec(x_3);
lean_dec_ref(x_1);
x_210 = !lean_is_exclusive(x_209);
if (x_210 == 0)
{
return x_209;
}
else
{
lean_object* x_211; lean_object* x_212; 
x_211 = lean_ctor_get(x_209, 0);
lean_inc(x_211);
lean_dec(x_209);
x_212 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_212, 0, x_211);
return x_212;
}
}
}
else
{
lean_dec_ref(x_181);
lean_dec_ref(x_65);
x_69 = x_184;
x_70 = x_183;
x_71 = x_175;
x_72 = x_179;
x_73 = x_182;
x_74 = x_178;
x_75 = x_177;
x_76 = x_176;
x_77 = lean_box(0);
goto block_174;
}
}
block_239:
{
lean_object* x_226; 
lean_inc(x_224);
lean_inc_ref(x_223);
lean_inc(x_222);
lean_inc_ref(x_221);
x_226 = lp_batteries_Lean_Meta_getAssignedExprMVars(x_61, x_215, x_221, x_222, x_223, x_224);
if (lean_obj_tag(x_226) == 0)
{
lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; uint8_t x_231; 
x_227 = lean_ctor_get(x_226, 0);
lean_inc(x_227);
lean_dec_ref(x_226);
x_228 = lean_array_get_size(x_227);
x_229 = lean_mk_empty_array_with_capacity(x_214);
x_230 = lp_aesop_Aesop_checkSimp___closed__31;
x_231 = lean_nat_dec_lt(x_214, x_228);
if (x_231 == 0)
{
lean_dec(x_227);
lean_dec_ref(x_66);
x_175 = x_219;
x_176 = x_224;
x_177 = x_223;
x_178 = x_222;
x_179 = x_220;
x_180 = lean_box(0);
x_181 = x_216;
x_182 = x_221;
x_183 = x_218;
x_184 = x_217;
x_185 = x_229;
goto block_213;
}
else
{
uint8_t x_232; 
x_232 = lean_nat_dec_le(x_228, x_228);
if (x_232 == 0)
{
lean_dec(x_227);
lean_dec_ref(x_66);
x_175 = x_219;
x_176 = x_224;
x_177 = x_223;
x_178 = x_222;
x_179 = x_220;
x_180 = lean_box(0);
x_181 = x_216;
x_182 = x_221;
x_183 = x_218;
x_184 = x_217;
x_185 = x_229;
goto block_213;
}
else
{
size_t x_233; size_t x_234; lean_object* x_235; 
x_233 = 0;
x_234 = lean_usize_of_nat(x_228);
x_235 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_230, x_66, x_227, x_233, x_234, x_229);
x_175 = x_219;
x_176 = x_224;
x_177 = x_223;
x_178 = x_222;
x_179 = x_220;
x_180 = lean_box(0);
x_181 = x_216;
x_182 = x_221;
x_183 = x_218;
x_184 = x_217;
x_185 = x_235;
goto block_213;
}
}
}
else
{
uint8_t x_236; 
lean_dec(x_224);
lean_dec_ref(x_223);
lean_dec(x_222);
lean_dec_ref(x_221);
lean_dec(x_220);
lean_dec(x_219);
lean_dec_ref(x_218);
lean_dec(x_217);
lean_dec_ref(x_216);
lean_dec_ref(x_66);
lean_dec_ref(x_65);
lean_dec(x_64);
lean_dec(x_63);
lean_dec_ref(x_51);
lean_dec(x_3);
lean_dec_ref(x_1);
x_236 = !lean_is_exclusive(x_226);
if (x_236 == 0)
{
return x_226;
}
else
{
lean_object* x_237; lean_object* x_238; 
x_237 = lean_ctor_get(x_226, 0);
lean_inc(x_237);
lean_dec(x_226);
x_238 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_238, 0, x_237);
return x_238;
}
}
}
block_272:
{
uint8_t x_245; 
x_245 = l_Array_isEmpty___redArg(x_244);
if (x_245 == 0)
{
lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; size_t x_258; size_t x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; 
x_246 = lp_aesop_Aesop_checkSimp___closed__14;
x_247 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_248 = lean_ctor_get(x_247, 0);
lean_inc_ref(x_248);
x_249 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_51);
x_250 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_249, x_51);
x_251 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_251, 0, x_246);
lean_ctor_set(x_251, 1, x_248);
lean_ctor_set(x_251, 2, x_250);
x_252 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_253 = l_Lean_stringToMessageData(x_1);
x_254 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_254, 0, x_252);
lean_ctor_set(x_254, 1, x_253);
x_255 = lp_aesop_Aesop_checkSimp___closed__34;
x_256 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_256, 0, x_254);
lean_ctor_set(x_256, 1, x_255);
x_257 = lp_aesop_Aesop_checkSimp___closed__31;
x_258 = lean_array_size(x_244);
x_259 = 0;
lean_inc_ref(x_244);
lean_inc_ref(x_65);
x_260 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_257, x_65, x_258, x_259, x_244);
x_261 = lean_array_to_list(x_260);
x_262 = lp_aesop_Aesop_checkSimp___closed__32;
x_263 = lean_box(0);
x_264 = l_List_mapTR_loop___redArg(x_262, x_261, x_263);
x_265 = l_Lean_MessageData_ofList(x_264);
x_266 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_266, 0, x_256);
lean_ctor_set(x_266, 1, x_265);
lean_inc_ref(x_51);
x_267 = l_Lean_throwError___redArg(x_51, x_251, x_266);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_268 = lean_apply_8(x_267, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_268) == 0)
{
lean_dec_ref(x_268);
x_214 = x_240;
x_215 = x_241;
x_216 = x_244;
x_217 = x_243;
x_218 = x_5;
x_219 = x_6;
x_220 = x_7;
x_221 = x_8;
x_222 = x_9;
x_223 = x_10;
x_224 = x_11;
x_225 = lean_box(0);
goto block_239;
}
else
{
uint8_t x_269; 
lean_dec_ref(x_244);
lean_dec(x_243);
lean_dec_ref(x_241);
lean_dec_ref(x_66);
lean_dec_ref(x_65);
lean_dec(x_64);
lean_dec(x_63);
lean_dec(x_61);
lean_dec_ref(x_51);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_269 = !lean_is_exclusive(x_268);
if (x_269 == 0)
{
return x_268;
}
else
{
lean_object* x_270; lean_object* x_271; 
x_270 = lean_ctor_get(x_268, 0);
lean_inc(x_270);
lean_dec(x_268);
x_271 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_271, 0, x_270);
return x_271;
}
}
}
else
{
x_214 = x_240;
x_215 = x_241;
x_216 = x_244;
x_217 = x_243;
x_218 = x_5;
x_219 = x_6;
x_220 = x_7;
x_221 = x_8;
x_222 = x_9;
x_223 = x_10;
x_224 = x_11;
x_225 = lean_box(0);
goto block_239;
}
}
block_294:
{
lean_object* x_274; 
x_274 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_274) == 0)
{
lean_object* x_275; lean_object* x_276; 
x_275 = lean_ctor_get(x_274, 0);
lean_inc(x_275);
lean_dec_ref(x_274);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_275);
lean_inc(x_61);
x_276 = lp_batteries_Lean_Meta_getIntroducedExprMVars(x_61, x_275, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_276) == 0)
{
lean_object* x_277; lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; uint8_t x_282; 
x_277 = lean_ctor_get(x_276, 0);
lean_inc(x_277);
lean_dec_ref(x_276);
x_278 = lean_unsigned_to_nat(0u);
x_279 = lean_array_get_size(x_277);
x_280 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_281 = lp_aesop_Aesop_checkSimp___closed__31;
x_282 = lean_nat_dec_lt(x_278, x_279);
if (x_282 == 0)
{
lean_dec(x_277);
x_240 = x_278;
x_241 = x_275;
x_242 = lean_box(0);
x_243 = x_273;
x_244 = x_280;
goto block_272;
}
else
{
uint8_t x_283; 
x_283 = lean_nat_dec_le(x_279, x_279);
if (x_283 == 0)
{
lean_dec(x_277);
x_240 = x_278;
x_241 = x_275;
x_242 = lean_box(0);
x_243 = x_273;
x_244 = x_280;
goto block_272;
}
else
{
lean_object* x_284; size_t x_285; size_t x_286; lean_object* x_287; 
lean_inc(x_273);
x_284 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__1), 3, 1);
lean_closure_set(x_284, 0, x_273);
x_285 = 0;
x_286 = lean_usize_of_nat(x_279);
x_287 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_281, x_284, x_277, x_285, x_286, x_280);
x_240 = x_278;
x_241 = x_275;
x_242 = lean_box(0);
x_243 = x_273;
x_244 = x_287;
goto block_272;
}
}
}
else
{
uint8_t x_288; 
lean_dec(x_275);
lean_dec(x_273);
lean_dec_ref(x_66);
lean_dec_ref(x_65);
lean_dec(x_64);
lean_dec(x_63);
lean_dec(x_61);
lean_dec_ref(x_51);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_288 = !lean_is_exclusive(x_276);
if (x_288 == 0)
{
return x_276;
}
else
{
lean_object* x_289; lean_object* x_290; 
x_289 = lean_ctor_get(x_276, 0);
lean_inc(x_289);
lean_dec(x_276);
x_290 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_290, 0, x_289);
return x_290;
}
}
}
else
{
uint8_t x_291; 
lean_dec(x_273);
lean_dec_ref(x_66);
lean_dec_ref(x_65);
lean_dec(x_64);
lean_dec(x_63);
lean_dec(x_61);
lean_dec_ref(x_51);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_291 = !lean_is_exclusive(x_274);
if (x_291 == 0)
{
return x_274;
}
else
{
lean_object* x_292; lean_object* x_293; 
x_292 = lean_ctor_get(x_274, 0);
lean_inc(x_292);
lean_dec(x_274);
x_293 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_293, 0, x_292);
return x_293;
}
}
}
}
else
{
lean_dec(x_61);
lean_dec_ref(x_51);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_62;
}
}
else
{
uint8_t x_298; 
lean_dec_ref(x_51);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_298 = !lean_is_exclusive(x_60);
if (x_298 == 0)
{
return x_60;
}
else
{
lean_object* x_299; lean_object* x_300; 
x_299 = lean_ctor_get(x_60, 0);
lean_inc(x_299);
lean_dec(x_60);
x_300 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_300, 0, x_299);
return x_300;
}
}
}
}
else
{
uint8_t x_301; 
lean_dec_ref(x_51);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_301 = !lean_is_exclusive(x_56);
if (x_301 == 0)
{
return x_56;
}
else
{
lean_object* x_302; lean_object* x_303; 
x_302 = lean_ctor_get(x_56, 0);
lean_inc(x_302);
lean_dec(x_56);
x_303 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_303, 0, x_302);
return x_303;
}
}
}
else
{
lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_323; lean_object* x_324; 
x_304 = lean_ctor_get(x_33, 0);
x_305 = lean_ctor_get(x_33, 2);
x_306 = lean_ctor_get(x_33, 3);
x_307 = lean_ctor_get(x_33, 4);
lean_inc(x_307);
lean_inc(x_306);
lean_inc(x_305);
lean_inc(x_304);
lean_dec(x_33);
x_308 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_309 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_304);
x_310 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_310, 0, x_304);
x_311 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_311, 0, x_304);
x_312 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_312, 0, x_310);
lean_ctor_set(x_312, 1, x_311);
x_313 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_313, 0, x_307);
x_314 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_314, 0, x_306);
x_315 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_315, 0, x_305);
x_316 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_316, 0, x_312);
lean_ctor_set(x_316, 1, x_308);
lean_ctor_set(x_316, 2, x_315);
lean_ctor_set(x_316, 3, x_314);
lean_ctor_set(x_316, 4, x_313);
lean_ctor_set(x_31, 1, x_309);
lean_ctor_set(x_31, 0, x_316);
x_317 = l_ReaderT_instMonad___redArg(x_31);
x_318 = l_ReaderT_instMonad___redArg(x_317);
x_319 = l_ReaderT_instMonad___redArg(x_318);
x_320 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_321 = lean_ctor_get(x_320, 0);
lean_inc(x_321);
x_322 = lp_aesop_Aesop_runNormRuleTac___closed__8;
lean_inc_ref(x_319);
x_323 = lp_aesop_Aesop_Check_isEnabled___redArg(x_319, x_321, x_322);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_324 = lean_apply_8(x_323, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_324) == 0)
{
lean_object* x_325; uint8_t x_326; 
x_325 = lean_ctor_get(x_324, 0);
lean_inc(x_325);
lean_dec_ref(x_324);
x_326 = lean_unbox(x_325);
lean_dec(x_325);
if (x_326 == 0)
{
lean_object* x_327; 
lean_dec_ref(x_319);
lean_dec(x_3);
lean_dec_ref(x_1);
x_327 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
return x_327;
}
else
{
lean_object* x_328; 
x_328 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_328) == 0)
{
lean_object* x_329; lean_object* x_330; 
x_329 = lean_ctor_get(x_328, 0);
lean_inc(x_329);
lean_dec_ref(x_328);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_330 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_330) == 0)
{
lean_object* x_331; lean_object* x_332; lean_object* x_333; lean_object* x_334; lean_object* x_335; lean_object* x_336; lean_object* x_337; lean_object* x_338; lean_object* x_339; lean_object* x_340; lean_object* x_341; lean_object* x_342; lean_object* x_343; lean_object* x_344; lean_object* x_345; lean_object* x_387; lean_object* x_388; lean_object* x_389; lean_object* x_390; lean_object* x_391; lean_object* x_392; lean_object* x_393; lean_object* x_394; lean_object* x_395; lean_object* x_396; lean_object* x_397; lean_object* x_426; lean_object* x_427; lean_object* x_428; lean_object* x_429; lean_object* x_430; lean_object* x_431; lean_object* x_432; lean_object* x_433; lean_object* x_434; lean_object* x_435; lean_object* x_436; lean_object* x_437; lean_object* x_452; lean_object* x_453; lean_object* x_454; lean_object* x_455; lean_object* x_456; lean_object* x_485; 
x_331 = lean_ctor_get(x_330, 0);
lean_inc(x_331);
if (lean_is_exclusive(x_330)) {
 lean_ctor_release(x_330, 0);
 x_332 = x_330;
} else {
 lean_dec_ref(x_330);
 x_332 = lean_box(0);
}
x_333 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__0___boxed), 1, 0);
lean_inc(x_3);
x_334 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__2___boxed), 3, 1);
lean_closure_set(x_334, 0, x_3);
x_335 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_336 = lp_aesop_Aesop_withNormTraceNode___closed__7;
if (lean_obj_tag(x_331) == 0)
{
lean_object* x_507; 
x_507 = lean_box(0);
x_485 = x_507;
goto block_506;
}
else
{
lean_object* x_508; lean_object* x_509; 
x_508 = lean_ctor_get(x_331, 0);
x_509 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_508);
x_485 = x_509;
goto block_506;
}
block_386:
{
if (x_2 == 0)
{
if (lean_obj_tag(x_337) == 0)
{
lean_object* x_346; lean_object* x_347; lean_object* x_348; lean_object* x_349; lean_object* x_350; lean_object* x_351; lean_object* x_352; lean_object* x_353; lean_object* x_354; lean_object* x_355; lean_object* x_356; lean_object* x_357; lean_object* x_358; 
lean_dec(x_332);
x_346 = l_Lean_Meta_instMonadMCtxMetaM;
x_347 = lean_ctor_get(x_346, 0);
lean_inc(x_347);
x_348 = lean_ctor_get(x_346, 1);
lean_inc(x_348);
if (lean_is_exclusive(x_346)) {
 lean_ctor_release(x_346, 0);
 lean_ctor_release(x_346, 1);
 x_349 = x_346;
} else {
 lean_dec_ref(x_346);
 x_349 = lean_box(0);
}
x_350 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_350, 0, x_348);
lean_closure_set(x_350, 1, x_336);
x_351 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_351, 0, lean_box(0));
lean_closure_set(x_351, 1, lean_box(0));
lean_closure_set(x_351, 2, lean_box(0));
lean_closure_set(x_351, 3, lean_box(0));
lean_closure_set(x_351, 4, x_347);
x_352 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_352, 0, x_350);
lean_closure_set(x_352, 1, x_336);
x_353 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_353, 0, lean_box(0));
lean_closure_set(x_353, 1, lean_box(0));
lean_closure_set(x_353, 2, lean_box(0));
lean_closure_set(x_353, 3, lean_box(0));
lean_closure_set(x_353, 4, x_351);
x_354 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_354, 0, x_352);
lean_closure_set(x_354, 1, x_335);
x_355 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_355, 0, lean_box(0));
lean_closure_set(x_355, 1, x_353);
if (lean_is_scalar(x_349)) {
 x_356 = lean_alloc_ctor(0, 2, 0);
} else {
 x_356 = x_349;
}
lean_ctor_set(x_356, 0, x_355);
lean_ctor_set(x_356, 1, x_354);
lean_inc_ref(x_319);
x_357 = l_Lean_MVarId_isAssigned___redArg(x_319, x_356, x_3);
lean_inc(x_344);
lean_inc_ref(x_343);
lean_inc(x_342);
lean_inc_ref(x_341);
lean_inc(x_340);
lean_inc(x_339);
lean_inc_ref(x_338);
x_358 = lean_apply_8(x_357, x_338, x_339, x_340, x_341, x_342, x_343, x_344, lean_box(0));
if (lean_obj_tag(x_358) == 0)
{
lean_object* x_359; lean_object* x_360; uint8_t x_361; 
x_359 = lean_ctor_get(x_358, 0);
lean_inc(x_359);
if (lean_is_exclusive(x_358)) {
 lean_ctor_release(x_358, 0);
 x_360 = x_358;
} else {
 lean_dec_ref(x_358);
 x_360 = lean_box(0);
}
x_361 = lean_unbox(x_359);
lean_dec(x_359);
if (x_361 == 0)
{
lean_object* x_362; 
lean_dec(x_344);
lean_dec_ref(x_343);
lean_dec(x_342);
lean_dec_ref(x_341);
lean_dec(x_340);
lean_dec(x_339);
lean_dec_ref(x_338);
lean_dec_ref(x_319);
lean_dec_ref(x_1);
if (lean_is_scalar(x_360)) {
 x_362 = lean_alloc_ctor(0, 1, 0);
} else {
 x_362 = x_360;
}
lean_ctor_set(x_362, 0, x_331);
return x_362;
}
else
{
lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; lean_object* x_367; lean_object* x_368; lean_object* x_369; lean_object* x_370; lean_object* x_371; lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; 
lean_dec(x_360);
x_363 = lp_aesop_Aesop_checkSimp___closed__14;
x_364 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_365 = lean_ctor_get(x_364, 0);
lean_inc_ref(x_365);
x_366 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_319);
x_367 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_366, x_319);
x_368 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_368, 0, x_363);
lean_ctor_set(x_368, 1, x_365);
lean_ctor_set(x_368, 2, x_367);
x_369 = lp_aesop_Aesop_checkSimp___closed__17;
x_370 = l_Lean_stringToMessageData(x_1);
x_371 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_371, 0, x_369);
lean_ctor_set(x_371, 1, x_370);
x_372 = lp_aesop_Aesop_checkSimp___closed__19;
x_373 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_373, 0, x_371);
lean_ctor_set(x_373, 1, x_372);
x_374 = l_Lean_throwError___redArg(x_319, x_368, x_373);
x_375 = lean_apply_8(x_374, x_338, x_339, x_340, x_341, x_342, x_343, x_344, lean_box(0));
if (lean_obj_tag(x_375) == 0)
{
lean_object* x_376; lean_object* x_377; 
if (lean_is_exclusive(x_375)) {
 lean_ctor_release(x_375, 0);
 x_376 = x_375;
} else {
 lean_dec_ref(x_375);
 x_376 = lean_box(0);
}
if (lean_is_scalar(x_376)) {
 x_377 = lean_alloc_ctor(0, 1, 0);
} else {
 x_377 = x_376;
}
lean_ctor_set(x_377, 0, x_331);
return x_377;
}
else
{
lean_object* x_378; lean_object* x_379; lean_object* x_380; 
lean_dec(x_331);
x_378 = lean_ctor_get(x_375, 0);
lean_inc(x_378);
if (lean_is_exclusive(x_375)) {
 lean_ctor_release(x_375, 0);
 x_379 = x_375;
} else {
 lean_dec_ref(x_375);
 x_379 = lean_box(0);
}
if (lean_is_scalar(x_379)) {
 x_380 = lean_alloc_ctor(1, 1, 0);
} else {
 x_380 = x_379;
}
lean_ctor_set(x_380, 0, x_378);
return x_380;
}
}
}
else
{
lean_object* x_381; lean_object* x_382; lean_object* x_383; 
lean_dec(x_344);
lean_dec_ref(x_343);
lean_dec(x_342);
lean_dec_ref(x_341);
lean_dec(x_340);
lean_dec(x_339);
lean_dec_ref(x_338);
lean_dec(x_331);
lean_dec_ref(x_319);
lean_dec_ref(x_1);
x_381 = lean_ctor_get(x_358, 0);
lean_inc(x_381);
if (lean_is_exclusive(x_358)) {
 lean_ctor_release(x_358, 0);
 x_382 = x_358;
} else {
 lean_dec_ref(x_358);
 x_382 = lean_box(0);
}
if (lean_is_scalar(x_382)) {
 x_383 = lean_alloc_ctor(1, 1, 0);
} else {
 x_383 = x_382;
}
lean_ctor_set(x_383, 0, x_381);
return x_383;
}
}
else
{
lean_object* x_384; 
lean_dec(x_344);
lean_dec_ref(x_343);
lean_dec(x_342);
lean_dec_ref(x_341);
lean_dec(x_340);
lean_dec(x_339);
lean_dec_ref(x_338);
lean_dec_ref(x_337);
lean_dec_ref(x_319);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_332)) {
 x_384 = lean_alloc_ctor(0, 1, 0);
} else {
 x_384 = x_332;
}
lean_ctor_set(x_384, 0, x_331);
return x_384;
}
}
else
{
lean_object* x_385; 
lean_dec(x_344);
lean_dec_ref(x_343);
lean_dec(x_342);
lean_dec_ref(x_341);
lean_dec(x_340);
lean_dec(x_339);
lean_dec_ref(x_338);
lean_dec(x_337);
lean_dec_ref(x_319);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_332)) {
 x_385 = lean_alloc_ctor(0, 1, 0);
} else {
 x_385 = x_332;
}
lean_ctor_set(x_385, 0, x_331);
return x_385;
}
}
block_425:
{
uint8_t x_398; 
x_398 = l_Array_isEmpty___redArg(x_397);
lean_dec_ref(x_397);
if (x_398 == 0)
{
lean_object* x_399; lean_object* x_400; lean_object* x_401; lean_object* x_402; lean_object* x_403; lean_object* x_404; lean_object* x_405; lean_object* x_406; lean_object* x_407; lean_object* x_408; lean_object* x_409; lean_object* x_410; size_t x_411; size_t x_412; lean_object* x_413; lean_object* x_414; lean_object* x_415; lean_object* x_416; lean_object* x_417; lean_object* x_418; lean_object* x_419; lean_object* x_420; lean_object* x_421; 
x_399 = lp_aesop_Aesop_checkSimp___closed__14;
x_400 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_401 = lean_ctor_get(x_400, 0);
lean_inc_ref(x_401);
x_402 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_319);
x_403 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_402, x_319);
x_404 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_404, 0, x_399);
lean_ctor_set(x_404, 1, x_401);
lean_ctor_set(x_404, 2, x_403);
x_405 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_406 = l_Lean_stringToMessageData(x_1);
x_407 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_407, 0, x_405);
lean_ctor_set(x_407, 1, x_406);
x_408 = lp_aesop_Aesop_checkSimp___closed__21;
x_409 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_409, 0, x_407);
lean_ctor_set(x_409, 1, x_408);
x_410 = lp_aesop_Aesop_checkSimp___closed__31;
x_411 = lean_array_size(x_393);
x_412 = 0;
x_413 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_410, x_333, x_411, x_412, x_393);
x_414 = lean_array_to_list(x_413);
x_415 = lp_aesop_Aesop_checkSimp___closed__32;
x_416 = lean_box(0);
x_417 = l_List_mapTR_loop___redArg(x_415, x_414, x_416);
x_418 = l_Lean_MessageData_ofList(x_417);
x_419 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_419, 0, x_409);
lean_ctor_set(x_419, 1, x_418);
lean_inc_ref(x_319);
x_420 = l_Lean_throwError___redArg(x_319, x_404, x_419);
lean_inc(x_388);
lean_inc_ref(x_389);
lean_inc(x_390);
lean_inc_ref(x_394);
lean_inc(x_391);
lean_inc(x_387);
lean_inc_ref(x_395);
x_421 = lean_apply_8(x_420, x_395, x_387, x_391, x_394, x_390, x_389, x_388, lean_box(0));
if (lean_obj_tag(x_421) == 0)
{
lean_dec_ref(x_421);
x_337 = x_396;
x_338 = x_395;
x_339 = x_387;
x_340 = x_391;
x_341 = x_394;
x_342 = x_390;
x_343 = x_389;
x_344 = x_388;
x_345 = lean_box(0);
goto block_386;
}
else
{
lean_object* x_422; lean_object* x_423; lean_object* x_424; 
lean_dec(x_396);
lean_dec_ref(x_395);
lean_dec_ref(x_394);
lean_dec(x_391);
lean_dec(x_390);
lean_dec_ref(x_389);
lean_dec(x_388);
lean_dec(x_387);
lean_dec(x_332);
lean_dec(x_331);
lean_dec_ref(x_319);
lean_dec(x_3);
lean_dec_ref(x_1);
x_422 = lean_ctor_get(x_421, 0);
lean_inc(x_422);
if (lean_is_exclusive(x_421)) {
 lean_ctor_release(x_421, 0);
 x_423 = x_421;
} else {
 lean_dec_ref(x_421);
 x_423 = lean_box(0);
}
if (lean_is_scalar(x_423)) {
 x_424 = lean_alloc_ctor(1, 1, 0);
} else {
 x_424 = x_423;
}
lean_ctor_set(x_424, 0, x_422);
return x_424;
}
}
else
{
lean_dec_ref(x_393);
lean_dec_ref(x_333);
x_337 = x_396;
x_338 = x_395;
x_339 = x_387;
x_340 = x_391;
x_341 = x_394;
x_342 = x_390;
x_343 = x_389;
x_344 = x_388;
x_345 = lean_box(0);
goto block_386;
}
}
block_451:
{
lean_object* x_438; 
lean_inc(x_436);
lean_inc_ref(x_435);
lean_inc(x_434);
lean_inc_ref(x_433);
x_438 = lp_batteries_Lean_Meta_getAssignedExprMVars(x_329, x_427, x_433, x_434, x_435, x_436);
if (lean_obj_tag(x_438) == 0)
{
lean_object* x_439; lean_object* x_440; lean_object* x_441; lean_object* x_442; uint8_t x_443; 
x_439 = lean_ctor_get(x_438, 0);
lean_inc(x_439);
lean_dec_ref(x_438);
x_440 = lean_array_get_size(x_439);
x_441 = lean_mk_empty_array_with_capacity(x_426);
x_442 = lp_aesop_Aesop_checkSimp___closed__31;
x_443 = lean_nat_dec_lt(x_426, x_440);
if (x_443 == 0)
{
lean_dec(x_439);
lean_dec_ref(x_334);
x_387 = x_431;
x_388 = x_436;
x_389 = x_435;
x_390 = x_434;
x_391 = x_432;
x_392 = lean_box(0);
x_393 = x_428;
x_394 = x_433;
x_395 = x_430;
x_396 = x_429;
x_397 = x_441;
goto block_425;
}
else
{
uint8_t x_444; 
x_444 = lean_nat_dec_le(x_440, x_440);
if (x_444 == 0)
{
lean_dec(x_439);
lean_dec_ref(x_334);
x_387 = x_431;
x_388 = x_436;
x_389 = x_435;
x_390 = x_434;
x_391 = x_432;
x_392 = lean_box(0);
x_393 = x_428;
x_394 = x_433;
x_395 = x_430;
x_396 = x_429;
x_397 = x_441;
goto block_425;
}
else
{
size_t x_445; size_t x_446; lean_object* x_447; 
x_445 = 0;
x_446 = lean_usize_of_nat(x_440);
x_447 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_442, x_334, x_439, x_445, x_446, x_441);
x_387 = x_431;
x_388 = x_436;
x_389 = x_435;
x_390 = x_434;
x_391 = x_432;
x_392 = lean_box(0);
x_393 = x_428;
x_394 = x_433;
x_395 = x_430;
x_396 = x_429;
x_397 = x_447;
goto block_425;
}
}
}
else
{
lean_object* x_448; lean_object* x_449; lean_object* x_450; 
lean_dec(x_436);
lean_dec_ref(x_435);
lean_dec(x_434);
lean_dec_ref(x_433);
lean_dec(x_432);
lean_dec(x_431);
lean_dec_ref(x_430);
lean_dec(x_429);
lean_dec_ref(x_428);
lean_dec_ref(x_334);
lean_dec_ref(x_333);
lean_dec(x_332);
lean_dec(x_331);
lean_dec_ref(x_319);
lean_dec(x_3);
lean_dec_ref(x_1);
x_448 = lean_ctor_get(x_438, 0);
lean_inc(x_448);
if (lean_is_exclusive(x_438)) {
 lean_ctor_release(x_438, 0);
 x_449 = x_438;
} else {
 lean_dec_ref(x_438);
 x_449 = lean_box(0);
}
if (lean_is_scalar(x_449)) {
 x_450 = lean_alloc_ctor(1, 1, 0);
} else {
 x_450 = x_449;
}
lean_ctor_set(x_450, 0, x_448);
return x_450;
}
}
block_484:
{
uint8_t x_457; 
x_457 = l_Array_isEmpty___redArg(x_456);
if (x_457 == 0)
{
lean_object* x_458; lean_object* x_459; lean_object* x_460; lean_object* x_461; lean_object* x_462; lean_object* x_463; lean_object* x_464; lean_object* x_465; lean_object* x_466; lean_object* x_467; lean_object* x_468; lean_object* x_469; size_t x_470; size_t x_471; lean_object* x_472; lean_object* x_473; lean_object* x_474; lean_object* x_475; lean_object* x_476; lean_object* x_477; lean_object* x_478; lean_object* x_479; lean_object* x_480; 
x_458 = lp_aesop_Aesop_checkSimp___closed__14;
x_459 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_460 = lean_ctor_get(x_459, 0);
lean_inc_ref(x_460);
x_461 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_319);
x_462 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_461, x_319);
x_463 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_463, 0, x_458);
lean_ctor_set(x_463, 1, x_460);
lean_ctor_set(x_463, 2, x_462);
x_464 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_465 = l_Lean_stringToMessageData(x_1);
x_466 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_466, 0, x_464);
lean_ctor_set(x_466, 1, x_465);
x_467 = lp_aesop_Aesop_checkSimp___closed__34;
x_468 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_468, 0, x_466);
lean_ctor_set(x_468, 1, x_467);
x_469 = lp_aesop_Aesop_checkSimp___closed__31;
x_470 = lean_array_size(x_456);
x_471 = 0;
lean_inc_ref(x_456);
lean_inc_ref(x_333);
x_472 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_469, x_333, x_470, x_471, x_456);
x_473 = lean_array_to_list(x_472);
x_474 = lp_aesop_Aesop_checkSimp___closed__32;
x_475 = lean_box(0);
x_476 = l_List_mapTR_loop___redArg(x_474, x_473, x_475);
x_477 = l_Lean_MessageData_ofList(x_476);
x_478 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_478, 0, x_468);
lean_ctor_set(x_478, 1, x_477);
lean_inc_ref(x_319);
x_479 = l_Lean_throwError___redArg(x_319, x_463, x_478);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_480 = lean_apply_8(x_479, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_480) == 0)
{
lean_dec_ref(x_480);
x_426 = x_452;
x_427 = x_453;
x_428 = x_456;
x_429 = x_455;
x_430 = x_5;
x_431 = x_6;
x_432 = x_7;
x_433 = x_8;
x_434 = x_9;
x_435 = x_10;
x_436 = x_11;
x_437 = lean_box(0);
goto block_451;
}
else
{
lean_object* x_481; lean_object* x_482; lean_object* x_483; 
lean_dec_ref(x_456);
lean_dec(x_455);
lean_dec_ref(x_453);
lean_dec_ref(x_334);
lean_dec_ref(x_333);
lean_dec(x_332);
lean_dec(x_331);
lean_dec(x_329);
lean_dec_ref(x_319);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_481 = lean_ctor_get(x_480, 0);
lean_inc(x_481);
if (lean_is_exclusive(x_480)) {
 lean_ctor_release(x_480, 0);
 x_482 = x_480;
} else {
 lean_dec_ref(x_480);
 x_482 = lean_box(0);
}
if (lean_is_scalar(x_482)) {
 x_483 = lean_alloc_ctor(1, 1, 0);
} else {
 x_483 = x_482;
}
lean_ctor_set(x_483, 0, x_481);
return x_483;
}
}
else
{
x_426 = x_452;
x_427 = x_453;
x_428 = x_456;
x_429 = x_455;
x_430 = x_5;
x_431 = x_6;
x_432 = x_7;
x_433 = x_8;
x_434 = x_9;
x_435 = x_10;
x_436 = x_11;
x_437 = lean_box(0);
goto block_451;
}
}
block_506:
{
lean_object* x_486; 
x_486 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_486) == 0)
{
lean_object* x_487; lean_object* x_488; 
x_487 = lean_ctor_get(x_486, 0);
lean_inc(x_487);
lean_dec_ref(x_486);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_487);
lean_inc(x_329);
x_488 = lp_batteries_Lean_Meta_getIntroducedExprMVars(x_329, x_487, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_488) == 0)
{
lean_object* x_489; lean_object* x_490; lean_object* x_491; lean_object* x_492; lean_object* x_493; uint8_t x_494; 
x_489 = lean_ctor_get(x_488, 0);
lean_inc(x_489);
lean_dec_ref(x_488);
x_490 = lean_unsigned_to_nat(0u);
x_491 = lean_array_get_size(x_489);
x_492 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_493 = lp_aesop_Aesop_checkSimp___closed__31;
x_494 = lean_nat_dec_lt(x_490, x_491);
if (x_494 == 0)
{
lean_dec(x_489);
x_452 = x_490;
x_453 = x_487;
x_454 = lean_box(0);
x_455 = x_485;
x_456 = x_492;
goto block_484;
}
else
{
uint8_t x_495; 
x_495 = lean_nat_dec_le(x_491, x_491);
if (x_495 == 0)
{
lean_dec(x_489);
x_452 = x_490;
x_453 = x_487;
x_454 = lean_box(0);
x_455 = x_485;
x_456 = x_492;
goto block_484;
}
else
{
lean_object* x_496; size_t x_497; size_t x_498; lean_object* x_499; 
lean_inc(x_485);
x_496 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__1), 3, 1);
lean_closure_set(x_496, 0, x_485);
x_497 = 0;
x_498 = lean_usize_of_nat(x_491);
x_499 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_493, x_496, x_489, x_497, x_498, x_492);
x_452 = x_490;
x_453 = x_487;
x_454 = lean_box(0);
x_455 = x_485;
x_456 = x_499;
goto block_484;
}
}
}
else
{
lean_object* x_500; lean_object* x_501; lean_object* x_502; 
lean_dec(x_487);
lean_dec(x_485);
lean_dec_ref(x_334);
lean_dec_ref(x_333);
lean_dec(x_332);
lean_dec(x_331);
lean_dec(x_329);
lean_dec_ref(x_319);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_500 = lean_ctor_get(x_488, 0);
lean_inc(x_500);
if (lean_is_exclusive(x_488)) {
 lean_ctor_release(x_488, 0);
 x_501 = x_488;
} else {
 lean_dec_ref(x_488);
 x_501 = lean_box(0);
}
if (lean_is_scalar(x_501)) {
 x_502 = lean_alloc_ctor(1, 1, 0);
} else {
 x_502 = x_501;
}
lean_ctor_set(x_502, 0, x_500);
return x_502;
}
}
else
{
lean_object* x_503; lean_object* x_504; lean_object* x_505; 
lean_dec(x_485);
lean_dec_ref(x_334);
lean_dec_ref(x_333);
lean_dec(x_332);
lean_dec(x_331);
lean_dec(x_329);
lean_dec_ref(x_319);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_503 = lean_ctor_get(x_486, 0);
lean_inc(x_503);
if (lean_is_exclusive(x_486)) {
 lean_ctor_release(x_486, 0);
 x_504 = x_486;
} else {
 lean_dec_ref(x_486);
 x_504 = lean_box(0);
}
if (lean_is_scalar(x_504)) {
 x_505 = lean_alloc_ctor(1, 1, 0);
} else {
 x_505 = x_504;
}
lean_ctor_set(x_505, 0, x_503);
return x_505;
}
}
}
else
{
lean_dec(x_329);
lean_dec_ref(x_319);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_330;
}
}
else
{
lean_object* x_510; lean_object* x_511; lean_object* x_512; 
lean_dec_ref(x_319);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_510 = lean_ctor_get(x_328, 0);
lean_inc(x_510);
if (lean_is_exclusive(x_328)) {
 lean_ctor_release(x_328, 0);
 x_511 = x_328;
} else {
 lean_dec_ref(x_328);
 x_511 = lean_box(0);
}
if (lean_is_scalar(x_511)) {
 x_512 = lean_alloc_ctor(1, 1, 0);
} else {
 x_512 = x_511;
}
lean_ctor_set(x_512, 0, x_510);
return x_512;
}
}
}
else
{
lean_object* x_513; lean_object* x_514; lean_object* x_515; 
lean_dec_ref(x_319);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_513 = lean_ctor_get(x_324, 0);
lean_inc(x_513);
if (lean_is_exclusive(x_324)) {
 lean_ctor_release(x_324, 0);
 x_514 = x_324;
} else {
 lean_dec_ref(x_324);
 x_514 = lean_box(0);
}
if (lean_is_scalar(x_514)) {
 x_515 = lean_alloc_ctor(1, 1, 0);
} else {
 x_515 = x_514;
}
lean_ctor_set(x_515, 0, x_513);
return x_515;
}
}
}
else
{
lean_object* x_516; lean_object* x_517; lean_object* x_518; lean_object* x_519; lean_object* x_520; lean_object* x_521; lean_object* x_522; lean_object* x_523; lean_object* x_524; lean_object* x_525; lean_object* x_526; lean_object* x_527; lean_object* x_528; lean_object* x_529; lean_object* x_530; lean_object* x_531; lean_object* x_532; lean_object* x_533; lean_object* x_534; lean_object* x_535; lean_object* x_536; lean_object* x_537; lean_object* x_538; lean_object* x_539; 
x_516 = lean_ctor_get(x_31, 0);
lean_inc(x_516);
lean_dec(x_31);
x_517 = lean_ctor_get(x_516, 0);
lean_inc_ref(x_517);
x_518 = lean_ctor_get(x_516, 2);
lean_inc(x_518);
x_519 = lean_ctor_get(x_516, 3);
lean_inc(x_519);
x_520 = lean_ctor_get(x_516, 4);
lean_inc(x_520);
if (lean_is_exclusive(x_516)) {
 lean_ctor_release(x_516, 0);
 lean_ctor_release(x_516, 1);
 lean_ctor_release(x_516, 2);
 lean_ctor_release(x_516, 3);
 lean_ctor_release(x_516, 4);
 x_521 = x_516;
} else {
 lean_dec_ref(x_516);
 x_521 = lean_box(0);
}
x_522 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_523 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_517);
x_524 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_524, 0, x_517);
x_525 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_525, 0, x_517);
x_526 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_526, 0, x_524);
lean_ctor_set(x_526, 1, x_525);
x_527 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_527, 0, x_520);
x_528 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_528, 0, x_519);
x_529 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_529, 0, x_518);
if (lean_is_scalar(x_521)) {
 x_530 = lean_alloc_ctor(0, 5, 0);
} else {
 x_530 = x_521;
}
lean_ctor_set(x_530, 0, x_526);
lean_ctor_set(x_530, 1, x_522);
lean_ctor_set(x_530, 2, x_529);
lean_ctor_set(x_530, 3, x_528);
lean_ctor_set(x_530, 4, x_527);
x_531 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_531, 0, x_530);
lean_ctor_set(x_531, 1, x_523);
x_532 = l_ReaderT_instMonad___redArg(x_531);
x_533 = l_ReaderT_instMonad___redArg(x_532);
x_534 = l_ReaderT_instMonad___redArg(x_533);
x_535 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_536 = lean_ctor_get(x_535, 0);
lean_inc(x_536);
x_537 = lp_aesop_Aesop_runNormRuleTac___closed__8;
lean_inc_ref(x_534);
x_538 = lp_aesop_Aesop_Check_isEnabled___redArg(x_534, x_536, x_537);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_539 = lean_apply_8(x_538, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_539) == 0)
{
lean_object* x_540; uint8_t x_541; 
x_540 = lean_ctor_get(x_539, 0);
lean_inc(x_540);
lean_dec_ref(x_539);
x_541 = lean_unbox(x_540);
lean_dec(x_540);
if (x_541 == 0)
{
lean_object* x_542; 
lean_dec_ref(x_534);
lean_dec(x_3);
lean_dec_ref(x_1);
x_542 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
return x_542;
}
else
{
lean_object* x_543; 
x_543 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_543) == 0)
{
lean_object* x_544; lean_object* x_545; 
x_544 = lean_ctor_get(x_543, 0);
lean_inc(x_544);
lean_dec_ref(x_543);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_545 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_545) == 0)
{
lean_object* x_546; lean_object* x_547; lean_object* x_548; lean_object* x_549; lean_object* x_550; lean_object* x_551; lean_object* x_552; lean_object* x_553; lean_object* x_554; lean_object* x_555; lean_object* x_556; lean_object* x_557; lean_object* x_558; lean_object* x_559; lean_object* x_560; lean_object* x_602; lean_object* x_603; lean_object* x_604; lean_object* x_605; lean_object* x_606; lean_object* x_607; lean_object* x_608; lean_object* x_609; lean_object* x_610; lean_object* x_611; lean_object* x_612; lean_object* x_641; lean_object* x_642; lean_object* x_643; lean_object* x_644; lean_object* x_645; lean_object* x_646; lean_object* x_647; lean_object* x_648; lean_object* x_649; lean_object* x_650; lean_object* x_651; lean_object* x_652; lean_object* x_667; lean_object* x_668; lean_object* x_669; lean_object* x_670; lean_object* x_671; lean_object* x_700; 
x_546 = lean_ctor_get(x_545, 0);
lean_inc(x_546);
if (lean_is_exclusive(x_545)) {
 lean_ctor_release(x_545, 0);
 x_547 = x_545;
} else {
 lean_dec_ref(x_545);
 x_547 = lean_box(0);
}
x_548 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__0___boxed), 1, 0);
lean_inc(x_3);
x_549 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__2___boxed), 3, 1);
lean_closure_set(x_549, 0, x_3);
x_550 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_551 = lp_aesop_Aesop_withNormTraceNode___closed__7;
if (lean_obj_tag(x_546) == 0)
{
lean_object* x_722; 
x_722 = lean_box(0);
x_700 = x_722;
goto block_721;
}
else
{
lean_object* x_723; lean_object* x_724; 
x_723 = lean_ctor_get(x_546, 0);
x_724 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_723);
x_700 = x_724;
goto block_721;
}
block_601:
{
if (x_2 == 0)
{
if (lean_obj_tag(x_552) == 0)
{
lean_object* x_561; lean_object* x_562; lean_object* x_563; lean_object* x_564; lean_object* x_565; lean_object* x_566; lean_object* x_567; lean_object* x_568; lean_object* x_569; lean_object* x_570; lean_object* x_571; lean_object* x_572; lean_object* x_573; 
lean_dec(x_547);
x_561 = l_Lean_Meta_instMonadMCtxMetaM;
x_562 = lean_ctor_get(x_561, 0);
lean_inc(x_562);
x_563 = lean_ctor_get(x_561, 1);
lean_inc(x_563);
if (lean_is_exclusive(x_561)) {
 lean_ctor_release(x_561, 0);
 lean_ctor_release(x_561, 1);
 x_564 = x_561;
} else {
 lean_dec_ref(x_561);
 x_564 = lean_box(0);
}
x_565 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_565, 0, x_563);
lean_closure_set(x_565, 1, x_551);
x_566 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_566, 0, lean_box(0));
lean_closure_set(x_566, 1, lean_box(0));
lean_closure_set(x_566, 2, lean_box(0));
lean_closure_set(x_566, 3, lean_box(0));
lean_closure_set(x_566, 4, x_562);
x_567 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_567, 0, x_565);
lean_closure_set(x_567, 1, x_551);
x_568 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_568, 0, lean_box(0));
lean_closure_set(x_568, 1, lean_box(0));
lean_closure_set(x_568, 2, lean_box(0));
lean_closure_set(x_568, 3, lean_box(0));
lean_closure_set(x_568, 4, x_566);
x_569 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_569, 0, x_567);
lean_closure_set(x_569, 1, x_550);
x_570 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_570, 0, lean_box(0));
lean_closure_set(x_570, 1, x_568);
if (lean_is_scalar(x_564)) {
 x_571 = lean_alloc_ctor(0, 2, 0);
} else {
 x_571 = x_564;
}
lean_ctor_set(x_571, 0, x_570);
lean_ctor_set(x_571, 1, x_569);
lean_inc_ref(x_534);
x_572 = l_Lean_MVarId_isAssigned___redArg(x_534, x_571, x_3);
lean_inc(x_559);
lean_inc_ref(x_558);
lean_inc(x_557);
lean_inc_ref(x_556);
lean_inc(x_555);
lean_inc(x_554);
lean_inc_ref(x_553);
x_573 = lean_apply_8(x_572, x_553, x_554, x_555, x_556, x_557, x_558, x_559, lean_box(0));
if (lean_obj_tag(x_573) == 0)
{
lean_object* x_574; lean_object* x_575; uint8_t x_576; 
x_574 = lean_ctor_get(x_573, 0);
lean_inc(x_574);
if (lean_is_exclusive(x_573)) {
 lean_ctor_release(x_573, 0);
 x_575 = x_573;
} else {
 lean_dec_ref(x_573);
 x_575 = lean_box(0);
}
x_576 = lean_unbox(x_574);
lean_dec(x_574);
if (x_576 == 0)
{
lean_object* x_577; 
lean_dec(x_559);
lean_dec_ref(x_558);
lean_dec(x_557);
lean_dec_ref(x_556);
lean_dec(x_555);
lean_dec(x_554);
lean_dec_ref(x_553);
lean_dec_ref(x_534);
lean_dec_ref(x_1);
if (lean_is_scalar(x_575)) {
 x_577 = lean_alloc_ctor(0, 1, 0);
} else {
 x_577 = x_575;
}
lean_ctor_set(x_577, 0, x_546);
return x_577;
}
else
{
lean_object* x_578; lean_object* x_579; lean_object* x_580; lean_object* x_581; lean_object* x_582; lean_object* x_583; lean_object* x_584; lean_object* x_585; lean_object* x_586; lean_object* x_587; lean_object* x_588; lean_object* x_589; lean_object* x_590; 
lean_dec(x_575);
x_578 = lp_aesop_Aesop_checkSimp___closed__14;
x_579 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_580 = lean_ctor_get(x_579, 0);
lean_inc_ref(x_580);
x_581 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_534);
x_582 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_581, x_534);
x_583 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_583, 0, x_578);
lean_ctor_set(x_583, 1, x_580);
lean_ctor_set(x_583, 2, x_582);
x_584 = lp_aesop_Aesop_checkSimp___closed__17;
x_585 = l_Lean_stringToMessageData(x_1);
x_586 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_586, 0, x_584);
lean_ctor_set(x_586, 1, x_585);
x_587 = lp_aesop_Aesop_checkSimp___closed__19;
x_588 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_588, 0, x_586);
lean_ctor_set(x_588, 1, x_587);
x_589 = l_Lean_throwError___redArg(x_534, x_583, x_588);
x_590 = lean_apply_8(x_589, x_553, x_554, x_555, x_556, x_557, x_558, x_559, lean_box(0));
if (lean_obj_tag(x_590) == 0)
{
lean_object* x_591; lean_object* x_592; 
if (lean_is_exclusive(x_590)) {
 lean_ctor_release(x_590, 0);
 x_591 = x_590;
} else {
 lean_dec_ref(x_590);
 x_591 = lean_box(0);
}
if (lean_is_scalar(x_591)) {
 x_592 = lean_alloc_ctor(0, 1, 0);
} else {
 x_592 = x_591;
}
lean_ctor_set(x_592, 0, x_546);
return x_592;
}
else
{
lean_object* x_593; lean_object* x_594; lean_object* x_595; 
lean_dec(x_546);
x_593 = lean_ctor_get(x_590, 0);
lean_inc(x_593);
if (lean_is_exclusive(x_590)) {
 lean_ctor_release(x_590, 0);
 x_594 = x_590;
} else {
 lean_dec_ref(x_590);
 x_594 = lean_box(0);
}
if (lean_is_scalar(x_594)) {
 x_595 = lean_alloc_ctor(1, 1, 0);
} else {
 x_595 = x_594;
}
lean_ctor_set(x_595, 0, x_593);
return x_595;
}
}
}
else
{
lean_object* x_596; lean_object* x_597; lean_object* x_598; 
lean_dec(x_559);
lean_dec_ref(x_558);
lean_dec(x_557);
lean_dec_ref(x_556);
lean_dec(x_555);
lean_dec(x_554);
lean_dec_ref(x_553);
lean_dec(x_546);
lean_dec_ref(x_534);
lean_dec_ref(x_1);
x_596 = lean_ctor_get(x_573, 0);
lean_inc(x_596);
if (lean_is_exclusive(x_573)) {
 lean_ctor_release(x_573, 0);
 x_597 = x_573;
} else {
 lean_dec_ref(x_573);
 x_597 = lean_box(0);
}
if (lean_is_scalar(x_597)) {
 x_598 = lean_alloc_ctor(1, 1, 0);
} else {
 x_598 = x_597;
}
lean_ctor_set(x_598, 0, x_596);
return x_598;
}
}
else
{
lean_object* x_599; 
lean_dec(x_559);
lean_dec_ref(x_558);
lean_dec(x_557);
lean_dec_ref(x_556);
lean_dec(x_555);
lean_dec(x_554);
lean_dec_ref(x_553);
lean_dec_ref(x_552);
lean_dec_ref(x_534);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_547)) {
 x_599 = lean_alloc_ctor(0, 1, 0);
} else {
 x_599 = x_547;
}
lean_ctor_set(x_599, 0, x_546);
return x_599;
}
}
else
{
lean_object* x_600; 
lean_dec(x_559);
lean_dec_ref(x_558);
lean_dec(x_557);
lean_dec_ref(x_556);
lean_dec(x_555);
lean_dec(x_554);
lean_dec_ref(x_553);
lean_dec(x_552);
lean_dec_ref(x_534);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_547)) {
 x_600 = lean_alloc_ctor(0, 1, 0);
} else {
 x_600 = x_547;
}
lean_ctor_set(x_600, 0, x_546);
return x_600;
}
}
block_640:
{
uint8_t x_613; 
x_613 = l_Array_isEmpty___redArg(x_612);
lean_dec_ref(x_612);
if (x_613 == 0)
{
lean_object* x_614; lean_object* x_615; lean_object* x_616; lean_object* x_617; lean_object* x_618; lean_object* x_619; lean_object* x_620; lean_object* x_621; lean_object* x_622; lean_object* x_623; lean_object* x_624; lean_object* x_625; size_t x_626; size_t x_627; lean_object* x_628; lean_object* x_629; lean_object* x_630; lean_object* x_631; lean_object* x_632; lean_object* x_633; lean_object* x_634; lean_object* x_635; lean_object* x_636; 
x_614 = lp_aesop_Aesop_checkSimp___closed__14;
x_615 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_616 = lean_ctor_get(x_615, 0);
lean_inc_ref(x_616);
x_617 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_534);
x_618 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_617, x_534);
x_619 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_619, 0, x_614);
lean_ctor_set(x_619, 1, x_616);
lean_ctor_set(x_619, 2, x_618);
x_620 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_621 = l_Lean_stringToMessageData(x_1);
x_622 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_622, 0, x_620);
lean_ctor_set(x_622, 1, x_621);
x_623 = lp_aesop_Aesop_checkSimp___closed__21;
x_624 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_624, 0, x_622);
lean_ctor_set(x_624, 1, x_623);
x_625 = lp_aesop_Aesop_checkSimp___closed__31;
x_626 = lean_array_size(x_608);
x_627 = 0;
x_628 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_625, x_548, x_626, x_627, x_608);
x_629 = lean_array_to_list(x_628);
x_630 = lp_aesop_Aesop_checkSimp___closed__32;
x_631 = lean_box(0);
x_632 = l_List_mapTR_loop___redArg(x_630, x_629, x_631);
x_633 = l_Lean_MessageData_ofList(x_632);
x_634 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_634, 0, x_624);
lean_ctor_set(x_634, 1, x_633);
lean_inc_ref(x_534);
x_635 = l_Lean_throwError___redArg(x_534, x_619, x_634);
lean_inc(x_603);
lean_inc_ref(x_604);
lean_inc(x_605);
lean_inc_ref(x_609);
lean_inc(x_606);
lean_inc(x_602);
lean_inc_ref(x_610);
x_636 = lean_apply_8(x_635, x_610, x_602, x_606, x_609, x_605, x_604, x_603, lean_box(0));
if (lean_obj_tag(x_636) == 0)
{
lean_dec_ref(x_636);
x_552 = x_611;
x_553 = x_610;
x_554 = x_602;
x_555 = x_606;
x_556 = x_609;
x_557 = x_605;
x_558 = x_604;
x_559 = x_603;
x_560 = lean_box(0);
goto block_601;
}
else
{
lean_object* x_637; lean_object* x_638; lean_object* x_639; 
lean_dec(x_611);
lean_dec_ref(x_610);
lean_dec_ref(x_609);
lean_dec(x_606);
lean_dec(x_605);
lean_dec_ref(x_604);
lean_dec(x_603);
lean_dec(x_602);
lean_dec(x_547);
lean_dec(x_546);
lean_dec_ref(x_534);
lean_dec(x_3);
lean_dec_ref(x_1);
x_637 = lean_ctor_get(x_636, 0);
lean_inc(x_637);
if (lean_is_exclusive(x_636)) {
 lean_ctor_release(x_636, 0);
 x_638 = x_636;
} else {
 lean_dec_ref(x_636);
 x_638 = lean_box(0);
}
if (lean_is_scalar(x_638)) {
 x_639 = lean_alloc_ctor(1, 1, 0);
} else {
 x_639 = x_638;
}
lean_ctor_set(x_639, 0, x_637);
return x_639;
}
}
else
{
lean_dec_ref(x_608);
lean_dec_ref(x_548);
x_552 = x_611;
x_553 = x_610;
x_554 = x_602;
x_555 = x_606;
x_556 = x_609;
x_557 = x_605;
x_558 = x_604;
x_559 = x_603;
x_560 = lean_box(0);
goto block_601;
}
}
block_666:
{
lean_object* x_653; 
lean_inc(x_651);
lean_inc_ref(x_650);
lean_inc(x_649);
lean_inc_ref(x_648);
x_653 = lp_batteries_Lean_Meta_getAssignedExprMVars(x_544, x_642, x_648, x_649, x_650, x_651);
if (lean_obj_tag(x_653) == 0)
{
lean_object* x_654; lean_object* x_655; lean_object* x_656; lean_object* x_657; uint8_t x_658; 
x_654 = lean_ctor_get(x_653, 0);
lean_inc(x_654);
lean_dec_ref(x_653);
x_655 = lean_array_get_size(x_654);
x_656 = lean_mk_empty_array_with_capacity(x_641);
x_657 = lp_aesop_Aesop_checkSimp___closed__31;
x_658 = lean_nat_dec_lt(x_641, x_655);
if (x_658 == 0)
{
lean_dec(x_654);
lean_dec_ref(x_549);
x_602 = x_646;
x_603 = x_651;
x_604 = x_650;
x_605 = x_649;
x_606 = x_647;
x_607 = lean_box(0);
x_608 = x_643;
x_609 = x_648;
x_610 = x_645;
x_611 = x_644;
x_612 = x_656;
goto block_640;
}
else
{
uint8_t x_659; 
x_659 = lean_nat_dec_le(x_655, x_655);
if (x_659 == 0)
{
lean_dec(x_654);
lean_dec_ref(x_549);
x_602 = x_646;
x_603 = x_651;
x_604 = x_650;
x_605 = x_649;
x_606 = x_647;
x_607 = lean_box(0);
x_608 = x_643;
x_609 = x_648;
x_610 = x_645;
x_611 = x_644;
x_612 = x_656;
goto block_640;
}
else
{
size_t x_660; size_t x_661; lean_object* x_662; 
x_660 = 0;
x_661 = lean_usize_of_nat(x_655);
x_662 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_657, x_549, x_654, x_660, x_661, x_656);
x_602 = x_646;
x_603 = x_651;
x_604 = x_650;
x_605 = x_649;
x_606 = x_647;
x_607 = lean_box(0);
x_608 = x_643;
x_609 = x_648;
x_610 = x_645;
x_611 = x_644;
x_612 = x_662;
goto block_640;
}
}
}
else
{
lean_object* x_663; lean_object* x_664; lean_object* x_665; 
lean_dec(x_651);
lean_dec_ref(x_650);
lean_dec(x_649);
lean_dec_ref(x_648);
lean_dec(x_647);
lean_dec(x_646);
lean_dec_ref(x_645);
lean_dec(x_644);
lean_dec_ref(x_643);
lean_dec_ref(x_549);
lean_dec_ref(x_548);
lean_dec(x_547);
lean_dec(x_546);
lean_dec_ref(x_534);
lean_dec(x_3);
lean_dec_ref(x_1);
x_663 = lean_ctor_get(x_653, 0);
lean_inc(x_663);
if (lean_is_exclusive(x_653)) {
 lean_ctor_release(x_653, 0);
 x_664 = x_653;
} else {
 lean_dec_ref(x_653);
 x_664 = lean_box(0);
}
if (lean_is_scalar(x_664)) {
 x_665 = lean_alloc_ctor(1, 1, 0);
} else {
 x_665 = x_664;
}
lean_ctor_set(x_665, 0, x_663);
return x_665;
}
}
block_699:
{
uint8_t x_672; 
x_672 = l_Array_isEmpty___redArg(x_671);
if (x_672 == 0)
{
lean_object* x_673; lean_object* x_674; lean_object* x_675; lean_object* x_676; lean_object* x_677; lean_object* x_678; lean_object* x_679; lean_object* x_680; lean_object* x_681; lean_object* x_682; lean_object* x_683; lean_object* x_684; size_t x_685; size_t x_686; lean_object* x_687; lean_object* x_688; lean_object* x_689; lean_object* x_690; lean_object* x_691; lean_object* x_692; lean_object* x_693; lean_object* x_694; lean_object* x_695; 
x_673 = lp_aesop_Aesop_checkSimp___closed__14;
x_674 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_675 = lean_ctor_get(x_674, 0);
lean_inc_ref(x_675);
x_676 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_534);
x_677 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_676, x_534);
x_678 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_678, 0, x_673);
lean_ctor_set(x_678, 1, x_675);
lean_ctor_set(x_678, 2, x_677);
x_679 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_680 = l_Lean_stringToMessageData(x_1);
x_681 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_681, 0, x_679);
lean_ctor_set(x_681, 1, x_680);
x_682 = lp_aesop_Aesop_checkSimp___closed__34;
x_683 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_683, 0, x_681);
lean_ctor_set(x_683, 1, x_682);
x_684 = lp_aesop_Aesop_checkSimp___closed__31;
x_685 = lean_array_size(x_671);
x_686 = 0;
lean_inc_ref(x_671);
lean_inc_ref(x_548);
x_687 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_684, x_548, x_685, x_686, x_671);
x_688 = lean_array_to_list(x_687);
x_689 = lp_aesop_Aesop_checkSimp___closed__32;
x_690 = lean_box(0);
x_691 = l_List_mapTR_loop___redArg(x_689, x_688, x_690);
x_692 = l_Lean_MessageData_ofList(x_691);
x_693 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_693, 0, x_683);
lean_ctor_set(x_693, 1, x_692);
lean_inc_ref(x_534);
x_694 = l_Lean_throwError___redArg(x_534, x_678, x_693);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_695 = lean_apply_8(x_694, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_695) == 0)
{
lean_dec_ref(x_695);
x_641 = x_667;
x_642 = x_668;
x_643 = x_671;
x_644 = x_670;
x_645 = x_5;
x_646 = x_6;
x_647 = x_7;
x_648 = x_8;
x_649 = x_9;
x_650 = x_10;
x_651 = x_11;
x_652 = lean_box(0);
goto block_666;
}
else
{
lean_object* x_696; lean_object* x_697; lean_object* x_698; 
lean_dec_ref(x_671);
lean_dec(x_670);
lean_dec_ref(x_668);
lean_dec_ref(x_549);
lean_dec_ref(x_548);
lean_dec(x_547);
lean_dec(x_546);
lean_dec(x_544);
lean_dec_ref(x_534);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_696 = lean_ctor_get(x_695, 0);
lean_inc(x_696);
if (lean_is_exclusive(x_695)) {
 lean_ctor_release(x_695, 0);
 x_697 = x_695;
} else {
 lean_dec_ref(x_695);
 x_697 = lean_box(0);
}
if (lean_is_scalar(x_697)) {
 x_698 = lean_alloc_ctor(1, 1, 0);
} else {
 x_698 = x_697;
}
lean_ctor_set(x_698, 0, x_696);
return x_698;
}
}
else
{
x_641 = x_667;
x_642 = x_668;
x_643 = x_671;
x_644 = x_670;
x_645 = x_5;
x_646 = x_6;
x_647 = x_7;
x_648 = x_8;
x_649 = x_9;
x_650 = x_10;
x_651 = x_11;
x_652 = lean_box(0);
goto block_666;
}
}
block_721:
{
lean_object* x_701; 
x_701 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_701) == 0)
{
lean_object* x_702; lean_object* x_703; 
x_702 = lean_ctor_get(x_701, 0);
lean_inc(x_702);
lean_dec_ref(x_701);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_702);
lean_inc(x_544);
x_703 = lp_batteries_Lean_Meta_getIntroducedExprMVars(x_544, x_702, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_703) == 0)
{
lean_object* x_704; lean_object* x_705; lean_object* x_706; lean_object* x_707; lean_object* x_708; uint8_t x_709; 
x_704 = lean_ctor_get(x_703, 0);
lean_inc(x_704);
lean_dec_ref(x_703);
x_705 = lean_unsigned_to_nat(0u);
x_706 = lean_array_get_size(x_704);
x_707 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_708 = lp_aesop_Aesop_checkSimp___closed__31;
x_709 = lean_nat_dec_lt(x_705, x_706);
if (x_709 == 0)
{
lean_dec(x_704);
x_667 = x_705;
x_668 = x_702;
x_669 = lean_box(0);
x_670 = x_700;
x_671 = x_707;
goto block_699;
}
else
{
uint8_t x_710; 
x_710 = lean_nat_dec_le(x_706, x_706);
if (x_710 == 0)
{
lean_dec(x_704);
x_667 = x_705;
x_668 = x_702;
x_669 = lean_box(0);
x_670 = x_700;
x_671 = x_707;
goto block_699;
}
else
{
lean_object* x_711; size_t x_712; size_t x_713; lean_object* x_714; 
lean_inc(x_700);
x_711 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__1), 3, 1);
lean_closure_set(x_711, 0, x_700);
x_712 = 0;
x_713 = lean_usize_of_nat(x_706);
x_714 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_708, x_711, x_704, x_712, x_713, x_707);
x_667 = x_705;
x_668 = x_702;
x_669 = lean_box(0);
x_670 = x_700;
x_671 = x_714;
goto block_699;
}
}
}
else
{
lean_object* x_715; lean_object* x_716; lean_object* x_717; 
lean_dec(x_702);
lean_dec(x_700);
lean_dec_ref(x_549);
lean_dec_ref(x_548);
lean_dec(x_547);
lean_dec(x_546);
lean_dec(x_544);
lean_dec_ref(x_534);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_715 = lean_ctor_get(x_703, 0);
lean_inc(x_715);
if (lean_is_exclusive(x_703)) {
 lean_ctor_release(x_703, 0);
 x_716 = x_703;
} else {
 lean_dec_ref(x_703);
 x_716 = lean_box(0);
}
if (lean_is_scalar(x_716)) {
 x_717 = lean_alloc_ctor(1, 1, 0);
} else {
 x_717 = x_716;
}
lean_ctor_set(x_717, 0, x_715);
return x_717;
}
}
else
{
lean_object* x_718; lean_object* x_719; lean_object* x_720; 
lean_dec(x_700);
lean_dec_ref(x_549);
lean_dec_ref(x_548);
lean_dec(x_547);
lean_dec(x_546);
lean_dec(x_544);
lean_dec_ref(x_534);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_718 = lean_ctor_get(x_701, 0);
lean_inc(x_718);
if (lean_is_exclusive(x_701)) {
 lean_ctor_release(x_701, 0);
 x_719 = x_701;
} else {
 lean_dec_ref(x_701);
 x_719 = lean_box(0);
}
if (lean_is_scalar(x_719)) {
 x_720 = lean_alloc_ctor(1, 1, 0);
} else {
 x_720 = x_719;
}
lean_ctor_set(x_720, 0, x_718);
return x_720;
}
}
}
else
{
lean_dec(x_544);
lean_dec_ref(x_534);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_545;
}
}
else
{
lean_object* x_725; lean_object* x_726; lean_object* x_727; 
lean_dec_ref(x_534);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_725 = lean_ctor_get(x_543, 0);
lean_inc(x_725);
if (lean_is_exclusive(x_543)) {
 lean_ctor_release(x_543, 0);
 x_726 = x_543;
} else {
 lean_dec_ref(x_543);
 x_726 = lean_box(0);
}
if (lean_is_scalar(x_726)) {
 x_727 = lean_alloc_ctor(1, 1, 0);
} else {
 x_727 = x_726;
}
lean_ctor_set(x_727, 0, x_725);
return x_727;
}
}
}
else
{
lean_object* x_728; lean_object* x_729; lean_object* x_730; 
lean_dec_ref(x_534);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_728 = lean_ctor_get(x_539, 0);
lean_inc(x_728);
if (lean_is_exclusive(x_539)) {
 lean_ctor_release(x_539, 0);
 x_729 = x_539;
} else {
 lean_dec_ref(x_539);
 x_729 = lean_box(0);
}
if (lean_is_scalar(x_729)) {
 x_730 = lean_alloc_ctor(1, 1, 0);
} else {
 x_730 = x_729;
}
lean_ctor_set(x_730, 0, x_728);
return x_730;
}
}
}
else
{
lean_object* x_731; lean_object* x_732; lean_object* x_733; lean_object* x_734; lean_object* x_735; lean_object* x_736; lean_object* x_737; lean_object* x_738; lean_object* x_739; lean_object* x_740; lean_object* x_741; lean_object* x_742; lean_object* x_743; lean_object* x_744; lean_object* x_745; lean_object* x_746; lean_object* x_747; lean_object* x_748; lean_object* x_749; lean_object* x_750; lean_object* x_751; lean_object* x_752; lean_object* x_753; lean_object* x_754; lean_object* x_755; lean_object* x_756; lean_object* x_757; lean_object* x_758; lean_object* x_759; lean_object* x_760; lean_object* x_761; lean_object* x_762; lean_object* x_763; lean_object* x_764; lean_object* x_765; lean_object* x_766; lean_object* x_767; lean_object* x_768; lean_object* x_769; 
x_731 = lean_ctor_get(x_15, 0);
x_732 = lean_ctor_get(x_15, 2);
x_733 = lean_ctor_get(x_15, 3);
x_734 = lean_ctor_get(x_15, 4);
lean_inc(x_734);
lean_inc(x_733);
lean_inc(x_732);
lean_inc(x_731);
lean_dec(x_15);
x_735 = lp_aesop_Aesop_withNormTraceNode___closed__2;
x_736 = lp_aesop_Aesop_withNormTraceNode___closed__3;
lean_inc_ref(x_731);
x_737 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_737, 0, x_731);
x_738 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_738, 0, x_731);
x_739 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_739, 0, x_737);
lean_ctor_set(x_739, 1, x_738);
x_740 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_740, 0, x_734);
x_741 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_741, 0, x_733);
x_742 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_742, 0, x_732);
x_743 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_743, 0, x_739);
lean_ctor_set(x_743, 1, x_735);
lean_ctor_set(x_743, 2, x_742);
lean_ctor_set(x_743, 3, x_741);
lean_ctor_set(x_743, 4, x_740);
lean_ctor_set(x_13, 1, x_736);
lean_ctor_set(x_13, 0, x_743);
x_744 = l_ReaderT_instMonad___redArg(x_13);
x_745 = lean_ctor_get(x_744, 0);
lean_inc_ref(x_745);
if (lean_is_exclusive(x_744)) {
 lean_ctor_release(x_744, 0);
 lean_ctor_release(x_744, 1);
 x_746 = x_744;
} else {
 lean_dec_ref(x_744);
 x_746 = lean_box(0);
}
x_747 = lean_ctor_get(x_745, 0);
lean_inc_ref(x_747);
x_748 = lean_ctor_get(x_745, 2);
lean_inc(x_748);
x_749 = lean_ctor_get(x_745, 3);
lean_inc(x_749);
x_750 = lean_ctor_get(x_745, 4);
lean_inc(x_750);
if (lean_is_exclusive(x_745)) {
 lean_ctor_release(x_745, 0);
 lean_ctor_release(x_745, 1);
 lean_ctor_release(x_745, 2);
 lean_ctor_release(x_745, 3);
 lean_ctor_release(x_745, 4);
 x_751 = x_745;
} else {
 lean_dec_ref(x_745);
 x_751 = lean_box(0);
}
x_752 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_753 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_747);
x_754 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_754, 0, x_747);
x_755 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_755, 0, x_747);
x_756 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_756, 0, x_754);
lean_ctor_set(x_756, 1, x_755);
x_757 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_757, 0, x_750);
x_758 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_758, 0, x_749);
x_759 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_759, 0, x_748);
if (lean_is_scalar(x_751)) {
 x_760 = lean_alloc_ctor(0, 5, 0);
} else {
 x_760 = x_751;
}
lean_ctor_set(x_760, 0, x_756);
lean_ctor_set(x_760, 1, x_752);
lean_ctor_set(x_760, 2, x_759);
lean_ctor_set(x_760, 3, x_758);
lean_ctor_set(x_760, 4, x_757);
if (lean_is_scalar(x_746)) {
 x_761 = lean_alloc_ctor(0, 2, 0);
} else {
 x_761 = x_746;
}
lean_ctor_set(x_761, 0, x_760);
lean_ctor_set(x_761, 1, x_753);
x_762 = l_ReaderT_instMonad___redArg(x_761);
x_763 = l_ReaderT_instMonad___redArg(x_762);
x_764 = l_ReaderT_instMonad___redArg(x_763);
x_765 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_766 = lean_ctor_get(x_765, 0);
lean_inc(x_766);
x_767 = lp_aesop_Aesop_runNormRuleTac___closed__8;
lean_inc_ref(x_764);
x_768 = lp_aesop_Aesop_Check_isEnabled___redArg(x_764, x_766, x_767);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_769 = lean_apply_8(x_768, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_769) == 0)
{
lean_object* x_770; uint8_t x_771; 
x_770 = lean_ctor_get(x_769, 0);
lean_inc(x_770);
lean_dec_ref(x_769);
x_771 = lean_unbox(x_770);
lean_dec(x_770);
if (x_771 == 0)
{
lean_object* x_772; 
lean_dec_ref(x_764);
lean_dec(x_3);
lean_dec_ref(x_1);
x_772 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
return x_772;
}
else
{
lean_object* x_773; 
x_773 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_773) == 0)
{
lean_object* x_774; lean_object* x_775; 
x_774 = lean_ctor_get(x_773, 0);
lean_inc(x_774);
lean_dec_ref(x_773);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_775 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_775) == 0)
{
lean_object* x_776; lean_object* x_777; lean_object* x_778; lean_object* x_779; lean_object* x_780; lean_object* x_781; lean_object* x_782; lean_object* x_783; lean_object* x_784; lean_object* x_785; lean_object* x_786; lean_object* x_787; lean_object* x_788; lean_object* x_789; lean_object* x_790; lean_object* x_832; lean_object* x_833; lean_object* x_834; lean_object* x_835; lean_object* x_836; lean_object* x_837; lean_object* x_838; lean_object* x_839; lean_object* x_840; lean_object* x_841; lean_object* x_842; lean_object* x_871; lean_object* x_872; lean_object* x_873; lean_object* x_874; lean_object* x_875; lean_object* x_876; lean_object* x_877; lean_object* x_878; lean_object* x_879; lean_object* x_880; lean_object* x_881; lean_object* x_882; lean_object* x_897; lean_object* x_898; lean_object* x_899; lean_object* x_900; lean_object* x_901; lean_object* x_930; 
x_776 = lean_ctor_get(x_775, 0);
lean_inc(x_776);
if (lean_is_exclusive(x_775)) {
 lean_ctor_release(x_775, 0);
 x_777 = x_775;
} else {
 lean_dec_ref(x_775);
 x_777 = lean_box(0);
}
x_778 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__0___boxed), 1, 0);
lean_inc(x_3);
x_779 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__2___boxed), 3, 1);
lean_closure_set(x_779, 0, x_3);
x_780 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_781 = lp_aesop_Aesop_withNormTraceNode___closed__7;
if (lean_obj_tag(x_776) == 0)
{
lean_object* x_952; 
x_952 = lean_box(0);
x_930 = x_952;
goto block_951;
}
else
{
lean_object* x_953; lean_object* x_954; 
x_953 = lean_ctor_get(x_776, 0);
x_954 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_953);
x_930 = x_954;
goto block_951;
}
block_831:
{
if (x_2 == 0)
{
if (lean_obj_tag(x_782) == 0)
{
lean_object* x_791; lean_object* x_792; lean_object* x_793; lean_object* x_794; lean_object* x_795; lean_object* x_796; lean_object* x_797; lean_object* x_798; lean_object* x_799; lean_object* x_800; lean_object* x_801; lean_object* x_802; lean_object* x_803; 
lean_dec(x_777);
x_791 = l_Lean_Meta_instMonadMCtxMetaM;
x_792 = lean_ctor_get(x_791, 0);
lean_inc(x_792);
x_793 = lean_ctor_get(x_791, 1);
lean_inc(x_793);
if (lean_is_exclusive(x_791)) {
 lean_ctor_release(x_791, 0);
 lean_ctor_release(x_791, 1);
 x_794 = x_791;
} else {
 lean_dec_ref(x_791);
 x_794 = lean_box(0);
}
x_795 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_795, 0, x_793);
lean_closure_set(x_795, 1, x_781);
x_796 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_796, 0, lean_box(0));
lean_closure_set(x_796, 1, lean_box(0));
lean_closure_set(x_796, 2, lean_box(0));
lean_closure_set(x_796, 3, lean_box(0));
lean_closure_set(x_796, 4, x_792);
x_797 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_797, 0, x_795);
lean_closure_set(x_797, 1, x_781);
x_798 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_798, 0, lean_box(0));
lean_closure_set(x_798, 1, lean_box(0));
lean_closure_set(x_798, 2, lean_box(0));
lean_closure_set(x_798, 3, lean_box(0));
lean_closure_set(x_798, 4, x_796);
x_799 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_799, 0, x_797);
lean_closure_set(x_799, 1, x_780);
x_800 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_800, 0, lean_box(0));
lean_closure_set(x_800, 1, x_798);
if (lean_is_scalar(x_794)) {
 x_801 = lean_alloc_ctor(0, 2, 0);
} else {
 x_801 = x_794;
}
lean_ctor_set(x_801, 0, x_800);
lean_ctor_set(x_801, 1, x_799);
lean_inc_ref(x_764);
x_802 = l_Lean_MVarId_isAssigned___redArg(x_764, x_801, x_3);
lean_inc(x_789);
lean_inc_ref(x_788);
lean_inc(x_787);
lean_inc_ref(x_786);
lean_inc(x_785);
lean_inc(x_784);
lean_inc_ref(x_783);
x_803 = lean_apply_8(x_802, x_783, x_784, x_785, x_786, x_787, x_788, x_789, lean_box(0));
if (lean_obj_tag(x_803) == 0)
{
lean_object* x_804; lean_object* x_805; uint8_t x_806; 
x_804 = lean_ctor_get(x_803, 0);
lean_inc(x_804);
if (lean_is_exclusive(x_803)) {
 lean_ctor_release(x_803, 0);
 x_805 = x_803;
} else {
 lean_dec_ref(x_803);
 x_805 = lean_box(0);
}
x_806 = lean_unbox(x_804);
lean_dec(x_804);
if (x_806 == 0)
{
lean_object* x_807; 
lean_dec(x_789);
lean_dec_ref(x_788);
lean_dec(x_787);
lean_dec_ref(x_786);
lean_dec(x_785);
lean_dec(x_784);
lean_dec_ref(x_783);
lean_dec_ref(x_764);
lean_dec_ref(x_1);
if (lean_is_scalar(x_805)) {
 x_807 = lean_alloc_ctor(0, 1, 0);
} else {
 x_807 = x_805;
}
lean_ctor_set(x_807, 0, x_776);
return x_807;
}
else
{
lean_object* x_808; lean_object* x_809; lean_object* x_810; lean_object* x_811; lean_object* x_812; lean_object* x_813; lean_object* x_814; lean_object* x_815; lean_object* x_816; lean_object* x_817; lean_object* x_818; lean_object* x_819; lean_object* x_820; 
lean_dec(x_805);
x_808 = lp_aesop_Aesop_checkSimp___closed__14;
x_809 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_810 = lean_ctor_get(x_809, 0);
lean_inc_ref(x_810);
x_811 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_764);
x_812 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_811, x_764);
x_813 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_813, 0, x_808);
lean_ctor_set(x_813, 1, x_810);
lean_ctor_set(x_813, 2, x_812);
x_814 = lp_aesop_Aesop_checkSimp___closed__17;
x_815 = l_Lean_stringToMessageData(x_1);
x_816 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_816, 0, x_814);
lean_ctor_set(x_816, 1, x_815);
x_817 = lp_aesop_Aesop_checkSimp___closed__19;
x_818 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_818, 0, x_816);
lean_ctor_set(x_818, 1, x_817);
x_819 = l_Lean_throwError___redArg(x_764, x_813, x_818);
x_820 = lean_apply_8(x_819, x_783, x_784, x_785, x_786, x_787, x_788, x_789, lean_box(0));
if (lean_obj_tag(x_820) == 0)
{
lean_object* x_821; lean_object* x_822; 
if (lean_is_exclusive(x_820)) {
 lean_ctor_release(x_820, 0);
 x_821 = x_820;
} else {
 lean_dec_ref(x_820);
 x_821 = lean_box(0);
}
if (lean_is_scalar(x_821)) {
 x_822 = lean_alloc_ctor(0, 1, 0);
} else {
 x_822 = x_821;
}
lean_ctor_set(x_822, 0, x_776);
return x_822;
}
else
{
lean_object* x_823; lean_object* x_824; lean_object* x_825; 
lean_dec(x_776);
x_823 = lean_ctor_get(x_820, 0);
lean_inc(x_823);
if (lean_is_exclusive(x_820)) {
 lean_ctor_release(x_820, 0);
 x_824 = x_820;
} else {
 lean_dec_ref(x_820);
 x_824 = lean_box(0);
}
if (lean_is_scalar(x_824)) {
 x_825 = lean_alloc_ctor(1, 1, 0);
} else {
 x_825 = x_824;
}
lean_ctor_set(x_825, 0, x_823);
return x_825;
}
}
}
else
{
lean_object* x_826; lean_object* x_827; lean_object* x_828; 
lean_dec(x_789);
lean_dec_ref(x_788);
lean_dec(x_787);
lean_dec_ref(x_786);
lean_dec(x_785);
lean_dec(x_784);
lean_dec_ref(x_783);
lean_dec(x_776);
lean_dec_ref(x_764);
lean_dec_ref(x_1);
x_826 = lean_ctor_get(x_803, 0);
lean_inc(x_826);
if (lean_is_exclusive(x_803)) {
 lean_ctor_release(x_803, 0);
 x_827 = x_803;
} else {
 lean_dec_ref(x_803);
 x_827 = lean_box(0);
}
if (lean_is_scalar(x_827)) {
 x_828 = lean_alloc_ctor(1, 1, 0);
} else {
 x_828 = x_827;
}
lean_ctor_set(x_828, 0, x_826);
return x_828;
}
}
else
{
lean_object* x_829; 
lean_dec(x_789);
lean_dec_ref(x_788);
lean_dec(x_787);
lean_dec_ref(x_786);
lean_dec(x_785);
lean_dec(x_784);
lean_dec_ref(x_783);
lean_dec_ref(x_782);
lean_dec_ref(x_764);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_777)) {
 x_829 = lean_alloc_ctor(0, 1, 0);
} else {
 x_829 = x_777;
}
lean_ctor_set(x_829, 0, x_776);
return x_829;
}
}
else
{
lean_object* x_830; 
lean_dec(x_789);
lean_dec_ref(x_788);
lean_dec(x_787);
lean_dec_ref(x_786);
lean_dec(x_785);
lean_dec(x_784);
lean_dec_ref(x_783);
lean_dec(x_782);
lean_dec_ref(x_764);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_777)) {
 x_830 = lean_alloc_ctor(0, 1, 0);
} else {
 x_830 = x_777;
}
lean_ctor_set(x_830, 0, x_776);
return x_830;
}
}
block_870:
{
uint8_t x_843; 
x_843 = l_Array_isEmpty___redArg(x_842);
lean_dec_ref(x_842);
if (x_843 == 0)
{
lean_object* x_844; lean_object* x_845; lean_object* x_846; lean_object* x_847; lean_object* x_848; lean_object* x_849; lean_object* x_850; lean_object* x_851; lean_object* x_852; lean_object* x_853; lean_object* x_854; lean_object* x_855; size_t x_856; size_t x_857; lean_object* x_858; lean_object* x_859; lean_object* x_860; lean_object* x_861; lean_object* x_862; lean_object* x_863; lean_object* x_864; lean_object* x_865; lean_object* x_866; 
x_844 = lp_aesop_Aesop_checkSimp___closed__14;
x_845 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_846 = lean_ctor_get(x_845, 0);
lean_inc_ref(x_846);
x_847 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_764);
x_848 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_847, x_764);
x_849 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_849, 0, x_844);
lean_ctor_set(x_849, 1, x_846);
lean_ctor_set(x_849, 2, x_848);
x_850 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_851 = l_Lean_stringToMessageData(x_1);
x_852 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_852, 0, x_850);
lean_ctor_set(x_852, 1, x_851);
x_853 = lp_aesop_Aesop_checkSimp___closed__21;
x_854 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_854, 0, x_852);
lean_ctor_set(x_854, 1, x_853);
x_855 = lp_aesop_Aesop_checkSimp___closed__31;
x_856 = lean_array_size(x_838);
x_857 = 0;
x_858 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_855, x_778, x_856, x_857, x_838);
x_859 = lean_array_to_list(x_858);
x_860 = lp_aesop_Aesop_checkSimp___closed__32;
x_861 = lean_box(0);
x_862 = l_List_mapTR_loop___redArg(x_860, x_859, x_861);
x_863 = l_Lean_MessageData_ofList(x_862);
x_864 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_864, 0, x_854);
lean_ctor_set(x_864, 1, x_863);
lean_inc_ref(x_764);
x_865 = l_Lean_throwError___redArg(x_764, x_849, x_864);
lean_inc(x_833);
lean_inc_ref(x_834);
lean_inc(x_835);
lean_inc_ref(x_839);
lean_inc(x_836);
lean_inc(x_832);
lean_inc_ref(x_840);
x_866 = lean_apply_8(x_865, x_840, x_832, x_836, x_839, x_835, x_834, x_833, lean_box(0));
if (lean_obj_tag(x_866) == 0)
{
lean_dec_ref(x_866);
x_782 = x_841;
x_783 = x_840;
x_784 = x_832;
x_785 = x_836;
x_786 = x_839;
x_787 = x_835;
x_788 = x_834;
x_789 = x_833;
x_790 = lean_box(0);
goto block_831;
}
else
{
lean_object* x_867; lean_object* x_868; lean_object* x_869; 
lean_dec(x_841);
lean_dec_ref(x_840);
lean_dec_ref(x_839);
lean_dec(x_836);
lean_dec(x_835);
lean_dec_ref(x_834);
lean_dec(x_833);
lean_dec(x_832);
lean_dec(x_777);
lean_dec(x_776);
lean_dec_ref(x_764);
lean_dec(x_3);
lean_dec_ref(x_1);
x_867 = lean_ctor_get(x_866, 0);
lean_inc(x_867);
if (lean_is_exclusive(x_866)) {
 lean_ctor_release(x_866, 0);
 x_868 = x_866;
} else {
 lean_dec_ref(x_866);
 x_868 = lean_box(0);
}
if (lean_is_scalar(x_868)) {
 x_869 = lean_alloc_ctor(1, 1, 0);
} else {
 x_869 = x_868;
}
lean_ctor_set(x_869, 0, x_867);
return x_869;
}
}
else
{
lean_dec_ref(x_838);
lean_dec_ref(x_778);
x_782 = x_841;
x_783 = x_840;
x_784 = x_832;
x_785 = x_836;
x_786 = x_839;
x_787 = x_835;
x_788 = x_834;
x_789 = x_833;
x_790 = lean_box(0);
goto block_831;
}
}
block_896:
{
lean_object* x_883; 
lean_inc(x_881);
lean_inc_ref(x_880);
lean_inc(x_879);
lean_inc_ref(x_878);
x_883 = lp_batteries_Lean_Meta_getAssignedExprMVars(x_774, x_872, x_878, x_879, x_880, x_881);
if (lean_obj_tag(x_883) == 0)
{
lean_object* x_884; lean_object* x_885; lean_object* x_886; lean_object* x_887; uint8_t x_888; 
x_884 = lean_ctor_get(x_883, 0);
lean_inc(x_884);
lean_dec_ref(x_883);
x_885 = lean_array_get_size(x_884);
x_886 = lean_mk_empty_array_with_capacity(x_871);
x_887 = lp_aesop_Aesop_checkSimp___closed__31;
x_888 = lean_nat_dec_lt(x_871, x_885);
if (x_888 == 0)
{
lean_dec(x_884);
lean_dec_ref(x_779);
x_832 = x_876;
x_833 = x_881;
x_834 = x_880;
x_835 = x_879;
x_836 = x_877;
x_837 = lean_box(0);
x_838 = x_873;
x_839 = x_878;
x_840 = x_875;
x_841 = x_874;
x_842 = x_886;
goto block_870;
}
else
{
uint8_t x_889; 
x_889 = lean_nat_dec_le(x_885, x_885);
if (x_889 == 0)
{
lean_dec(x_884);
lean_dec_ref(x_779);
x_832 = x_876;
x_833 = x_881;
x_834 = x_880;
x_835 = x_879;
x_836 = x_877;
x_837 = lean_box(0);
x_838 = x_873;
x_839 = x_878;
x_840 = x_875;
x_841 = x_874;
x_842 = x_886;
goto block_870;
}
else
{
size_t x_890; size_t x_891; lean_object* x_892; 
x_890 = 0;
x_891 = lean_usize_of_nat(x_885);
x_892 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_887, x_779, x_884, x_890, x_891, x_886);
x_832 = x_876;
x_833 = x_881;
x_834 = x_880;
x_835 = x_879;
x_836 = x_877;
x_837 = lean_box(0);
x_838 = x_873;
x_839 = x_878;
x_840 = x_875;
x_841 = x_874;
x_842 = x_892;
goto block_870;
}
}
}
else
{
lean_object* x_893; lean_object* x_894; lean_object* x_895; 
lean_dec(x_881);
lean_dec_ref(x_880);
lean_dec(x_879);
lean_dec_ref(x_878);
lean_dec(x_877);
lean_dec(x_876);
lean_dec_ref(x_875);
lean_dec(x_874);
lean_dec_ref(x_873);
lean_dec_ref(x_779);
lean_dec_ref(x_778);
lean_dec(x_777);
lean_dec(x_776);
lean_dec_ref(x_764);
lean_dec(x_3);
lean_dec_ref(x_1);
x_893 = lean_ctor_get(x_883, 0);
lean_inc(x_893);
if (lean_is_exclusive(x_883)) {
 lean_ctor_release(x_883, 0);
 x_894 = x_883;
} else {
 lean_dec_ref(x_883);
 x_894 = lean_box(0);
}
if (lean_is_scalar(x_894)) {
 x_895 = lean_alloc_ctor(1, 1, 0);
} else {
 x_895 = x_894;
}
lean_ctor_set(x_895, 0, x_893);
return x_895;
}
}
block_929:
{
uint8_t x_902; 
x_902 = l_Array_isEmpty___redArg(x_901);
if (x_902 == 0)
{
lean_object* x_903; lean_object* x_904; lean_object* x_905; lean_object* x_906; lean_object* x_907; lean_object* x_908; lean_object* x_909; lean_object* x_910; lean_object* x_911; lean_object* x_912; lean_object* x_913; lean_object* x_914; size_t x_915; size_t x_916; lean_object* x_917; lean_object* x_918; lean_object* x_919; lean_object* x_920; lean_object* x_921; lean_object* x_922; lean_object* x_923; lean_object* x_924; lean_object* x_925; 
x_903 = lp_aesop_Aesop_checkSimp___closed__14;
x_904 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_905 = lean_ctor_get(x_904, 0);
lean_inc_ref(x_905);
x_906 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_764);
x_907 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_906, x_764);
x_908 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_908, 0, x_903);
lean_ctor_set(x_908, 1, x_905);
lean_ctor_set(x_908, 2, x_907);
x_909 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_910 = l_Lean_stringToMessageData(x_1);
x_911 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_911, 0, x_909);
lean_ctor_set(x_911, 1, x_910);
x_912 = lp_aesop_Aesop_checkSimp___closed__34;
x_913 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_913, 0, x_911);
lean_ctor_set(x_913, 1, x_912);
x_914 = lp_aesop_Aesop_checkSimp___closed__31;
x_915 = lean_array_size(x_901);
x_916 = 0;
lean_inc_ref(x_901);
lean_inc_ref(x_778);
x_917 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_914, x_778, x_915, x_916, x_901);
x_918 = lean_array_to_list(x_917);
x_919 = lp_aesop_Aesop_checkSimp___closed__32;
x_920 = lean_box(0);
x_921 = l_List_mapTR_loop___redArg(x_919, x_918, x_920);
x_922 = l_Lean_MessageData_ofList(x_921);
x_923 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_923, 0, x_913);
lean_ctor_set(x_923, 1, x_922);
lean_inc_ref(x_764);
x_924 = l_Lean_throwError___redArg(x_764, x_908, x_923);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_925 = lean_apply_8(x_924, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_925) == 0)
{
lean_dec_ref(x_925);
x_871 = x_897;
x_872 = x_898;
x_873 = x_901;
x_874 = x_900;
x_875 = x_5;
x_876 = x_6;
x_877 = x_7;
x_878 = x_8;
x_879 = x_9;
x_880 = x_10;
x_881 = x_11;
x_882 = lean_box(0);
goto block_896;
}
else
{
lean_object* x_926; lean_object* x_927; lean_object* x_928; 
lean_dec_ref(x_901);
lean_dec(x_900);
lean_dec_ref(x_898);
lean_dec_ref(x_779);
lean_dec_ref(x_778);
lean_dec(x_777);
lean_dec(x_776);
lean_dec(x_774);
lean_dec_ref(x_764);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_926 = lean_ctor_get(x_925, 0);
lean_inc(x_926);
if (lean_is_exclusive(x_925)) {
 lean_ctor_release(x_925, 0);
 x_927 = x_925;
} else {
 lean_dec_ref(x_925);
 x_927 = lean_box(0);
}
if (lean_is_scalar(x_927)) {
 x_928 = lean_alloc_ctor(1, 1, 0);
} else {
 x_928 = x_927;
}
lean_ctor_set(x_928, 0, x_926);
return x_928;
}
}
else
{
x_871 = x_897;
x_872 = x_898;
x_873 = x_901;
x_874 = x_900;
x_875 = x_5;
x_876 = x_6;
x_877 = x_7;
x_878 = x_8;
x_879 = x_9;
x_880 = x_10;
x_881 = x_11;
x_882 = lean_box(0);
goto block_896;
}
}
block_951:
{
lean_object* x_931; 
x_931 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_931) == 0)
{
lean_object* x_932; lean_object* x_933; 
x_932 = lean_ctor_get(x_931, 0);
lean_inc(x_932);
lean_dec_ref(x_931);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_932);
lean_inc(x_774);
x_933 = lp_batteries_Lean_Meta_getIntroducedExprMVars(x_774, x_932, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_933) == 0)
{
lean_object* x_934; lean_object* x_935; lean_object* x_936; lean_object* x_937; lean_object* x_938; uint8_t x_939; 
x_934 = lean_ctor_get(x_933, 0);
lean_inc(x_934);
lean_dec_ref(x_933);
x_935 = lean_unsigned_to_nat(0u);
x_936 = lean_array_get_size(x_934);
x_937 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_938 = lp_aesop_Aesop_checkSimp___closed__31;
x_939 = lean_nat_dec_lt(x_935, x_936);
if (x_939 == 0)
{
lean_dec(x_934);
x_897 = x_935;
x_898 = x_932;
x_899 = lean_box(0);
x_900 = x_930;
x_901 = x_937;
goto block_929;
}
else
{
uint8_t x_940; 
x_940 = lean_nat_dec_le(x_936, x_936);
if (x_940 == 0)
{
lean_dec(x_934);
x_897 = x_935;
x_898 = x_932;
x_899 = lean_box(0);
x_900 = x_930;
x_901 = x_937;
goto block_929;
}
else
{
lean_object* x_941; size_t x_942; size_t x_943; lean_object* x_944; 
lean_inc(x_930);
x_941 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__1), 3, 1);
lean_closure_set(x_941, 0, x_930);
x_942 = 0;
x_943 = lean_usize_of_nat(x_936);
x_944 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_938, x_941, x_934, x_942, x_943, x_937);
x_897 = x_935;
x_898 = x_932;
x_899 = lean_box(0);
x_900 = x_930;
x_901 = x_944;
goto block_929;
}
}
}
else
{
lean_object* x_945; lean_object* x_946; lean_object* x_947; 
lean_dec(x_932);
lean_dec(x_930);
lean_dec_ref(x_779);
lean_dec_ref(x_778);
lean_dec(x_777);
lean_dec(x_776);
lean_dec(x_774);
lean_dec_ref(x_764);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_945 = lean_ctor_get(x_933, 0);
lean_inc(x_945);
if (lean_is_exclusive(x_933)) {
 lean_ctor_release(x_933, 0);
 x_946 = x_933;
} else {
 lean_dec_ref(x_933);
 x_946 = lean_box(0);
}
if (lean_is_scalar(x_946)) {
 x_947 = lean_alloc_ctor(1, 1, 0);
} else {
 x_947 = x_946;
}
lean_ctor_set(x_947, 0, x_945);
return x_947;
}
}
else
{
lean_object* x_948; lean_object* x_949; lean_object* x_950; 
lean_dec(x_930);
lean_dec_ref(x_779);
lean_dec_ref(x_778);
lean_dec(x_777);
lean_dec(x_776);
lean_dec(x_774);
lean_dec_ref(x_764);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_948 = lean_ctor_get(x_931, 0);
lean_inc(x_948);
if (lean_is_exclusive(x_931)) {
 lean_ctor_release(x_931, 0);
 x_949 = x_931;
} else {
 lean_dec_ref(x_931);
 x_949 = lean_box(0);
}
if (lean_is_scalar(x_949)) {
 x_950 = lean_alloc_ctor(1, 1, 0);
} else {
 x_950 = x_949;
}
lean_ctor_set(x_950, 0, x_948);
return x_950;
}
}
}
else
{
lean_dec(x_774);
lean_dec_ref(x_764);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_775;
}
}
else
{
lean_object* x_955; lean_object* x_956; lean_object* x_957; 
lean_dec_ref(x_764);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_955 = lean_ctor_get(x_773, 0);
lean_inc(x_955);
if (lean_is_exclusive(x_773)) {
 lean_ctor_release(x_773, 0);
 x_956 = x_773;
} else {
 lean_dec_ref(x_773);
 x_956 = lean_box(0);
}
if (lean_is_scalar(x_956)) {
 x_957 = lean_alloc_ctor(1, 1, 0);
} else {
 x_957 = x_956;
}
lean_ctor_set(x_957, 0, x_955);
return x_957;
}
}
}
else
{
lean_object* x_958; lean_object* x_959; lean_object* x_960; 
lean_dec_ref(x_764);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_958 = lean_ctor_get(x_769, 0);
lean_inc(x_958);
if (lean_is_exclusive(x_769)) {
 lean_ctor_release(x_769, 0);
 x_959 = x_769;
} else {
 lean_dec_ref(x_769);
 x_959 = lean_box(0);
}
if (lean_is_scalar(x_959)) {
 x_960 = lean_alloc_ctor(1, 1, 0);
} else {
 x_960 = x_959;
}
lean_ctor_set(x_960, 0, x_958);
return x_960;
}
}
}
else
{
lean_object* x_961; lean_object* x_962; lean_object* x_963; lean_object* x_964; lean_object* x_965; lean_object* x_966; lean_object* x_967; lean_object* x_968; lean_object* x_969; lean_object* x_970; lean_object* x_971; lean_object* x_972; lean_object* x_973; lean_object* x_974; lean_object* x_975; lean_object* x_976; lean_object* x_977; lean_object* x_978; lean_object* x_979; lean_object* x_980; lean_object* x_981; lean_object* x_982; lean_object* x_983; lean_object* x_984; lean_object* x_985; lean_object* x_986; lean_object* x_987; lean_object* x_988; lean_object* x_989; lean_object* x_990; lean_object* x_991; lean_object* x_992; lean_object* x_993; lean_object* x_994; lean_object* x_995; lean_object* x_996; lean_object* x_997; lean_object* x_998; lean_object* x_999; lean_object* x_1000; lean_object* x_1001; lean_object* x_1002; 
x_961 = lean_ctor_get(x_13, 0);
lean_inc(x_961);
lean_dec(x_13);
x_962 = lean_ctor_get(x_961, 0);
lean_inc_ref(x_962);
x_963 = lean_ctor_get(x_961, 2);
lean_inc(x_963);
x_964 = lean_ctor_get(x_961, 3);
lean_inc(x_964);
x_965 = lean_ctor_get(x_961, 4);
lean_inc(x_965);
if (lean_is_exclusive(x_961)) {
 lean_ctor_release(x_961, 0);
 lean_ctor_release(x_961, 1);
 lean_ctor_release(x_961, 2);
 lean_ctor_release(x_961, 3);
 lean_ctor_release(x_961, 4);
 x_966 = x_961;
} else {
 lean_dec_ref(x_961);
 x_966 = lean_box(0);
}
x_967 = lp_aesop_Aesop_withNormTraceNode___closed__2;
x_968 = lp_aesop_Aesop_withNormTraceNode___closed__3;
lean_inc_ref(x_962);
x_969 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_969, 0, x_962);
x_970 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_970, 0, x_962);
x_971 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_971, 0, x_969);
lean_ctor_set(x_971, 1, x_970);
x_972 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_972, 0, x_965);
x_973 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_973, 0, x_964);
x_974 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_974, 0, x_963);
if (lean_is_scalar(x_966)) {
 x_975 = lean_alloc_ctor(0, 5, 0);
} else {
 x_975 = x_966;
}
lean_ctor_set(x_975, 0, x_971);
lean_ctor_set(x_975, 1, x_967);
lean_ctor_set(x_975, 2, x_974);
lean_ctor_set(x_975, 3, x_973);
lean_ctor_set(x_975, 4, x_972);
x_976 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_976, 0, x_975);
lean_ctor_set(x_976, 1, x_968);
x_977 = l_ReaderT_instMonad___redArg(x_976);
x_978 = lean_ctor_get(x_977, 0);
lean_inc_ref(x_978);
if (lean_is_exclusive(x_977)) {
 lean_ctor_release(x_977, 0);
 lean_ctor_release(x_977, 1);
 x_979 = x_977;
} else {
 lean_dec_ref(x_977);
 x_979 = lean_box(0);
}
x_980 = lean_ctor_get(x_978, 0);
lean_inc_ref(x_980);
x_981 = lean_ctor_get(x_978, 2);
lean_inc(x_981);
x_982 = lean_ctor_get(x_978, 3);
lean_inc(x_982);
x_983 = lean_ctor_get(x_978, 4);
lean_inc(x_983);
if (lean_is_exclusive(x_978)) {
 lean_ctor_release(x_978, 0);
 lean_ctor_release(x_978, 1);
 lean_ctor_release(x_978, 2);
 lean_ctor_release(x_978, 3);
 lean_ctor_release(x_978, 4);
 x_984 = x_978;
} else {
 lean_dec_ref(x_978);
 x_984 = lean_box(0);
}
x_985 = lp_aesop_Aesop_withNormTraceNode___closed__4;
x_986 = lp_aesop_Aesop_withNormTraceNode___closed__5;
lean_inc_ref(x_980);
x_987 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_987, 0, x_980);
x_988 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_988, 0, x_980);
x_989 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_989, 0, x_987);
lean_ctor_set(x_989, 1, x_988);
x_990 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_990, 0, x_983);
x_991 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_991, 0, x_982);
x_992 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_992, 0, x_981);
if (lean_is_scalar(x_984)) {
 x_993 = lean_alloc_ctor(0, 5, 0);
} else {
 x_993 = x_984;
}
lean_ctor_set(x_993, 0, x_989);
lean_ctor_set(x_993, 1, x_985);
lean_ctor_set(x_993, 2, x_992);
lean_ctor_set(x_993, 3, x_991);
lean_ctor_set(x_993, 4, x_990);
if (lean_is_scalar(x_979)) {
 x_994 = lean_alloc_ctor(0, 2, 0);
} else {
 x_994 = x_979;
}
lean_ctor_set(x_994, 0, x_993);
lean_ctor_set(x_994, 1, x_986);
x_995 = l_ReaderT_instMonad___redArg(x_994);
x_996 = l_ReaderT_instMonad___redArg(x_995);
x_997 = l_ReaderT_instMonad___redArg(x_996);
x_998 = lp_aesop_Aesop_withNormTraceNode___closed__20;
x_999 = lean_ctor_get(x_998, 0);
lean_inc(x_999);
x_1000 = lp_aesop_Aesop_runNormRuleTac___closed__8;
lean_inc_ref(x_997);
x_1001 = lp_aesop_Aesop_Check_isEnabled___redArg(x_997, x_999, x_1000);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_1002 = lean_apply_8(x_1001, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_1002) == 0)
{
lean_object* x_1003; uint8_t x_1004; 
x_1003 = lean_ctor_get(x_1002, 0);
lean_inc(x_1003);
lean_dec_ref(x_1002);
x_1004 = lean_unbox(x_1003);
lean_dec(x_1003);
if (x_1004 == 0)
{
lean_object* x_1005; 
lean_dec_ref(x_997);
lean_dec(x_3);
lean_dec_ref(x_1);
x_1005 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
return x_1005;
}
else
{
lean_object* x_1006; 
x_1006 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_1006) == 0)
{
lean_object* x_1007; lean_object* x_1008; 
x_1007 = lean_ctor_get(x_1006, 0);
lean_inc(x_1007);
lean_dec_ref(x_1006);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_1008 = lean_apply_8(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_1008) == 0)
{
lean_object* x_1009; lean_object* x_1010; lean_object* x_1011; lean_object* x_1012; lean_object* x_1013; lean_object* x_1014; lean_object* x_1015; lean_object* x_1016; lean_object* x_1017; lean_object* x_1018; lean_object* x_1019; lean_object* x_1020; lean_object* x_1021; lean_object* x_1022; lean_object* x_1023; lean_object* x_1065; lean_object* x_1066; lean_object* x_1067; lean_object* x_1068; lean_object* x_1069; lean_object* x_1070; lean_object* x_1071; lean_object* x_1072; lean_object* x_1073; lean_object* x_1074; lean_object* x_1075; lean_object* x_1104; lean_object* x_1105; lean_object* x_1106; lean_object* x_1107; lean_object* x_1108; lean_object* x_1109; lean_object* x_1110; lean_object* x_1111; lean_object* x_1112; lean_object* x_1113; lean_object* x_1114; lean_object* x_1115; lean_object* x_1130; lean_object* x_1131; lean_object* x_1132; lean_object* x_1133; lean_object* x_1134; lean_object* x_1163; 
x_1009 = lean_ctor_get(x_1008, 0);
lean_inc(x_1009);
if (lean_is_exclusive(x_1008)) {
 lean_ctor_release(x_1008, 0);
 x_1010 = x_1008;
} else {
 lean_dec_ref(x_1008);
 x_1010 = lean_box(0);
}
x_1011 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__0___boxed), 1, 0);
lean_inc(x_3);
x_1012 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__2___boxed), 3, 1);
lean_closure_set(x_1012, 0, x_3);
x_1013 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_1014 = lp_aesop_Aesop_withNormTraceNode___closed__7;
if (lean_obj_tag(x_1009) == 0)
{
lean_object* x_1185; 
x_1185 = lean_box(0);
x_1163 = x_1185;
goto block_1184;
}
else
{
lean_object* x_1186; lean_object* x_1187; 
x_1186 = lean_ctor_get(x_1009, 0);
x_1187 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_1186);
x_1163 = x_1187;
goto block_1184;
}
block_1064:
{
if (x_2 == 0)
{
if (lean_obj_tag(x_1015) == 0)
{
lean_object* x_1024; lean_object* x_1025; lean_object* x_1026; lean_object* x_1027; lean_object* x_1028; lean_object* x_1029; lean_object* x_1030; lean_object* x_1031; lean_object* x_1032; lean_object* x_1033; lean_object* x_1034; lean_object* x_1035; lean_object* x_1036; 
lean_dec(x_1010);
x_1024 = l_Lean_Meta_instMonadMCtxMetaM;
x_1025 = lean_ctor_get(x_1024, 0);
lean_inc(x_1025);
x_1026 = lean_ctor_get(x_1024, 1);
lean_inc(x_1026);
if (lean_is_exclusive(x_1024)) {
 lean_ctor_release(x_1024, 0);
 lean_ctor_release(x_1024, 1);
 x_1027 = x_1024;
} else {
 lean_dec_ref(x_1024);
 x_1027 = lean_box(0);
}
x_1028 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_1028, 0, x_1026);
lean_closure_set(x_1028, 1, x_1014);
x_1029 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_1029, 0, lean_box(0));
lean_closure_set(x_1029, 1, lean_box(0));
lean_closure_set(x_1029, 2, lean_box(0));
lean_closure_set(x_1029, 3, lean_box(0));
lean_closure_set(x_1029, 4, x_1025);
x_1030 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_1030, 0, x_1028);
lean_closure_set(x_1030, 1, x_1014);
x_1031 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_1031, 0, lean_box(0));
lean_closure_set(x_1031, 1, lean_box(0));
lean_closure_set(x_1031, 2, lean_box(0));
lean_closure_set(x_1031, 3, lean_box(0));
lean_closure_set(x_1031, 4, x_1029);
x_1032 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_1032, 0, x_1030);
lean_closure_set(x_1032, 1, x_1013);
x_1033 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_1033, 0, lean_box(0));
lean_closure_set(x_1033, 1, x_1031);
if (lean_is_scalar(x_1027)) {
 x_1034 = lean_alloc_ctor(0, 2, 0);
} else {
 x_1034 = x_1027;
}
lean_ctor_set(x_1034, 0, x_1033);
lean_ctor_set(x_1034, 1, x_1032);
lean_inc_ref(x_997);
x_1035 = l_Lean_MVarId_isAssigned___redArg(x_997, x_1034, x_3);
lean_inc(x_1022);
lean_inc_ref(x_1021);
lean_inc(x_1020);
lean_inc_ref(x_1019);
lean_inc(x_1018);
lean_inc(x_1017);
lean_inc_ref(x_1016);
x_1036 = lean_apply_8(x_1035, x_1016, x_1017, x_1018, x_1019, x_1020, x_1021, x_1022, lean_box(0));
if (lean_obj_tag(x_1036) == 0)
{
lean_object* x_1037; lean_object* x_1038; uint8_t x_1039; 
x_1037 = lean_ctor_get(x_1036, 0);
lean_inc(x_1037);
if (lean_is_exclusive(x_1036)) {
 lean_ctor_release(x_1036, 0);
 x_1038 = x_1036;
} else {
 lean_dec_ref(x_1036);
 x_1038 = lean_box(0);
}
x_1039 = lean_unbox(x_1037);
lean_dec(x_1037);
if (x_1039 == 0)
{
lean_object* x_1040; 
lean_dec(x_1022);
lean_dec_ref(x_1021);
lean_dec(x_1020);
lean_dec_ref(x_1019);
lean_dec(x_1018);
lean_dec(x_1017);
lean_dec_ref(x_1016);
lean_dec_ref(x_997);
lean_dec_ref(x_1);
if (lean_is_scalar(x_1038)) {
 x_1040 = lean_alloc_ctor(0, 1, 0);
} else {
 x_1040 = x_1038;
}
lean_ctor_set(x_1040, 0, x_1009);
return x_1040;
}
else
{
lean_object* x_1041; lean_object* x_1042; lean_object* x_1043; lean_object* x_1044; lean_object* x_1045; lean_object* x_1046; lean_object* x_1047; lean_object* x_1048; lean_object* x_1049; lean_object* x_1050; lean_object* x_1051; lean_object* x_1052; lean_object* x_1053; 
lean_dec(x_1038);
x_1041 = lp_aesop_Aesop_checkSimp___closed__14;
x_1042 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_1043 = lean_ctor_get(x_1042, 0);
lean_inc_ref(x_1043);
x_1044 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_997);
x_1045 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_1044, x_997);
x_1046 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_1046, 0, x_1041);
lean_ctor_set(x_1046, 1, x_1043);
lean_ctor_set(x_1046, 2, x_1045);
x_1047 = lp_aesop_Aesop_checkSimp___closed__17;
x_1048 = l_Lean_stringToMessageData(x_1);
x_1049 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1049, 0, x_1047);
lean_ctor_set(x_1049, 1, x_1048);
x_1050 = lp_aesop_Aesop_checkSimp___closed__19;
x_1051 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1051, 0, x_1049);
lean_ctor_set(x_1051, 1, x_1050);
x_1052 = l_Lean_throwError___redArg(x_997, x_1046, x_1051);
x_1053 = lean_apply_8(x_1052, x_1016, x_1017, x_1018, x_1019, x_1020, x_1021, x_1022, lean_box(0));
if (lean_obj_tag(x_1053) == 0)
{
lean_object* x_1054; lean_object* x_1055; 
if (lean_is_exclusive(x_1053)) {
 lean_ctor_release(x_1053, 0);
 x_1054 = x_1053;
} else {
 lean_dec_ref(x_1053);
 x_1054 = lean_box(0);
}
if (lean_is_scalar(x_1054)) {
 x_1055 = lean_alloc_ctor(0, 1, 0);
} else {
 x_1055 = x_1054;
}
lean_ctor_set(x_1055, 0, x_1009);
return x_1055;
}
else
{
lean_object* x_1056; lean_object* x_1057; lean_object* x_1058; 
lean_dec(x_1009);
x_1056 = lean_ctor_get(x_1053, 0);
lean_inc(x_1056);
if (lean_is_exclusive(x_1053)) {
 lean_ctor_release(x_1053, 0);
 x_1057 = x_1053;
} else {
 lean_dec_ref(x_1053);
 x_1057 = lean_box(0);
}
if (lean_is_scalar(x_1057)) {
 x_1058 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1058 = x_1057;
}
lean_ctor_set(x_1058, 0, x_1056);
return x_1058;
}
}
}
else
{
lean_object* x_1059; lean_object* x_1060; lean_object* x_1061; 
lean_dec(x_1022);
lean_dec_ref(x_1021);
lean_dec(x_1020);
lean_dec_ref(x_1019);
lean_dec(x_1018);
lean_dec(x_1017);
lean_dec_ref(x_1016);
lean_dec(x_1009);
lean_dec_ref(x_997);
lean_dec_ref(x_1);
x_1059 = lean_ctor_get(x_1036, 0);
lean_inc(x_1059);
if (lean_is_exclusive(x_1036)) {
 lean_ctor_release(x_1036, 0);
 x_1060 = x_1036;
} else {
 lean_dec_ref(x_1036);
 x_1060 = lean_box(0);
}
if (lean_is_scalar(x_1060)) {
 x_1061 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1061 = x_1060;
}
lean_ctor_set(x_1061, 0, x_1059);
return x_1061;
}
}
else
{
lean_object* x_1062; 
lean_dec(x_1022);
lean_dec_ref(x_1021);
lean_dec(x_1020);
lean_dec_ref(x_1019);
lean_dec(x_1018);
lean_dec(x_1017);
lean_dec_ref(x_1016);
lean_dec_ref(x_1015);
lean_dec_ref(x_997);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_1010)) {
 x_1062 = lean_alloc_ctor(0, 1, 0);
} else {
 x_1062 = x_1010;
}
lean_ctor_set(x_1062, 0, x_1009);
return x_1062;
}
}
else
{
lean_object* x_1063; 
lean_dec(x_1022);
lean_dec_ref(x_1021);
lean_dec(x_1020);
lean_dec_ref(x_1019);
lean_dec(x_1018);
lean_dec(x_1017);
lean_dec_ref(x_1016);
lean_dec(x_1015);
lean_dec_ref(x_997);
lean_dec(x_3);
lean_dec_ref(x_1);
if (lean_is_scalar(x_1010)) {
 x_1063 = lean_alloc_ctor(0, 1, 0);
} else {
 x_1063 = x_1010;
}
lean_ctor_set(x_1063, 0, x_1009);
return x_1063;
}
}
block_1103:
{
uint8_t x_1076; 
x_1076 = l_Array_isEmpty___redArg(x_1075);
lean_dec_ref(x_1075);
if (x_1076 == 0)
{
lean_object* x_1077; lean_object* x_1078; lean_object* x_1079; lean_object* x_1080; lean_object* x_1081; lean_object* x_1082; lean_object* x_1083; lean_object* x_1084; lean_object* x_1085; lean_object* x_1086; lean_object* x_1087; lean_object* x_1088; size_t x_1089; size_t x_1090; lean_object* x_1091; lean_object* x_1092; lean_object* x_1093; lean_object* x_1094; lean_object* x_1095; lean_object* x_1096; lean_object* x_1097; lean_object* x_1098; lean_object* x_1099; 
x_1077 = lp_aesop_Aesop_checkSimp___closed__14;
x_1078 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_1079 = lean_ctor_get(x_1078, 0);
lean_inc_ref(x_1079);
x_1080 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_997);
x_1081 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_1080, x_997);
x_1082 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_1082, 0, x_1077);
lean_ctor_set(x_1082, 1, x_1079);
lean_ctor_set(x_1082, 2, x_1081);
x_1083 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_1084 = l_Lean_stringToMessageData(x_1);
x_1085 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1085, 0, x_1083);
lean_ctor_set(x_1085, 1, x_1084);
x_1086 = lp_aesop_Aesop_checkSimp___closed__21;
x_1087 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1087, 0, x_1085);
lean_ctor_set(x_1087, 1, x_1086);
x_1088 = lp_aesop_Aesop_checkSimp___closed__31;
x_1089 = lean_array_size(x_1071);
x_1090 = 0;
x_1091 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_1088, x_1011, x_1089, x_1090, x_1071);
x_1092 = lean_array_to_list(x_1091);
x_1093 = lp_aesop_Aesop_checkSimp___closed__32;
x_1094 = lean_box(0);
x_1095 = l_List_mapTR_loop___redArg(x_1093, x_1092, x_1094);
x_1096 = l_Lean_MessageData_ofList(x_1095);
x_1097 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1097, 0, x_1087);
lean_ctor_set(x_1097, 1, x_1096);
lean_inc_ref(x_997);
x_1098 = l_Lean_throwError___redArg(x_997, x_1082, x_1097);
lean_inc(x_1066);
lean_inc_ref(x_1067);
lean_inc(x_1068);
lean_inc_ref(x_1072);
lean_inc(x_1069);
lean_inc(x_1065);
lean_inc_ref(x_1073);
x_1099 = lean_apply_8(x_1098, x_1073, x_1065, x_1069, x_1072, x_1068, x_1067, x_1066, lean_box(0));
if (lean_obj_tag(x_1099) == 0)
{
lean_dec_ref(x_1099);
x_1015 = x_1074;
x_1016 = x_1073;
x_1017 = x_1065;
x_1018 = x_1069;
x_1019 = x_1072;
x_1020 = x_1068;
x_1021 = x_1067;
x_1022 = x_1066;
x_1023 = lean_box(0);
goto block_1064;
}
else
{
lean_object* x_1100; lean_object* x_1101; lean_object* x_1102; 
lean_dec(x_1074);
lean_dec_ref(x_1073);
lean_dec_ref(x_1072);
lean_dec(x_1069);
lean_dec(x_1068);
lean_dec_ref(x_1067);
lean_dec(x_1066);
lean_dec(x_1065);
lean_dec(x_1010);
lean_dec(x_1009);
lean_dec_ref(x_997);
lean_dec(x_3);
lean_dec_ref(x_1);
x_1100 = lean_ctor_get(x_1099, 0);
lean_inc(x_1100);
if (lean_is_exclusive(x_1099)) {
 lean_ctor_release(x_1099, 0);
 x_1101 = x_1099;
} else {
 lean_dec_ref(x_1099);
 x_1101 = lean_box(0);
}
if (lean_is_scalar(x_1101)) {
 x_1102 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1102 = x_1101;
}
lean_ctor_set(x_1102, 0, x_1100);
return x_1102;
}
}
else
{
lean_dec_ref(x_1071);
lean_dec_ref(x_1011);
x_1015 = x_1074;
x_1016 = x_1073;
x_1017 = x_1065;
x_1018 = x_1069;
x_1019 = x_1072;
x_1020 = x_1068;
x_1021 = x_1067;
x_1022 = x_1066;
x_1023 = lean_box(0);
goto block_1064;
}
}
block_1129:
{
lean_object* x_1116; 
lean_inc(x_1114);
lean_inc_ref(x_1113);
lean_inc(x_1112);
lean_inc_ref(x_1111);
x_1116 = lp_batteries_Lean_Meta_getAssignedExprMVars(x_1007, x_1105, x_1111, x_1112, x_1113, x_1114);
if (lean_obj_tag(x_1116) == 0)
{
lean_object* x_1117; lean_object* x_1118; lean_object* x_1119; lean_object* x_1120; uint8_t x_1121; 
x_1117 = lean_ctor_get(x_1116, 0);
lean_inc(x_1117);
lean_dec_ref(x_1116);
x_1118 = lean_array_get_size(x_1117);
x_1119 = lean_mk_empty_array_with_capacity(x_1104);
x_1120 = lp_aesop_Aesop_checkSimp___closed__31;
x_1121 = lean_nat_dec_lt(x_1104, x_1118);
if (x_1121 == 0)
{
lean_dec(x_1117);
lean_dec_ref(x_1012);
x_1065 = x_1109;
x_1066 = x_1114;
x_1067 = x_1113;
x_1068 = x_1112;
x_1069 = x_1110;
x_1070 = lean_box(0);
x_1071 = x_1106;
x_1072 = x_1111;
x_1073 = x_1108;
x_1074 = x_1107;
x_1075 = x_1119;
goto block_1103;
}
else
{
uint8_t x_1122; 
x_1122 = lean_nat_dec_le(x_1118, x_1118);
if (x_1122 == 0)
{
lean_dec(x_1117);
lean_dec_ref(x_1012);
x_1065 = x_1109;
x_1066 = x_1114;
x_1067 = x_1113;
x_1068 = x_1112;
x_1069 = x_1110;
x_1070 = lean_box(0);
x_1071 = x_1106;
x_1072 = x_1111;
x_1073 = x_1108;
x_1074 = x_1107;
x_1075 = x_1119;
goto block_1103;
}
else
{
size_t x_1123; size_t x_1124; lean_object* x_1125; 
x_1123 = 0;
x_1124 = lean_usize_of_nat(x_1118);
x_1125 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_1120, x_1012, x_1117, x_1123, x_1124, x_1119);
x_1065 = x_1109;
x_1066 = x_1114;
x_1067 = x_1113;
x_1068 = x_1112;
x_1069 = x_1110;
x_1070 = lean_box(0);
x_1071 = x_1106;
x_1072 = x_1111;
x_1073 = x_1108;
x_1074 = x_1107;
x_1075 = x_1125;
goto block_1103;
}
}
}
else
{
lean_object* x_1126; lean_object* x_1127; lean_object* x_1128; 
lean_dec(x_1114);
lean_dec_ref(x_1113);
lean_dec(x_1112);
lean_dec_ref(x_1111);
lean_dec(x_1110);
lean_dec(x_1109);
lean_dec_ref(x_1108);
lean_dec(x_1107);
lean_dec_ref(x_1106);
lean_dec_ref(x_1012);
lean_dec_ref(x_1011);
lean_dec(x_1010);
lean_dec(x_1009);
lean_dec_ref(x_997);
lean_dec(x_3);
lean_dec_ref(x_1);
x_1126 = lean_ctor_get(x_1116, 0);
lean_inc(x_1126);
if (lean_is_exclusive(x_1116)) {
 lean_ctor_release(x_1116, 0);
 x_1127 = x_1116;
} else {
 lean_dec_ref(x_1116);
 x_1127 = lean_box(0);
}
if (lean_is_scalar(x_1127)) {
 x_1128 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1128 = x_1127;
}
lean_ctor_set(x_1128, 0, x_1126);
return x_1128;
}
}
block_1162:
{
uint8_t x_1135; 
x_1135 = l_Array_isEmpty___redArg(x_1134);
if (x_1135 == 0)
{
lean_object* x_1136; lean_object* x_1137; lean_object* x_1138; lean_object* x_1139; lean_object* x_1140; lean_object* x_1141; lean_object* x_1142; lean_object* x_1143; lean_object* x_1144; lean_object* x_1145; lean_object* x_1146; lean_object* x_1147; size_t x_1148; size_t x_1149; lean_object* x_1150; lean_object* x_1151; lean_object* x_1152; lean_object* x_1153; lean_object* x_1154; lean_object* x_1155; lean_object* x_1156; lean_object* x_1157; lean_object* x_1158; 
x_1136 = lp_aesop_Aesop_checkSimp___closed__14;
x_1137 = lp_aesop_Aesop_withNormTraceNode___closed__18;
x_1138 = lean_ctor_get(x_1137, 0);
lean_inc_ref(x_1138);
x_1139 = lp_aesop_Aesop_withNormTraceNode___closed__42;
lean_inc_ref(x_997);
x_1140 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_1139, x_997);
x_1141 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_1141, 0, x_1136);
lean_ctor_set(x_1141, 1, x_1138);
lean_ctor_set(x_1141, 2, x_1140);
x_1142 = lp_aesop_Aesop_checkSimp___closed__17;
lean_inc_ref(x_1);
x_1143 = l_Lean_stringToMessageData(x_1);
x_1144 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1144, 0, x_1142);
lean_ctor_set(x_1144, 1, x_1143);
x_1145 = lp_aesop_Aesop_checkSimp___closed__34;
x_1146 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1146, 0, x_1144);
lean_ctor_set(x_1146, 1, x_1145);
x_1147 = lp_aesop_Aesop_checkSimp___closed__31;
x_1148 = lean_array_size(x_1134);
x_1149 = 0;
lean_inc_ref(x_1134);
lean_inc_ref(x_1011);
x_1150 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_1147, x_1011, x_1148, x_1149, x_1134);
x_1151 = lean_array_to_list(x_1150);
x_1152 = lp_aesop_Aesop_checkSimp___closed__32;
x_1153 = lean_box(0);
x_1154 = l_List_mapTR_loop___redArg(x_1152, x_1151, x_1153);
x_1155 = l_Lean_MessageData_ofList(x_1154);
x_1156 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1156, 0, x_1146);
lean_ctor_set(x_1156, 1, x_1155);
lean_inc_ref(x_997);
x_1157 = l_Lean_throwError___redArg(x_997, x_1141, x_1156);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_1158 = lean_apply_8(x_1157, x_5, x_6, x_7, x_8, x_9, x_10, x_11, lean_box(0));
if (lean_obj_tag(x_1158) == 0)
{
lean_dec_ref(x_1158);
x_1104 = x_1130;
x_1105 = x_1131;
x_1106 = x_1134;
x_1107 = x_1133;
x_1108 = x_5;
x_1109 = x_6;
x_1110 = x_7;
x_1111 = x_8;
x_1112 = x_9;
x_1113 = x_10;
x_1114 = x_11;
x_1115 = lean_box(0);
goto block_1129;
}
else
{
lean_object* x_1159; lean_object* x_1160; lean_object* x_1161; 
lean_dec_ref(x_1134);
lean_dec(x_1133);
lean_dec_ref(x_1131);
lean_dec_ref(x_1012);
lean_dec_ref(x_1011);
lean_dec(x_1010);
lean_dec(x_1009);
lean_dec(x_1007);
lean_dec_ref(x_997);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_1159 = lean_ctor_get(x_1158, 0);
lean_inc(x_1159);
if (lean_is_exclusive(x_1158)) {
 lean_ctor_release(x_1158, 0);
 x_1160 = x_1158;
} else {
 lean_dec_ref(x_1158);
 x_1160 = lean_box(0);
}
if (lean_is_scalar(x_1160)) {
 x_1161 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1161 = x_1160;
}
lean_ctor_set(x_1161, 0, x_1159);
return x_1161;
}
}
else
{
x_1104 = x_1130;
x_1105 = x_1131;
x_1106 = x_1134;
x_1107 = x_1133;
x_1108 = x_5;
x_1109 = x_6;
x_1110 = x_7;
x_1111 = x_8;
x_1112 = x_9;
x_1113 = x_10;
x_1114 = x_11;
x_1115 = lean_box(0);
goto block_1129;
}
}
block_1184:
{
lean_object* x_1164; 
x_1164 = l_Lean_Meta_saveState___redArg(x_9, x_11);
if (lean_obj_tag(x_1164) == 0)
{
lean_object* x_1165; lean_object* x_1166; 
x_1165 = lean_ctor_get(x_1164, 0);
lean_inc(x_1165);
lean_dec_ref(x_1164);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_1165);
lean_inc(x_1007);
x_1166 = lp_batteries_Lean_Meta_getIntroducedExprMVars(x_1007, x_1165, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_1166) == 0)
{
lean_object* x_1167; lean_object* x_1168; lean_object* x_1169; lean_object* x_1170; lean_object* x_1171; uint8_t x_1172; 
x_1167 = lean_ctor_get(x_1166, 0);
lean_inc(x_1167);
lean_dec_ref(x_1166);
x_1168 = lean_unsigned_to_nat(0u);
x_1169 = lean_array_get_size(x_1167);
x_1170 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_1171 = lp_aesop_Aesop_checkSimp___closed__31;
x_1172 = lean_nat_dec_lt(x_1168, x_1169);
if (x_1172 == 0)
{
lean_dec(x_1167);
x_1130 = x_1168;
x_1131 = x_1165;
x_1132 = lean_box(0);
x_1133 = x_1163;
x_1134 = x_1170;
goto block_1162;
}
else
{
uint8_t x_1173; 
x_1173 = lean_nat_dec_le(x_1169, x_1169);
if (x_1173 == 0)
{
lean_dec(x_1167);
x_1130 = x_1168;
x_1131 = x_1165;
x_1132 = lean_box(0);
x_1133 = x_1163;
x_1134 = x_1170;
goto block_1162;
}
else
{
lean_object* x_1174; size_t x_1175; size_t x_1176; lean_object* x_1177; 
lean_inc(x_1163);
x_1174 = lean_alloc_closure((void*)(lp_aesop_Aesop_checkSimp___lam__1), 3, 1);
lean_closure_set(x_1174, 0, x_1163);
x_1175 = 0;
x_1176 = lean_usize_of_nat(x_1169);
x_1177 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_1171, x_1174, x_1167, x_1175, x_1176, x_1170);
x_1130 = x_1168;
x_1131 = x_1165;
x_1132 = lean_box(0);
x_1133 = x_1163;
x_1134 = x_1177;
goto block_1162;
}
}
}
else
{
lean_object* x_1178; lean_object* x_1179; lean_object* x_1180; 
lean_dec(x_1165);
lean_dec(x_1163);
lean_dec_ref(x_1012);
lean_dec_ref(x_1011);
lean_dec(x_1010);
lean_dec(x_1009);
lean_dec(x_1007);
lean_dec_ref(x_997);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_1178 = lean_ctor_get(x_1166, 0);
lean_inc(x_1178);
if (lean_is_exclusive(x_1166)) {
 lean_ctor_release(x_1166, 0);
 x_1179 = x_1166;
} else {
 lean_dec_ref(x_1166);
 x_1179 = lean_box(0);
}
if (lean_is_scalar(x_1179)) {
 x_1180 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1180 = x_1179;
}
lean_ctor_set(x_1180, 0, x_1178);
return x_1180;
}
}
else
{
lean_object* x_1181; lean_object* x_1182; lean_object* x_1183; 
lean_dec(x_1163);
lean_dec_ref(x_1012);
lean_dec_ref(x_1011);
lean_dec(x_1010);
lean_dec(x_1009);
lean_dec(x_1007);
lean_dec_ref(x_997);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_1181 = lean_ctor_get(x_1164, 0);
lean_inc(x_1181);
if (lean_is_exclusive(x_1164)) {
 lean_ctor_release(x_1164, 0);
 x_1182 = x_1164;
} else {
 lean_dec_ref(x_1164);
 x_1182 = lean_box(0);
}
if (lean_is_scalar(x_1182)) {
 x_1183 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1183 = x_1182;
}
lean_ctor_set(x_1183, 0, x_1181);
return x_1183;
}
}
}
else
{
lean_dec(x_1007);
lean_dec_ref(x_997);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_1008;
}
}
else
{
lean_object* x_1188; lean_object* x_1189; lean_object* x_1190; 
lean_dec_ref(x_997);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_1188 = lean_ctor_get(x_1006, 0);
lean_inc(x_1188);
if (lean_is_exclusive(x_1006)) {
 lean_ctor_release(x_1006, 0);
 x_1189 = x_1006;
} else {
 lean_dec_ref(x_1006);
 x_1189 = lean_box(0);
}
if (lean_is_scalar(x_1189)) {
 x_1190 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1190 = x_1189;
}
lean_ctor_set(x_1190, 0, x_1188);
return x_1190;
}
}
}
else
{
lean_object* x_1191; lean_object* x_1192; lean_object* x_1193; 
lean_dec_ref(x_997);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_1191 = lean_ctor_get(x_1002, 0);
lean_inc(x_1191);
if (lean_is_exclusive(x_1002)) {
 lean_ctor_release(x_1002, 0);
 x_1192 = x_1002;
} else {
 lean_dec_ref(x_1002);
 x_1192 = lean_box(0);
}
if (lean_is_scalar(x_1192)) {
 x_1193 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1193 = x_1192;
}
lean_ctor_set(x_1193, 0, x_1191);
return x_1193;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkSimp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
uint8_t x_13; lean_object* x_14; 
x_13 = lean_unbox(x_2);
x_14 = lp_aesop_Aesop_checkSimp(x_1, x_13, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_ctor_get(x_4, 5);
x_8 = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_7);
lean_ctor_set(x_11, 1, x_10);
lean_ctor_set_tag(x_8, 1);
lean_ctor_set(x_8, 0, x_11);
return x_8;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_8, 0);
lean_inc(x_12);
lean_dec(x_8);
lean_inc(x_7);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_7);
lean_ctor_set(x_13, 1, x_12);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_2, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = l_Lean_MessageData_ofName(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = l_Lean_MessageData_ofName(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(size_t x_1, size_t x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_lt(x_2, x_1);
if (x_4 == 0)
{
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; size_t x_8; size_t x_9; lean_object* x_10; 
x_5 = lean_array_uget(x_3, x_2);
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_array_uset(x_3, x_2, x_6);
x_8 = 1;
x_9 = lean_usize_add(x_2, x_8);
x_10 = lean_array_uset(x_7, x_2, x_5);
x_2 = x_9;
x_3 = x_10;
goto _start;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Option_instBEq_beq___at___00Aesop_normSimp_spec__1(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_3; 
x_3 = 1;
return x_3;
}
else
{
uint8_t x_4; 
x_4 = 0;
return x_4;
}
}
else
{
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_5; 
x_5 = 0;
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_2, 0);
x_8 = l_Lean_instBEqMVarId_beq(x_6, x_7);
return x_8;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__4(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_11; 
x_11 = lean_usize_dec_eq(x_3, x_4);
if (x_11 == 0)
{
lean_object* x_12; uint8_t x_13; 
x_12 = lean_array_uget(x_2, x_3);
x_13 = l_Lean_instBEqMVarId_beq(x_12, x_1);
if (x_13 == 0)
{
lean_object* x_14; 
x_14 = lean_array_push(x_5, x_12);
x_6 = x_14;
goto block_10;
}
else
{
lean_dec(x_12);
x_6 = x_5;
goto block_10;
}
}
else
{
return x_5;
}
block_10:
{
size_t x_7; size_t x_8; 
x_7 = 1;
x_8 = lean_usize_add(x_3, x_7);
x_3 = x_8;
x_5 = x_6;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__5(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_11; 
x_11 = lean_usize_dec_eq(x_3, x_4);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_12 = lean_array_uget(x_2, x_3);
lean_inc(x_12);
x_13 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_13, 0, x_12);
x_14 = lp_aesop_Option_instBEq_beq___at___00Aesop_normSimp_spec__1(x_13, x_1);
lean_dec_ref(x_13);
if (x_14 == 0)
{
lean_object* x_15; 
x_15 = lean_array_push(x_5, x_12);
x_6 = x_15;
goto block_10;
}
else
{
lean_dec(x_12);
x_6 = x_5;
goto block_10;
}
}
else
{
return x_5;
}
block_10:
{
size_t x_7; size_t x_8; 
x_7 = 1;
x_8 = lean_usize_add(x_3, x_7);
x_3 = x_8;
x_5 = x_6;
goto _start;
}
}
}
static lean_object* _init_lp_aesop_Aesop_normSimp___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: error in norm simp: ", 27, 27);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimp___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normSimp___lam__0___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_11 = lp_aesop_Aesop_normSimpCore(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_11;
}
else
{
lean_object* x_12; uint8_t x_13; uint8_t x_19; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
x_19 = l_Lean_Exception_isInterrupt(x_12);
if (x_19 == 0)
{
uint8_t x_20; 
lean_inc(x_12);
x_20 = l_Lean_Exception_isRuntime(x_12);
x_13 = x_20;
goto block_18;
}
else
{
x_13 = x_19;
goto block_18;
}
block_18:
{
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_dec_ref(x_11);
x_14 = lp_aesop_Aesop_normSimp___lam__0___closed__1;
x_15 = l_Lean_Exception_toMessageData(x_12);
x_16 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
x_17 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_16, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_17;
}
else
{
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_11;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_11 = lean_apply_8(x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
if (lean_obj_tag(x_12) == 0)
{
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_2);
return x_11;
}
else
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_ctor_get(x_12, 0);
x_14 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_13);
if (lean_obj_tag(x_14) == 1)
{
uint8_t x_15; 
lean_dec_ref(x_11);
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; uint8_t x_18; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_2, x_8);
x_18 = !lean_is_exclusive(x_17);
if (x_18 == 0)
{
lean_object* x_19; uint8_t x_20; 
x_19 = lean_ctor_get(x_17, 0);
x_20 = lean_unbox(x_19);
lean_dec(x_19);
if (x_20 == 0)
{
lean_free_object(x_14);
lean_dec(x_16);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_2);
lean_ctor_set(x_17, 0, x_12);
return x_17;
}
else
{
lean_object* x_21; lean_object* x_22; 
lean_free_object(x_17);
x_21 = lean_ctor_get(x_2, 0);
lean_inc(x_21);
lean_dec_ref(x_2);
x_22 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_21, x_14, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
if (lean_obj_tag(x_22) == 0)
{
uint8_t x_23; 
x_23 = !lean_is_exclusive(x_22);
if (x_23 == 0)
{
lean_object* x_24; 
x_24 = lean_ctor_get(x_22, 0);
lean_dec(x_24);
lean_ctor_set(x_22, 0, x_12);
return x_22;
}
else
{
lean_object* x_25; 
lean_dec(x_22);
x_25 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_25, 0, x_12);
return x_25;
}
}
else
{
uint8_t x_26; 
lean_dec_ref(x_12);
x_26 = !lean_is_exclusive(x_22);
if (x_26 == 0)
{
return x_22;
}
else
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_22, 0);
lean_inc(x_27);
lean_dec(x_22);
x_28 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_28, 0, x_27);
return x_28;
}
}
}
}
else
{
lean_object* x_29; uint8_t x_30; 
x_29 = lean_ctor_get(x_17, 0);
lean_inc(x_29);
lean_dec(x_17);
x_30 = lean_unbox(x_29);
lean_dec(x_29);
if (x_30 == 0)
{
lean_object* x_31; 
lean_free_object(x_14);
lean_dec(x_16);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_2);
x_31 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_31, 0, x_12);
return x_31;
}
else
{
lean_object* x_32; lean_object* x_33; 
x_32 = lean_ctor_get(x_2, 0);
lean_inc(x_32);
lean_dec_ref(x_2);
x_33 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_32, x_14, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
if (lean_obj_tag(x_33) == 0)
{
lean_object* x_34; lean_object* x_35; 
if (lean_is_exclusive(x_33)) {
 lean_ctor_release(x_33, 0);
 x_34 = x_33;
} else {
 lean_dec_ref(x_33);
 x_34 = lean_box(0);
}
if (lean_is_scalar(x_34)) {
 x_35 = lean_alloc_ctor(0, 1, 0);
} else {
 x_35 = x_34;
}
lean_ctor_set(x_35, 0, x_12);
return x_35;
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; 
lean_dec_ref(x_12);
x_36 = lean_ctor_get(x_33, 0);
lean_inc(x_36);
if (lean_is_exclusive(x_33)) {
 lean_ctor_release(x_33, 0);
 x_37 = x_33;
} else {
 lean_dec_ref(x_33);
 x_37 = lean_box(0);
}
if (lean_is_scalar(x_37)) {
 x_38 = lean_alloc_ctor(1, 1, 0);
} else {
 x_38 = x_37;
}
lean_ctor_set(x_38, 0, x_36);
return x_38;
}
}
}
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; uint8_t x_43; 
x_39 = lean_ctor_get(x_14, 0);
lean_inc(x_39);
lean_dec(x_14);
x_40 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_2, x_8);
x_41 = lean_ctor_get(x_40, 0);
lean_inc(x_41);
if (lean_is_exclusive(x_40)) {
 lean_ctor_release(x_40, 0);
 x_42 = x_40;
} else {
 lean_dec_ref(x_40);
 x_42 = lean_box(0);
}
x_43 = lean_unbox(x_41);
lean_dec(x_41);
if (x_43 == 0)
{
lean_object* x_44; 
lean_dec(x_39);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_2);
if (lean_is_scalar(x_42)) {
 x_44 = lean_alloc_ctor(0, 1, 0);
} else {
 x_44 = x_42;
}
lean_ctor_set(x_44, 0, x_12);
return x_44;
}
else
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; 
lean_dec(x_42);
x_45 = lean_ctor_get(x_2, 0);
lean_inc(x_45);
lean_dec_ref(x_2);
x_46 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_46, 0, x_39);
x_47 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_45, x_46, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
if (lean_obj_tag(x_47) == 0)
{
lean_object* x_48; lean_object* x_49; 
if (lean_is_exclusive(x_47)) {
 lean_ctor_release(x_47, 0);
 x_48 = x_47;
} else {
 lean_dec_ref(x_47);
 x_48 = lean_box(0);
}
if (lean_is_scalar(x_48)) {
 x_49 = lean_alloc_ctor(0, 1, 0);
} else {
 x_49 = x_48;
}
lean_ctor_set(x_49, 0, x_12);
return x_49;
}
else
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; 
lean_dec_ref(x_12);
x_50 = lean_ctor_get(x_47, 0);
lean_inc(x_50);
if (lean_is_exclusive(x_47)) {
 lean_ctor_release(x_47, 0);
 x_51 = x_47;
} else {
 lean_dec_ref(x_47);
 x_51 = lean_box(0);
}
if (lean_is_scalar(x_51)) {
 x_52 = lean_alloc_ctor(1, 1, 0);
} else {
 x_52 = x_51;
}
lean_ctor_set(x_52, 0, x_50);
return x_52;
}
}
}
}
else
{
lean_dec(x_14);
lean_dec_ref(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_2);
return x_11;
}
}
}
else
{
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_2);
return x_11;
}
}
}
static lean_object* _init_lp_aesop_Aesop_normSimp___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("norm simp", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimp___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normSimp___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normSimp___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(1);
x_2 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed), 10, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_normSimp___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_normSimp___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; uint8_t x_66; lean_object* x_67; lean_object* x_68; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_186; lean_object* x_215; uint8_t x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; uint8_t x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; uint8_t x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_311; uint8_t x_312; lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; uint8_t x_337; lean_object* x_338; lean_object* x_367; lean_object* x_372; uint8_t x_373; 
x_11 = lean_ctor_get(x_8, 2);
lean_inc(x_1);
x_12 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimp___lam__0___boxed), 10, 2);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_2);
x_13 = lean_box(1);
x_372 = lp_aesop_Aesop_runNormRule___closed__0;
x_373 = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(x_11, x_372);
if (x_373 == 0)
{
lean_object* x_374; lean_object* x_375; lean_object* x_376; uint8_t x_377; 
x_374 = lp_aesop_Aesop_runNormRule___closed__1;
x_375 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_374, x_8);
x_376 = lean_ctor_get(x_375, 0);
lean_inc(x_376);
x_377 = lean_unbox(x_376);
lean_dec(x_376);
if (x_377 == 0)
{
lean_object* x_378; lean_object* x_379; lean_object* x_380; uint8_t x_381; 
lean_dec_ref(x_375);
x_378 = lp_aesop_Aesop_runNormRule___closed__2;
x_379 = lp_aesop_Lean_Option_get___at___00Aesop_runNormRule_spec__8(x_11, x_378);
x_380 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_381 = lean_string_dec_eq(x_379, x_380);
lean_dec_ref(x_379);
if (x_381 == 0)
{
uint8_t x_382; 
x_382 = 1;
x_337 = x_382;
x_338 = lean_box(0);
goto block_366;
}
else
{
x_186 = lean_box(0);
goto block_214;
}
}
else
{
x_367 = x_375;
goto block_371;
}
}
else
{
x_337 = x_373;
x_338 = lean_box(0);
goto block_366;
}
block_65:
{
lean_object* x_18; lean_object* x_19; uint8_t x_20; 
x_18 = lean_io_mono_nanos_now();
x_19 = lean_st_ref_take(x_5);
x_20 = !lean_is_exclusive(x_19);
if (x_20 == 0)
{
lean_object* x_21; uint8_t x_22; 
x_21 = lean_ctor_get(x_19, 1);
x_22 = !lean_is_exclusive(x_21);
if (x_22 == 0)
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_23 = lean_ctor_get(x_21, 8);
x_24 = lean_nat_sub(x_18, x_15);
lean_dec(x_15);
lean_dec(x_18);
x_25 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_25, 0, x_13);
lean_ctor_set(x_25, 1, x_24);
lean_ctor_set_uint8(x_25, sizeof(void*)*2, x_14);
x_26 = lean_array_push(x_23, x_25);
lean_ctor_set(x_21, 8, x_26);
x_27 = lean_st_ref_set(x_5, x_19);
lean_dec(x_5);
x_28 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_28, 0, x_16);
return x_28;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_29 = lean_ctor_get(x_21, 0);
x_30 = lean_ctor_get(x_21, 1);
x_31 = lean_ctor_get(x_21, 2);
x_32 = lean_ctor_get(x_21, 3);
x_33 = lean_ctor_get(x_21, 4);
x_34 = lean_ctor_get(x_21, 5);
x_35 = lean_ctor_get(x_21, 6);
x_36 = lean_ctor_get(x_21, 7);
x_37 = lean_ctor_get(x_21, 8);
x_38 = lean_ctor_get(x_21, 9);
lean_inc(x_38);
lean_inc(x_37);
lean_inc(x_36);
lean_inc(x_35);
lean_inc(x_34);
lean_inc(x_33);
lean_inc(x_32);
lean_inc(x_31);
lean_inc(x_30);
lean_inc(x_29);
lean_dec(x_21);
x_39 = lean_nat_sub(x_18, x_15);
lean_dec(x_15);
lean_dec(x_18);
x_40 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_40, 0, x_13);
lean_ctor_set(x_40, 1, x_39);
lean_ctor_set_uint8(x_40, sizeof(void*)*2, x_14);
x_41 = lean_array_push(x_37, x_40);
x_42 = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(x_42, 0, x_29);
lean_ctor_set(x_42, 1, x_30);
lean_ctor_set(x_42, 2, x_31);
lean_ctor_set(x_42, 3, x_32);
lean_ctor_set(x_42, 4, x_33);
lean_ctor_set(x_42, 5, x_34);
lean_ctor_set(x_42, 6, x_35);
lean_ctor_set(x_42, 7, x_36);
lean_ctor_set(x_42, 8, x_41);
lean_ctor_set(x_42, 9, x_38);
lean_ctor_set(x_19, 1, x_42);
x_43 = lean_st_ref_set(x_5, x_19);
lean_dec(x_5);
x_44 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_44, 0, x_16);
return x_44;
}
}
else
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; 
x_45 = lean_ctor_get(x_19, 1);
x_46 = lean_ctor_get(x_19, 0);
lean_inc(x_45);
lean_inc(x_46);
lean_dec(x_19);
x_47 = lean_ctor_get(x_45, 0);
lean_inc(x_47);
x_48 = lean_ctor_get(x_45, 1);
lean_inc(x_48);
x_49 = lean_ctor_get(x_45, 2);
lean_inc(x_49);
x_50 = lean_ctor_get(x_45, 3);
lean_inc(x_50);
x_51 = lean_ctor_get(x_45, 4);
lean_inc(x_51);
x_52 = lean_ctor_get(x_45, 5);
lean_inc(x_52);
x_53 = lean_ctor_get(x_45, 6);
lean_inc(x_53);
x_54 = lean_ctor_get(x_45, 7);
lean_inc(x_54);
x_55 = lean_ctor_get(x_45, 8);
lean_inc_ref(x_55);
x_56 = lean_ctor_get(x_45, 9);
lean_inc_ref(x_56);
if (lean_is_exclusive(x_45)) {
 lean_ctor_release(x_45, 0);
 lean_ctor_release(x_45, 1);
 lean_ctor_release(x_45, 2);
 lean_ctor_release(x_45, 3);
 lean_ctor_release(x_45, 4);
 lean_ctor_release(x_45, 5);
 lean_ctor_release(x_45, 6);
 lean_ctor_release(x_45, 7);
 lean_ctor_release(x_45, 8);
 lean_ctor_release(x_45, 9);
 x_57 = x_45;
} else {
 lean_dec_ref(x_45);
 x_57 = lean_box(0);
}
x_58 = lean_nat_sub(x_18, x_15);
lean_dec(x_15);
lean_dec(x_18);
x_59 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_59, 0, x_13);
lean_ctor_set(x_59, 1, x_58);
lean_ctor_set_uint8(x_59, sizeof(void*)*2, x_14);
x_60 = lean_array_push(x_55, x_59);
if (lean_is_scalar(x_57)) {
 x_61 = lean_alloc_ctor(0, 10, 0);
} else {
 x_61 = x_57;
}
lean_ctor_set(x_61, 0, x_47);
lean_ctor_set(x_61, 1, x_48);
lean_ctor_set(x_61, 2, x_49);
lean_ctor_set(x_61, 3, x_50);
lean_ctor_set(x_61, 4, x_51);
lean_ctor_set(x_61, 5, x_52);
lean_ctor_set(x_61, 6, x_53);
lean_ctor_set(x_61, 7, x_54);
lean_ctor_set(x_61, 8, x_60);
lean_ctor_set(x_61, 9, x_56);
x_62 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_62, 0, x_46);
lean_ctor_set(x_62, 1, x_61);
x_63 = lean_st_ref_set(x_5, x_62);
lean_dec(x_5);
x_64 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_64, 0, x_16);
return x_64;
}
}
block_70:
{
if (lean_obj_tag(x_68) == 0)
{
lean_object* x_69; 
x_69 = lean_ctor_get(x_68, 0);
lean_inc(x_69);
lean_dec_ref(x_68);
x_14 = x_66;
x_15 = x_67;
x_16 = x_69;
x_17 = lean_box(0);
goto block_65;
}
else
{
lean_dec(x_67);
lean_dec(x_5);
return x_68;
}
}
block_105:
{
uint8_t x_83; 
lean_dec(x_80);
lean_dec(x_79);
lean_dec_ref(x_71);
x_83 = l_Array_isEmpty___redArg(x_82);
lean_dec_ref(x_82);
if (x_83 == 0)
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; size_t x_92; size_t x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; uint8_t x_101; 
lean_dec(x_74);
x_84 = lp_aesop_Aesop_Check_name(x_73);
lean_dec_ref(x_73);
x_85 = l_Lean_MessageData_ofName(x_84);
x_86 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_87 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_87, 0, x_85);
lean_ctor_set(x_87, 1, x_86);
x_88 = lp_aesop_Aesop_normSimp___closed__1;
x_89 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_89, 0, x_87);
lean_ctor_set(x_89, 1, x_88);
x_90 = lp_aesop_Aesop_checkSimp___closed__21;
x_91 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_91, 0, x_89);
lean_ctor_set(x_91, 1, x_90);
x_92 = lean_array_size(x_75);
x_93 = 0;
x_94 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(x_92, x_93, x_75);
x_95 = lean_array_to_list(x_94);
x_96 = lean_box(0);
x_97 = lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(x_95, x_96);
x_98 = l_Lean_MessageData_ofList(x_97);
x_99 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_99, 0, x_91);
lean_ctor_set(x_99, 1, x_98);
x_100 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_99, x_78, x_76, x_81, x_72);
lean_dec(x_72);
lean_dec_ref(x_81);
lean_dec(x_76);
lean_dec_ref(x_78);
x_101 = !lean_is_exclusive(x_100);
if (x_101 == 0)
{
return x_100;
}
else
{
lean_object* x_102; lean_object* x_103; 
x_102 = lean_ctor_get(x_100, 0);
lean_inc(x_102);
lean_dec(x_100);
x_103 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_103, 0, x_102);
return x_103;
}
}
else
{
lean_object* x_104; 
lean_dec_ref(x_81);
lean_dec_ref(x_78);
lean_dec(x_76);
lean_dec_ref(x_75);
lean_dec_ref(x_73);
lean_dec(x_72);
x_104 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_104, 0, x_74);
return x_104;
}
}
block_132:
{
lean_object* x_120; 
lean_inc(x_118);
lean_inc_ref(x_117);
lean_inc(x_116);
lean_inc_ref(x_115);
x_120 = lp_batteries_Lean_Meta_getAssignedExprMVars(x_107, x_106, x_115, x_116, x_117, x_118);
if (lean_obj_tag(x_120) == 0)
{
lean_object* x_121; lean_object* x_122; lean_object* x_123; uint8_t x_124; 
x_121 = lean_ctor_get(x_120, 0);
lean_inc(x_121);
lean_dec_ref(x_120);
x_122 = lean_array_get_size(x_121);
x_123 = lean_mk_empty_array_with_capacity(x_108);
x_124 = lean_nat_dec_lt(x_108, x_122);
if (x_124 == 0)
{
lean_dec(x_121);
lean_dec(x_1);
x_71 = x_112;
x_72 = x_118;
x_73 = x_109;
x_74 = x_110;
x_75 = x_111;
x_76 = x_116;
x_77 = lean_box(0);
x_78 = x_115;
x_79 = x_114;
x_80 = x_113;
x_81 = x_117;
x_82 = x_123;
goto block_105;
}
else
{
uint8_t x_125; 
x_125 = lean_nat_dec_le(x_122, x_122);
if (x_125 == 0)
{
lean_dec(x_121);
lean_dec(x_1);
x_71 = x_112;
x_72 = x_118;
x_73 = x_109;
x_74 = x_110;
x_75 = x_111;
x_76 = x_116;
x_77 = lean_box(0);
x_78 = x_115;
x_79 = x_114;
x_80 = x_113;
x_81 = x_117;
x_82 = x_123;
goto block_105;
}
else
{
size_t x_126; size_t x_127; lean_object* x_128; 
x_126 = 0;
x_127 = lean_usize_of_nat(x_122);
x_128 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__4(x_1, x_121, x_126, x_127, x_123);
lean_dec(x_121);
lean_dec(x_1);
x_71 = x_112;
x_72 = x_118;
x_73 = x_109;
x_74 = x_110;
x_75 = x_111;
x_76 = x_116;
x_77 = lean_box(0);
x_78 = x_115;
x_79 = x_114;
x_80 = x_113;
x_81 = x_117;
x_82 = x_128;
goto block_105;
}
}
}
else
{
uint8_t x_129; 
lean_dec(x_118);
lean_dec_ref(x_117);
lean_dec(x_116);
lean_dec_ref(x_115);
lean_dec(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec_ref(x_111);
lean_dec(x_110);
lean_dec_ref(x_109);
lean_dec(x_1);
x_129 = !lean_is_exclusive(x_120);
if (x_129 == 0)
{
return x_120;
}
else
{
lean_object* x_130; lean_object* x_131; 
x_130 = lean_ctor_get(x_120, 0);
lean_inc(x_130);
lean_dec(x_120);
x_131 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_131, 0, x_130);
return x_131;
}
}
}
block_161:
{
uint8_t x_140; 
x_140 = l_Array_isEmpty___redArg(x_139);
if (x_140 == 0)
{
lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; size_t x_149; size_t x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; uint8_t x_158; 
lean_dec(x_137);
lean_dec_ref(x_134);
lean_dec_ref(x_133);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
x_141 = lp_aesop_Aesop_Check_name(x_135);
lean_dec_ref(x_135);
x_142 = l_Lean_MessageData_ofName(x_141);
x_143 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_144 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_144, 0, x_142);
lean_ctor_set(x_144, 1, x_143);
x_145 = lp_aesop_Aesop_normSimp___closed__1;
x_146 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_146, 0, x_144);
lean_ctor_set(x_146, 1, x_145);
x_147 = lp_aesop_Aesop_checkSimp___closed__34;
x_148 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_148, 0, x_146);
lean_ctor_set(x_148, 1, x_147);
x_149 = lean_array_size(x_139);
x_150 = 0;
x_151 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(x_149, x_150, x_139);
x_152 = lean_array_to_list(x_151);
x_153 = lean_box(0);
x_154 = lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(x_152, x_153);
x_155 = l_Lean_MessageData_ofList(x_154);
x_156 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_156, 0, x_148);
lean_ctor_set(x_156, 1, x_155);
x_157 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_156, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_158 = !lean_is_exclusive(x_157);
if (x_158 == 0)
{
return x_157;
}
else
{
lean_object* x_159; lean_object* x_160; 
x_159 = lean_ctor_get(x_157, 0);
lean_inc(x_159);
lean_dec(x_157);
x_160 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_160, 0, x_159);
return x_160;
}
}
else
{
x_106 = x_133;
x_107 = x_134;
x_108 = x_136;
x_109 = x_135;
x_110 = x_137;
x_111 = x_139;
x_112 = x_3;
x_113 = x_4;
x_114 = x_5;
x_115 = x_6;
x_116 = x_7;
x_117 = x_8;
x_118 = x_9;
x_119 = lean_box(0);
goto block_132;
}
}
block_185:
{
lean_object* x_167; 
x_167 = l_Lean_Meta_saveState___redArg(x_7, x_9);
if (lean_obj_tag(x_167) == 0)
{
lean_object* x_168; lean_object* x_169; 
x_168 = lean_ctor_get(x_167, 0);
lean_inc(x_168);
lean_dec_ref(x_167);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_168);
lean_inc_ref(x_162);
x_169 = lp_batteries_Lean_Meta_getIntroducedExprMVars(x_162, x_168, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_169) == 0)
{
lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; uint8_t x_174; 
x_170 = lean_ctor_get(x_169, 0);
lean_inc(x_170);
lean_dec_ref(x_169);
x_171 = lean_unsigned_to_nat(0u);
x_172 = lean_array_get_size(x_170);
x_173 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_174 = lean_nat_dec_lt(x_171, x_172);
if (x_174 == 0)
{
lean_dec(x_170);
lean_dec(x_166);
x_133 = x_168;
x_134 = x_162;
x_135 = x_164;
x_136 = x_171;
x_137 = x_165;
x_138 = lean_box(0);
x_139 = x_173;
goto block_161;
}
else
{
uint8_t x_175; 
x_175 = lean_nat_dec_le(x_172, x_172);
if (x_175 == 0)
{
lean_dec(x_170);
lean_dec(x_166);
x_133 = x_168;
x_134 = x_162;
x_135 = x_164;
x_136 = x_171;
x_137 = x_165;
x_138 = lean_box(0);
x_139 = x_173;
goto block_161;
}
else
{
size_t x_176; size_t x_177; lean_object* x_178; 
x_176 = 0;
x_177 = lean_usize_of_nat(x_172);
x_178 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__5(x_166, x_170, x_176, x_177, x_173);
lean_dec(x_170);
lean_dec(x_166);
x_133 = x_168;
x_134 = x_162;
x_135 = x_164;
x_136 = x_171;
x_137 = x_165;
x_138 = lean_box(0);
x_139 = x_178;
goto block_161;
}
}
}
else
{
uint8_t x_179; 
lean_dec(x_168);
lean_dec(x_166);
lean_dec(x_165);
lean_dec_ref(x_164);
lean_dec_ref(x_162);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
x_179 = !lean_is_exclusive(x_169);
if (x_179 == 0)
{
return x_169;
}
else
{
lean_object* x_180; lean_object* x_181; 
x_180 = lean_ctor_get(x_169, 0);
lean_inc(x_180);
lean_dec(x_169);
x_181 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_181, 0, x_180);
return x_181;
}
}
}
else
{
uint8_t x_182; 
lean_dec(x_166);
lean_dec(x_165);
lean_dec_ref(x_164);
lean_dec_ref(x_162);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
x_182 = !lean_is_exclusive(x_167);
if (x_182 == 0)
{
return x_167;
}
else
{
lean_object* x_183; lean_object* x_184; 
x_183 = lean_ctor_get(x_167, 0);
lean_inc(x_183);
lean_dec(x_167);
x_184 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_184, 0, x_183);
return x_184;
}
}
}
block_214:
{
lean_object* x_187; lean_object* x_188; lean_object* x_189; uint8_t x_190; 
x_187 = lp_aesop_Aesop_runNormRuleTac___closed__8;
x_188 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(x_187, x_8);
x_189 = lean_ctor_get(x_188, 0);
lean_inc(x_189);
lean_dec_ref(x_188);
x_190 = lean_unbox(x_189);
if (x_190 == 0)
{
lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; uint8_t x_195; lean_object* x_196; lean_object* x_197; 
lean_dec(x_189);
lean_dec(x_1);
x_191 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_192 = lean_ctor_get(x_191, 0);
lean_inc(x_192);
x_193 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimp___lam__1___boxed), 10, 2);
lean_closure_set(x_193, 0, x_12);
lean_closure_set(x_193, 1, x_191);
x_194 = lp_aesop_Aesop_normSimp___closed__2;
x_195 = 1;
x_196 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_197 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_192, x_194, x_193, x_195, x_196, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_197;
}
else
{
lean_object* x_198; 
x_198 = l_Lean_Meta_saveState___redArg(x_7, x_9);
if (lean_obj_tag(x_198) == 0)
{
lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; uint8_t x_205; lean_object* x_206; 
x_199 = lean_ctor_get(x_198, 0);
lean_inc(x_199);
lean_dec_ref(x_198);
x_200 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_201 = lean_ctor_get(x_200, 0);
lean_inc(x_201);
x_202 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimp___lam__1___boxed), 10, 2);
lean_closure_set(x_202, 0, x_12);
lean_closure_set(x_202, 1, x_200);
x_203 = lp_aesop_Aesop_normSimp___closed__2;
x_204 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_205 = lean_unbox(x_189);
lean_dec(x_189);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_206 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_201, x_203, x_202, x_205, x_204, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_206) == 0)
{
lean_object* x_207; 
x_207 = lean_ctor_get(x_206, 0);
lean_inc(x_207);
lean_dec_ref(x_206);
if (lean_obj_tag(x_207) == 0)
{
lean_object* x_208; 
x_208 = lean_box(0);
x_162 = x_199;
x_163 = lean_box(0);
x_164 = x_187;
x_165 = x_207;
x_166 = x_208;
goto block_185;
}
else
{
lean_object* x_209; lean_object* x_210; 
x_209 = lean_ctor_get(x_207, 0);
x_210 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_209);
x_162 = x_199;
x_163 = lean_box(0);
x_164 = x_187;
x_165 = x_207;
x_166 = x_210;
goto block_185;
}
}
else
{
lean_dec(x_199);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
return x_206;
}
}
else
{
uint8_t x_211; 
lean_dec(x_189);
lean_dec_ref(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
x_211 = !lean_is_exclusive(x_198);
if (x_211 == 0)
{
return x_198;
}
else
{
lean_object* x_212; lean_object* x_213; 
x_212 = lean_ctor_get(x_198, 0);
lean_inc(x_212);
lean_dec(x_198);
x_213 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_213, 0, x_212);
return x_213;
}
}
}
}
block_250:
{
uint8_t x_229; 
lean_dec(x_225);
lean_dec(x_221);
lean_dec_ref(x_219);
x_229 = l_Array_isEmpty___redArg(x_228);
lean_dec_ref(x_228);
if (x_229 == 0)
{
lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; size_t x_238; size_t x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; lean_object* x_245; lean_object* x_246; uint8_t x_247; 
lean_dec(x_226);
lean_dec(x_220);
lean_dec(x_5);
x_230 = lp_aesop_Aesop_Check_name(x_227);
lean_dec_ref(x_227);
x_231 = l_Lean_MessageData_ofName(x_230);
x_232 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_233 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_233, 0, x_231);
lean_ctor_set(x_233, 1, x_232);
x_234 = lp_aesop_Aesop_normSimp___closed__1;
x_235 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_235, 0, x_233);
lean_ctor_set(x_235, 1, x_234);
x_236 = lp_aesop_Aesop_checkSimp___closed__21;
x_237 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_237, 0, x_235);
lean_ctor_set(x_237, 1, x_236);
x_238 = lean_array_size(x_222);
x_239 = 0;
x_240 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(x_238, x_239, x_222);
x_241 = lean_array_to_list(x_240);
x_242 = lean_box(0);
x_243 = lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(x_241, x_242);
x_244 = l_Lean_MessageData_ofList(x_243);
x_245 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_245, 0, x_237);
lean_ctor_set(x_245, 1, x_244);
x_246 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_245, x_217, x_215, x_224, x_223);
lean_dec(x_223);
lean_dec_ref(x_224);
lean_dec(x_215);
lean_dec_ref(x_217);
x_247 = !lean_is_exclusive(x_246);
if (x_247 == 0)
{
return x_246;
}
else
{
lean_object* x_248; lean_object* x_249; 
x_248 = lean_ctor_get(x_246, 0);
lean_inc(x_248);
lean_dec(x_246);
x_249 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_249, 0, x_248);
return x_249;
}
}
else
{
lean_dec_ref(x_227);
lean_dec_ref(x_224);
lean_dec(x_223);
lean_dec_ref(x_222);
lean_dec_ref(x_217);
lean_dec(x_215);
x_14 = x_216;
x_15 = x_220;
x_16 = x_226;
x_17 = lean_box(0);
goto block_65;
}
}
block_279:
{
lean_object* x_267; 
lean_inc(x_265);
lean_inc_ref(x_264);
lean_inc(x_263);
lean_inc_ref(x_262);
x_267 = lp_batteries_Lean_Meta_getAssignedExprMVars(x_253, x_257, x_262, x_263, x_264, x_265);
if (lean_obj_tag(x_267) == 0)
{
lean_object* x_268; lean_object* x_269; lean_object* x_270; uint8_t x_271; 
x_268 = lean_ctor_get(x_267, 0);
lean_inc(x_268);
lean_dec_ref(x_267);
x_269 = lean_array_get_size(x_268);
x_270 = lean_mk_empty_array_with_capacity(x_256);
x_271 = lean_nat_dec_lt(x_256, x_269);
if (x_271 == 0)
{
lean_dec(x_268);
lean_dec(x_1);
x_215 = x_263;
x_216 = x_251;
x_217 = x_262;
x_218 = lean_box(0);
x_219 = x_259;
x_220 = x_255;
x_221 = x_261;
x_222 = x_258;
x_223 = x_265;
x_224 = x_264;
x_225 = x_260;
x_226 = x_252;
x_227 = x_254;
x_228 = x_270;
goto block_250;
}
else
{
uint8_t x_272; 
x_272 = lean_nat_dec_le(x_269, x_269);
if (x_272 == 0)
{
lean_dec(x_268);
lean_dec(x_1);
x_215 = x_263;
x_216 = x_251;
x_217 = x_262;
x_218 = lean_box(0);
x_219 = x_259;
x_220 = x_255;
x_221 = x_261;
x_222 = x_258;
x_223 = x_265;
x_224 = x_264;
x_225 = x_260;
x_226 = x_252;
x_227 = x_254;
x_228 = x_270;
goto block_250;
}
else
{
size_t x_273; size_t x_274; lean_object* x_275; 
x_273 = 0;
x_274 = lean_usize_of_nat(x_269);
x_275 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__4(x_1, x_268, x_273, x_274, x_270);
lean_dec(x_268);
lean_dec(x_1);
x_215 = x_263;
x_216 = x_251;
x_217 = x_262;
x_218 = lean_box(0);
x_219 = x_259;
x_220 = x_255;
x_221 = x_261;
x_222 = x_258;
x_223 = x_265;
x_224 = x_264;
x_225 = x_260;
x_226 = x_252;
x_227 = x_254;
x_228 = x_275;
goto block_250;
}
}
}
else
{
uint8_t x_276; 
lean_dec(x_265);
lean_dec_ref(x_264);
lean_dec(x_263);
lean_dec_ref(x_262);
lean_dec(x_261);
lean_dec(x_260);
lean_dec_ref(x_259);
lean_dec_ref(x_258);
lean_dec(x_255);
lean_dec_ref(x_254);
lean_dec(x_252);
lean_dec(x_5);
lean_dec(x_1);
x_276 = !lean_is_exclusive(x_267);
if (x_276 == 0)
{
return x_267;
}
else
{
lean_object* x_277; lean_object* x_278; 
x_277 = lean_ctor_get(x_267, 0);
lean_inc(x_277);
lean_dec(x_267);
x_278 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_278, 0, x_277);
return x_278;
}
}
}
block_310:
{
uint8_t x_289; 
x_289 = l_Array_isEmpty___redArg(x_288);
if (x_289 == 0)
{
lean_object* x_290; lean_object* x_291; lean_object* x_292; lean_object* x_293; lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; size_t x_298; size_t x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; uint8_t x_307; 
lean_dec_ref(x_285);
lean_dec(x_284);
lean_dec(x_282);
lean_dec_ref(x_281);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
x_290 = lp_aesop_Aesop_Check_name(x_283);
lean_dec_ref(x_283);
x_291 = l_Lean_MessageData_ofName(x_290);
x_292 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_293 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_293, 0, x_291);
lean_ctor_set(x_293, 1, x_292);
x_294 = lp_aesop_Aesop_normSimp___closed__1;
x_295 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_295, 0, x_293);
lean_ctor_set(x_295, 1, x_294);
x_296 = lp_aesop_Aesop_checkSimp___closed__34;
x_297 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_297, 0, x_295);
lean_ctor_set(x_297, 1, x_296);
x_298 = lean_array_size(x_288);
x_299 = 0;
x_300 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(x_298, x_299, x_288);
x_301 = lean_array_to_list(x_300);
x_302 = lean_box(0);
x_303 = lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(x_301, x_302);
x_304 = l_Lean_MessageData_ofList(x_303);
x_305 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_305, 0, x_297);
lean_ctor_set(x_305, 1, x_304);
x_306 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_305, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_307 = !lean_is_exclusive(x_306);
if (x_307 == 0)
{
return x_306;
}
else
{
lean_object* x_308; lean_object* x_309; 
x_308 = lean_ctor_get(x_306, 0);
lean_inc(x_308);
lean_dec(x_306);
x_309 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_309, 0, x_308);
return x_309;
}
}
else
{
lean_inc(x_5);
x_251 = x_280;
x_252 = x_282;
x_253 = x_281;
x_254 = x_283;
x_255 = x_284;
x_256 = x_286;
x_257 = x_285;
x_258 = x_288;
x_259 = x_3;
x_260 = x_4;
x_261 = x_5;
x_262 = x_6;
x_263 = x_7;
x_264 = x_8;
x_265 = x_9;
x_266 = lean_box(0);
goto block_279;
}
}
block_336:
{
lean_object* x_318; 
x_318 = l_Lean_Meta_saveState___redArg(x_7, x_9);
if (lean_obj_tag(x_318) == 0)
{
lean_object* x_319; lean_object* x_320; 
x_319 = lean_ctor_get(x_318, 0);
lean_inc(x_319);
lean_dec_ref(x_318);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_319);
lean_inc_ref(x_314);
x_320 = lp_batteries_Lean_Meta_getIntroducedExprMVars(x_314, x_319, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_320) == 0)
{
lean_object* x_321; lean_object* x_322; lean_object* x_323; lean_object* x_324; uint8_t x_325; 
x_321 = lean_ctor_get(x_320, 0);
lean_inc(x_321);
lean_dec_ref(x_320);
x_322 = lean_unsigned_to_nat(0u);
x_323 = lean_array_get_size(x_321);
x_324 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_325 = lean_nat_dec_lt(x_322, x_323);
if (x_325 == 0)
{
lean_dec(x_321);
lean_dec(x_317);
x_280 = x_312;
x_281 = x_314;
x_282 = x_313;
x_283 = x_315;
x_284 = x_316;
x_285 = x_319;
x_286 = x_322;
x_287 = lean_box(0);
x_288 = x_324;
goto block_310;
}
else
{
uint8_t x_326; 
x_326 = lean_nat_dec_le(x_323, x_323);
if (x_326 == 0)
{
lean_dec(x_321);
lean_dec(x_317);
x_280 = x_312;
x_281 = x_314;
x_282 = x_313;
x_283 = x_315;
x_284 = x_316;
x_285 = x_319;
x_286 = x_322;
x_287 = lean_box(0);
x_288 = x_324;
goto block_310;
}
else
{
size_t x_327; size_t x_328; lean_object* x_329; 
x_327 = 0;
x_328 = lean_usize_of_nat(x_323);
x_329 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__5(x_317, x_321, x_327, x_328, x_324);
lean_dec(x_321);
lean_dec(x_317);
x_280 = x_312;
x_281 = x_314;
x_282 = x_313;
x_283 = x_315;
x_284 = x_316;
x_285 = x_319;
x_286 = x_322;
x_287 = lean_box(0);
x_288 = x_329;
goto block_310;
}
}
}
else
{
uint8_t x_330; 
lean_dec(x_319);
lean_dec(x_317);
lean_dec(x_316);
lean_dec_ref(x_315);
lean_dec_ref(x_314);
lean_dec(x_313);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
x_330 = !lean_is_exclusive(x_320);
if (x_330 == 0)
{
return x_320;
}
else
{
lean_object* x_331; lean_object* x_332; 
x_331 = lean_ctor_get(x_320, 0);
lean_inc(x_331);
lean_dec(x_320);
x_332 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_332, 0, x_331);
return x_332;
}
}
}
else
{
uint8_t x_333; 
lean_dec(x_317);
lean_dec(x_316);
lean_dec_ref(x_315);
lean_dec_ref(x_314);
lean_dec(x_313);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
x_333 = !lean_is_exclusive(x_318);
if (x_333 == 0)
{
return x_318;
}
else
{
lean_object* x_334; lean_object* x_335; 
x_334 = lean_ctor_get(x_318, 0);
lean_inc(x_334);
lean_dec(x_318);
x_335 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_335, 0, x_334);
return x_335;
}
}
}
block_366:
{
lean_object* x_339; lean_object* x_340; lean_object* x_341; lean_object* x_342; uint8_t x_343; 
x_339 = lean_io_mono_nanos_now();
x_340 = lp_aesop_Aesop_runNormRuleTac___closed__8;
x_341 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(x_340, x_8);
x_342 = lean_ctor_get(x_341, 0);
lean_inc(x_342);
lean_dec_ref(x_341);
x_343 = lean_unbox(x_342);
if (x_343 == 0)
{
lean_object* x_344; lean_object* x_345; lean_object* x_346; lean_object* x_347; lean_object* x_348; lean_object* x_349; 
lean_dec(x_342);
lean_dec(x_1);
x_344 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_345 = lean_ctor_get(x_344, 0);
lean_inc(x_345);
x_346 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimp___lam__1___boxed), 10, 2);
lean_closure_set(x_346, 0, x_12);
lean_closure_set(x_346, 1, x_344);
x_347 = lp_aesop_Aesop_normSimp___closed__2;
x_348 = lp_aesop_Aesop_withNormTraceNode___closed__43;
lean_inc(x_5);
x_349 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_345, x_347, x_346, x_337, x_348, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
x_66 = x_337;
x_67 = x_339;
x_68 = x_349;
goto block_70;
}
else
{
lean_object* x_350; 
x_350 = l_Lean_Meta_saveState___redArg(x_7, x_9);
if (lean_obj_tag(x_350) == 0)
{
lean_object* x_351; lean_object* x_352; lean_object* x_353; lean_object* x_354; lean_object* x_355; lean_object* x_356; uint8_t x_357; lean_object* x_358; 
x_351 = lean_ctor_get(x_350, 0);
lean_inc(x_351);
lean_dec_ref(x_350);
x_352 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_353 = lean_ctor_get(x_352, 0);
lean_inc(x_353);
x_354 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimp___lam__1___boxed), 10, 2);
lean_closure_set(x_354, 0, x_12);
lean_closure_set(x_354, 1, x_352);
x_355 = lp_aesop_Aesop_normSimp___closed__2;
x_356 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_357 = lean_unbox(x_342);
lean_dec(x_342);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_358 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_353, x_355, x_354, x_357, x_356, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_358) == 0)
{
lean_object* x_359; 
x_359 = lean_ctor_get(x_358, 0);
lean_inc(x_359);
lean_dec_ref(x_358);
if (lean_obj_tag(x_359) == 0)
{
lean_object* x_360; 
x_360 = lean_box(0);
x_311 = lean_box(0);
x_312 = x_337;
x_313 = x_359;
x_314 = x_351;
x_315 = x_340;
x_316 = x_339;
x_317 = x_360;
goto block_336;
}
else
{
lean_object* x_361; lean_object* x_362; 
x_361 = lean_ctor_get(x_359, 0);
x_362 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_361);
x_311 = lean_box(0);
x_312 = x_337;
x_313 = x_359;
x_314 = x_351;
x_315 = x_340;
x_316 = x_339;
x_317 = x_362;
goto block_336;
}
}
else
{
lean_dec(x_351);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
x_66 = x_337;
x_67 = x_339;
x_68 = x_358;
goto block_70;
}
}
else
{
uint8_t x_363; 
lean_dec(x_342);
lean_dec(x_339);
lean_dec_ref(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_1);
x_363 = !lean_is_exclusive(x_350);
if (x_363 == 0)
{
return x_350;
}
else
{
lean_object* x_364; lean_object* x_365; 
x_364 = lean_ctor_get(x_350, 0);
lean_inc(x_364);
lean_dec(x_350);
x_365 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_365, 0, x_364);
return x_365;
}
}
}
}
block_371:
{
lean_object* x_368; uint8_t x_369; 
x_368 = lean_ctor_get(x_367, 0);
lean_inc(x_368);
lean_dec_ref(x_367);
x_369 = lean_unbox(x_368);
if (x_369 == 0)
{
lean_dec(x_368);
x_186 = lean_box(0);
goto block_214;
}
else
{
uint8_t x_370; 
x_370 = lean_unbox(x_368);
lean_dec(x_368);
x_337 = x_370;
x_338 = lean_box(0);
goto block_366;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; lean_object* x_6; 
x_4 = lean_unbox_usize(x_1);
lean_dec(x_1);
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(x_4, x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_instBEq_beq___at___00Aesop_normSimp_spec__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Option_instBEq_beq___at___00Aesop_normSimp_spec__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_8 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__4(x_1, x_2, x_6, x_7, x_5);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_8 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__5(x_1, x_2, x_6, x_7, x_5);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normSimp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_normSimp(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
static lean_object* _init_lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg___closed__0;
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg(x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_array_get_size(x_1);
x_6 = lean_nat_dec_lt(x_3, x_5);
if (x_6 == 0)
{
lean_object* x_7; 
lean_dec(x_3);
x_7 = lean_box(0);
return x_7;
}
else
{
lean_object* x_8; uint8_t x_9; 
x_8 = lean_array_fget_borrowed(x_1, x_3);
x_9 = lean_name_eq(x_4, x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_unsigned_to_nat(1u);
x_11 = lean_nat_add(x_3, x_10);
lean_dec(x_3);
x_3 = x_11;
goto _start;
}
else
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_array_fget_borrowed(x_2, x_3);
lean_dec(x_3);
lean_inc(x_13);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0___redArg(lean_object* x_1, size_t x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; size_t x_7; size_t x_8; size_t x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_box(2);
x_7 = 5;
x_8 = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__1;
x_9 = lean_usize_land(x_2, x_8);
x_10 = lean_usize_to_nat(x_9);
x_11 = lean_array_get(x_6, x_5, x_10);
lean_dec(x_10);
lean_dec_ref(x_5);
switch (lean_obj_tag(x_11)) {
case 0:
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_11, 1);
lean_inc(x_13);
lean_dec_ref(x_11);
x_14 = lean_name_eq(x_3, x_12);
lean_dec(x_12);
if (x_14 == 0)
{
lean_object* x_15; 
lean_dec(x_13);
lean_free_object(x_1);
x_15 = lean_box(0);
return x_15;
}
else
{
lean_ctor_set_tag(x_1, 1);
lean_ctor_set(x_1, 0, x_13);
return x_1;
}
}
case 1:
{
lean_object* x_16; size_t x_17; 
lean_free_object(x_1);
x_16 = lean_ctor_get(x_11, 0);
lean_inc(x_16);
lean_dec_ref(x_11);
x_17 = lean_usize_shift_right(x_2, x_7);
x_1 = x_16;
x_2 = x_17;
goto _start;
}
default: 
{
lean_object* x_19; 
lean_free_object(x_1);
x_19 = lean_box(0);
return x_19;
}
}
}
else
{
lean_object* x_20; lean_object* x_21; size_t x_22; size_t x_23; size_t x_24; lean_object* x_25; lean_object* x_26; 
x_20 = lean_ctor_get(x_1, 0);
lean_inc(x_20);
lean_dec(x_1);
x_21 = lean_box(2);
x_22 = 5;
x_23 = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__1;
x_24 = lean_usize_land(x_2, x_23);
x_25 = lean_usize_to_nat(x_24);
x_26 = lean_array_get(x_21, x_20, x_25);
lean_dec(x_25);
lean_dec_ref(x_20);
switch (lean_obj_tag(x_26)) {
case 0:
{
lean_object* x_27; lean_object* x_28; uint8_t x_29; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 1);
lean_inc(x_28);
lean_dec_ref(x_26);
x_29 = lean_name_eq(x_3, x_27);
lean_dec(x_27);
if (x_29 == 0)
{
lean_object* x_30; 
lean_dec(x_28);
x_30 = lean_box(0);
return x_30;
}
else
{
lean_object* x_31; 
x_31 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_31, 0, x_28);
return x_31;
}
}
case 1:
{
lean_object* x_32; size_t x_33; 
x_32 = lean_ctor_get(x_26, 0);
lean_inc(x_32);
lean_dec_ref(x_26);
x_33 = lean_usize_shift_right(x_2, x_22);
x_1 = x_32;
x_2 = x_33;
goto _start;
}
default: 
{
lean_object* x_35; 
x_35 = lean_box(0);
return x_35;
}
}
}
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_36 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_36);
x_37 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_37);
lean_dec_ref(x_1);
x_38 = lean_unsigned_to_nat(0u);
x_39 = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0___redArg(x_36, x_37, x_38, x_3);
lean_dec_ref(x_37);
lean_dec_ref(x_36);
return x_39;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint64_t x_3; size_t x_4; lean_object* x_5; 
x_3 = l_Lean_Name_hash___override(x_2);
x_4 = lean_uint64_to_usize(x_3);
x_5 = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0___redArg(x_1, x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfoldCore___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0___redArg(x_2, x_3, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0(lean_object* x_1, lean_object* x_2, size_t x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_normUnfoldCore___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("nothing to unfold", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normUnfoldCore___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normUnfoldCore___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfoldCore___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_normUnfoldCore___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfoldCore(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_14 = lean_ctor_get(x_2, 1);
x_15 = lean_ctor_get(x_14, 0);
x_16 = lean_ctor_get(x_15, 3);
lean_inc_ref(x_16);
x_17 = lean_alloc_closure((void*)(lp_aesop_Aesop_normUnfoldCore___lam__0___boxed), 2, 1);
lean_closure_set(x_17, 0, x_16);
lean_inc(x_1);
x_18 = lean_alloc_closure((void*)(lp_aesop_Aesop_unfoldManyStarS___boxed), 9, 2);
lean_closure_set(x_18, 0, x_1);
lean_closure_set(x_18, 1, x_17);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
x_19 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg(x_18, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_19) == 0)
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
if (lean_obj_tag(x_21) == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
lean_dec(x_20);
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
x_22 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_23 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_22, x_7);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
lean_dec_ref(x_23);
x_25 = lean_unbox(x_24);
lean_dec(x_24);
if (x_25 == 0)
{
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
x_10 = lean_box(0);
goto block_13;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_26 = lean_ctor_get(x_22, 0);
lean_inc(x_26);
x_27 = lp_aesop_Aesop_normUnfoldCore___closed__1;
x_28 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_26, x_27, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
if (lean_obj_tag(x_28) == 0)
{
lean_dec_ref(x_28);
x_10 = lean_box(0);
goto block_13;
}
else
{
uint8_t x_29; 
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
return x_28;
}
else
{
lean_object* x_30; lean_object* x_31; 
x_30 = lean_ctor_get(x_28, 0);
lean_inc(x_30);
lean_dec(x_28);
x_31 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_31, 0, x_30);
return x_31;
}
}
}
}
else
{
uint8_t x_32; 
x_32 = !lean_is_exclusive(x_20);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; uint8_t x_35; 
x_33 = lean_ctor_get(x_20, 1);
x_34 = lean_ctor_get(x_20, 0);
lean_dec(x_34);
x_35 = !lean_is_exclusive(x_21);
if (x_35 == 0)
{
lean_object* x_36; lean_object* x_37; 
x_36 = lean_ctor_get(x_21, 0);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc(x_36);
x_37 = lp_aesop_Aesop_diffGoals(x_1, x_36, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_37) == 0)
{
lean_object* x_38; lean_object* x_39; 
x_38 = lean_ctor_get(x_37, 0);
lean_inc(x_38);
lean_dec_ref(x_37);
x_39 = lp_aesop_Aesop_applyDiffToForwardState(x_38, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_39) == 0)
{
uint8_t x_40; 
x_40 = !lean_is_exclusive(x_39);
if (x_40 == 0)
{
lean_object* x_41; lean_object* x_42; 
x_41 = lean_ctor_get(x_39, 0);
lean_dec(x_41);
lean_ctor_set(x_21, 0, x_33);
lean_ctor_set(x_20, 1, x_21);
lean_ctor_set(x_20, 0, x_36);
x_42 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_42, 0, x_20);
lean_ctor_set(x_39, 0, x_42);
return x_39;
}
else
{
lean_object* x_43; lean_object* x_44; 
lean_dec(x_39);
lean_ctor_set(x_21, 0, x_33);
lean_ctor_set(x_20, 1, x_21);
lean_ctor_set(x_20, 0, x_36);
x_43 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_43, 0, x_20);
x_44 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_44, 0, x_43);
return x_44;
}
}
else
{
uint8_t x_45; 
lean_free_object(x_21);
lean_dec(x_36);
lean_free_object(x_20);
lean_dec(x_33);
x_45 = !lean_is_exclusive(x_39);
if (x_45 == 0)
{
return x_39;
}
else
{
lean_object* x_46; lean_object* x_47; 
x_46 = lean_ctor_get(x_39, 0);
lean_inc(x_46);
lean_dec(x_39);
x_47 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_47, 0, x_46);
return x_47;
}
}
}
else
{
uint8_t x_48; 
lean_free_object(x_21);
lean_dec(x_36);
lean_free_object(x_20);
lean_dec(x_33);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
x_48 = !lean_is_exclusive(x_37);
if (x_48 == 0)
{
return x_37;
}
else
{
lean_object* x_49; lean_object* x_50; 
x_49 = lean_ctor_get(x_37, 0);
lean_inc(x_49);
lean_dec(x_37);
x_50 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_50, 0, x_49);
return x_50;
}
}
}
else
{
lean_object* x_51; lean_object* x_52; 
x_51 = lean_ctor_get(x_21, 0);
lean_inc(x_51);
lean_dec(x_21);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc(x_51);
x_52 = lp_aesop_Aesop_diffGoals(x_1, x_51, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_52) == 0)
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_ctor_get(x_52, 0);
lean_inc(x_53);
lean_dec_ref(x_52);
x_54 = lp_aesop_Aesop_applyDiffToForwardState(x_53, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_54) == 0)
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; 
if (lean_is_exclusive(x_54)) {
 lean_ctor_release(x_54, 0);
 x_55 = x_54;
} else {
 lean_dec_ref(x_54);
 x_55 = lean_box(0);
}
x_56 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_56, 0, x_33);
lean_ctor_set(x_20, 1, x_56);
lean_ctor_set(x_20, 0, x_51);
x_57 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_57, 0, x_20);
if (lean_is_scalar(x_55)) {
 x_58 = lean_alloc_ctor(0, 1, 0);
} else {
 x_58 = x_55;
}
lean_ctor_set(x_58, 0, x_57);
return x_58;
}
else
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; 
lean_dec(x_51);
lean_free_object(x_20);
lean_dec(x_33);
x_59 = lean_ctor_get(x_54, 0);
lean_inc(x_59);
if (lean_is_exclusive(x_54)) {
 lean_ctor_release(x_54, 0);
 x_60 = x_54;
} else {
 lean_dec_ref(x_54);
 x_60 = lean_box(0);
}
if (lean_is_scalar(x_60)) {
 x_61 = lean_alloc_ctor(1, 1, 0);
} else {
 x_61 = x_60;
}
lean_ctor_set(x_61, 0, x_59);
return x_61;
}
}
else
{
lean_object* x_62; lean_object* x_63; lean_object* x_64; 
lean_dec(x_51);
lean_free_object(x_20);
lean_dec(x_33);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
x_62 = lean_ctor_get(x_52, 0);
lean_inc(x_62);
if (lean_is_exclusive(x_52)) {
 lean_ctor_release(x_52, 0);
 x_63 = x_52;
} else {
 lean_dec_ref(x_52);
 x_63 = lean_box(0);
}
if (lean_is_scalar(x_63)) {
 x_64 = lean_alloc_ctor(1, 1, 0);
} else {
 x_64 = x_63;
}
lean_ctor_set(x_64, 0, x_62);
return x_64;
}
}
}
else
{
lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_65 = lean_ctor_get(x_20, 1);
lean_inc(x_65);
lean_dec(x_20);
x_66 = lean_ctor_get(x_21, 0);
lean_inc(x_66);
if (lean_is_exclusive(x_21)) {
 lean_ctor_release(x_21, 0);
 x_67 = x_21;
} else {
 lean_dec_ref(x_21);
 x_67 = lean_box(0);
}
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc(x_66);
x_68 = lp_aesop_Aesop_diffGoals(x_1, x_66, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_68) == 0)
{
lean_object* x_69; lean_object* x_70; 
x_69 = lean_ctor_get(x_68, 0);
lean_inc(x_69);
lean_dec_ref(x_68);
x_70 = lp_aesop_Aesop_applyDiffToForwardState(x_69, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_70) == 0)
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
if (lean_is_exclusive(x_70)) {
 lean_ctor_release(x_70, 0);
 x_71 = x_70;
} else {
 lean_dec_ref(x_70);
 x_71 = lean_box(0);
}
if (lean_is_scalar(x_67)) {
 x_72 = lean_alloc_ctor(1, 1, 0);
} else {
 x_72 = x_67;
}
lean_ctor_set(x_72, 0, x_65);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_66);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_74, 0, x_73);
if (lean_is_scalar(x_71)) {
 x_75 = lean_alloc_ctor(0, 1, 0);
} else {
 x_75 = x_71;
}
lean_ctor_set(x_75, 0, x_74);
return x_75;
}
else
{
lean_object* x_76; lean_object* x_77; lean_object* x_78; 
lean_dec(x_67);
lean_dec(x_66);
lean_dec(x_65);
x_76 = lean_ctor_get(x_70, 0);
lean_inc(x_76);
if (lean_is_exclusive(x_70)) {
 lean_ctor_release(x_70, 0);
 x_77 = x_70;
} else {
 lean_dec_ref(x_70);
 x_77 = lean_box(0);
}
if (lean_is_scalar(x_77)) {
 x_78 = lean_alloc_ctor(1, 1, 0);
} else {
 x_78 = x_77;
}
lean_ctor_set(x_78, 0, x_76);
return x_78;
}
}
else
{
lean_object* x_79; lean_object* x_80; lean_object* x_81; 
lean_dec(x_67);
lean_dec(x_66);
lean_dec(x_65);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
x_79 = lean_ctor_get(x_68, 0);
lean_inc(x_79);
if (lean_is_exclusive(x_68)) {
 lean_ctor_release(x_68, 0);
 x_80 = x_68;
} else {
 lean_dec_ref(x_68);
 x_80 = lean_box(0);
}
if (lean_is_scalar(x_80)) {
 x_81 = lean_alloc_ctor(1, 1, 0);
} else {
 x_81 = x_80;
}
lean_ctor_set(x_81, 0, x_79);
return x_81;
}
}
}
}
else
{
uint8_t x_82; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
x_82 = !lean_is_exclusive(x_19);
if (x_82 == 0)
{
return x_19;
}
else
{
lean_object* x_83; lean_object* x_84; 
x_83 = lean_ctor_get(x_19, 0);
lean_inc(x_83);
lean_dec(x_19);
x_84 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_84, 0, x_83);
return x_84;
}
}
block_13:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_box(0);
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; lean_object* x_6; 
x_5 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_6 = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0(x_1, x_2, x_5, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0___redArg(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0_spec__0___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; lean_object* x_5; 
x_4 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_5 = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_normUnfoldCore_spec__0_spec__0___redArg(x_1, x_4, x_3);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfoldCore___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_normUnfoldCore(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_st_ref_get(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec(x_4);
x_6 = lean_ctor_get(x_5, 7);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0___redArg(x_6, x_1);
x_8 = lean_box(x_7);
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___redArg(x_1, x_6);
return x_10;
}
}
static lean_object* _init_lp_aesop_Aesop_normUnfold___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: error in norm unfold: ", 29, 29);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normUnfold___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normUnfold___lam__0___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfold___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_10 = lp_aesop_Aesop_normUnfoldCore(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_10) == 0)
{
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_10;
}
else
{
lean_object* x_11; uint8_t x_12; uint8_t x_18; 
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_18 = l_Lean_Exception_isInterrupt(x_11);
if (x_18 == 0)
{
uint8_t x_19; 
lean_inc(x_11);
x_19 = l_Lean_Exception_isRuntime(x_11);
x_12 = x_19;
goto block_17;
}
else
{
x_12 = x_18;
goto block_17;
}
block_17:
{
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_dec_ref(x_10);
x_13 = lp_aesop_Aesop_normUnfold___lam__0___closed__1;
x_14 = l_Lean_Exception_toMessageData(x_11);
x_15 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
x_16 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_15, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_16;
}
else
{
lean_dec(x_11);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_normUnfold___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("unfold simp", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normUnfold___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normUnfold___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normUnfold___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(2);
x_2 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___boxed), 10, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfold___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_normUnfold___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfold(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_65; lean_object* x_66; lean_object* x_67; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_233; lean_object* x_262; uint8_t x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; lean_object* x_273; lean_object* x_274; lean_object* x_291; lean_object* x_292; lean_object* x_293; lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; uint8_t x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_328; lean_object* x_329; uint8_t x_330; lean_object* x_331; lean_object* x_332; lean_object* x_333; lean_object* x_334; lean_object* x_335; lean_object* x_336; lean_object* x_337; lean_object* x_338; lean_object* x_339; lean_object* x_340; lean_object* x_341; lean_object* x_342; lean_object* x_343; lean_object* x_344; lean_object* x_358; lean_object* x_359; uint8_t x_360; lean_object* x_361; lean_object* x_362; lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; lean_object* x_367; lean_object* x_390; uint8_t x_391; lean_object* x_392; lean_object* x_393; lean_object* x_394; lean_object* x_395; lean_object* x_396; uint8_t x_416; lean_object* x_417; lean_object* x_446; lean_object* x_451; uint8_t x_452; 
x_10 = lean_ctor_get(x_7, 2);
lean_inc(x_1);
x_11 = lean_alloc_closure((void*)(lp_aesop_Aesop_normUnfold___lam__0___boxed), 9, 1);
lean_closure_set(x_11, 0, x_1);
x_12 = lean_box(2);
x_451 = lp_aesop_Aesop_runNormRule___closed__0;
x_452 = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0_spec__0(x_10, x_451);
if (x_452 == 0)
{
lean_object* x_453; lean_object* x_454; lean_object* x_455; uint8_t x_456; 
x_453 = lp_aesop_Aesop_runNormRule___closed__1;
x_454 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_453, x_7);
x_455 = lean_ctor_get(x_454, 0);
lean_inc(x_455);
x_456 = lean_unbox(x_455);
lean_dec(x_455);
if (x_456 == 0)
{
lean_object* x_457; lean_object* x_458; lean_object* x_459; uint8_t x_460; 
lean_dec_ref(x_454);
x_457 = lp_aesop_Aesop_runNormRule___closed__2;
x_458 = lp_aesop_Lean_Option_get___at___00Aesop_runNormRule_spec__8(x_10, x_457);
x_459 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_460 = lean_string_dec_eq(x_458, x_459);
lean_dec_ref(x_458);
if (x_460 == 0)
{
uint8_t x_461; 
x_461 = 1;
x_416 = x_461;
x_417 = lean_box(0);
goto block_445;
}
else
{
x_233 = lean_box(0);
goto block_261;
}
}
else
{
x_446 = x_454;
goto block_450;
}
}
else
{
x_416 = x_452;
x_417 = lean_box(0);
goto block_445;
}
block_64:
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_17 = lean_io_mono_nanos_now();
x_18 = lean_st_ref_take(x_4);
x_19 = !lean_is_exclusive(x_18);
if (x_19 == 0)
{
lean_object* x_20; uint8_t x_21; 
x_20 = lean_ctor_get(x_18, 1);
x_21 = !lean_is_exclusive(x_20);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_22 = lean_ctor_get(x_20, 8);
x_23 = lean_nat_sub(x_17, x_14);
lean_dec(x_14);
lean_dec(x_17);
x_24 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_24, 0, x_12);
lean_ctor_set(x_24, 1, x_23);
lean_ctor_set_uint8(x_24, sizeof(void*)*2, x_13);
x_25 = lean_array_push(x_22, x_24);
lean_ctor_set(x_20, 8, x_25);
x_26 = lean_st_ref_set(x_4, x_18);
lean_dec(x_4);
x_27 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_27, 0, x_15);
return x_27;
}
else
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; 
x_28 = lean_ctor_get(x_20, 0);
x_29 = lean_ctor_get(x_20, 1);
x_30 = lean_ctor_get(x_20, 2);
x_31 = lean_ctor_get(x_20, 3);
x_32 = lean_ctor_get(x_20, 4);
x_33 = lean_ctor_get(x_20, 5);
x_34 = lean_ctor_get(x_20, 6);
x_35 = lean_ctor_get(x_20, 7);
x_36 = lean_ctor_get(x_20, 8);
x_37 = lean_ctor_get(x_20, 9);
lean_inc(x_37);
lean_inc(x_36);
lean_inc(x_35);
lean_inc(x_34);
lean_inc(x_33);
lean_inc(x_32);
lean_inc(x_31);
lean_inc(x_30);
lean_inc(x_29);
lean_inc(x_28);
lean_dec(x_20);
x_38 = lean_nat_sub(x_17, x_14);
lean_dec(x_14);
lean_dec(x_17);
x_39 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_39, 0, x_12);
lean_ctor_set(x_39, 1, x_38);
lean_ctor_set_uint8(x_39, sizeof(void*)*2, x_13);
x_40 = lean_array_push(x_36, x_39);
x_41 = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(x_41, 0, x_28);
lean_ctor_set(x_41, 1, x_29);
lean_ctor_set(x_41, 2, x_30);
lean_ctor_set(x_41, 3, x_31);
lean_ctor_set(x_41, 4, x_32);
lean_ctor_set(x_41, 5, x_33);
lean_ctor_set(x_41, 6, x_34);
lean_ctor_set(x_41, 7, x_35);
lean_ctor_set(x_41, 8, x_40);
lean_ctor_set(x_41, 9, x_37);
lean_ctor_set(x_18, 1, x_41);
x_42 = lean_st_ref_set(x_4, x_18);
lean_dec(x_4);
x_43 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_43, 0, x_15);
return x_43;
}
}
else
{
lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_44 = lean_ctor_get(x_18, 1);
x_45 = lean_ctor_get(x_18, 0);
lean_inc(x_44);
lean_inc(x_45);
lean_dec(x_18);
x_46 = lean_ctor_get(x_44, 0);
lean_inc(x_46);
x_47 = lean_ctor_get(x_44, 1);
lean_inc(x_47);
x_48 = lean_ctor_get(x_44, 2);
lean_inc(x_48);
x_49 = lean_ctor_get(x_44, 3);
lean_inc(x_49);
x_50 = lean_ctor_get(x_44, 4);
lean_inc(x_50);
x_51 = lean_ctor_get(x_44, 5);
lean_inc(x_51);
x_52 = lean_ctor_get(x_44, 6);
lean_inc(x_52);
x_53 = lean_ctor_get(x_44, 7);
lean_inc(x_53);
x_54 = lean_ctor_get(x_44, 8);
lean_inc_ref(x_54);
x_55 = lean_ctor_get(x_44, 9);
lean_inc_ref(x_55);
if (lean_is_exclusive(x_44)) {
 lean_ctor_release(x_44, 0);
 lean_ctor_release(x_44, 1);
 lean_ctor_release(x_44, 2);
 lean_ctor_release(x_44, 3);
 lean_ctor_release(x_44, 4);
 lean_ctor_release(x_44, 5);
 lean_ctor_release(x_44, 6);
 lean_ctor_release(x_44, 7);
 lean_ctor_release(x_44, 8);
 lean_ctor_release(x_44, 9);
 x_56 = x_44;
} else {
 lean_dec_ref(x_44);
 x_56 = lean_box(0);
}
x_57 = lean_nat_sub(x_17, x_14);
lean_dec(x_14);
lean_dec(x_17);
x_58 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_58, 0, x_12);
lean_ctor_set(x_58, 1, x_57);
lean_ctor_set_uint8(x_58, sizeof(void*)*2, x_13);
x_59 = lean_array_push(x_54, x_58);
if (lean_is_scalar(x_56)) {
 x_60 = lean_alloc_ctor(0, 10, 0);
} else {
 x_60 = x_56;
}
lean_ctor_set(x_60, 0, x_46);
lean_ctor_set(x_60, 1, x_47);
lean_ctor_set(x_60, 2, x_48);
lean_ctor_set(x_60, 3, x_49);
lean_ctor_set(x_60, 4, x_50);
lean_ctor_set(x_60, 5, x_51);
lean_ctor_set(x_60, 6, x_52);
lean_ctor_set(x_60, 7, x_53);
lean_ctor_set(x_60, 8, x_59);
lean_ctor_set(x_60, 9, x_55);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_45);
lean_ctor_set(x_61, 1, x_60);
x_62 = lean_st_ref_set(x_4, x_61);
lean_dec(x_4);
x_63 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_63, 0, x_15);
return x_63;
}
}
block_69:
{
if (lean_obj_tag(x_67) == 0)
{
lean_object* x_68; 
x_68 = lean_ctor_get(x_67, 0);
lean_inc(x_68);
lean_dec_ref(x_67);
x_13 = x_65;
x_14 = x_66;
x_15 = x_68;
x_16 = lean_box(0);
goto block_64;
}
else
{
lean_dec(x_66);
lean_dec(x_4);
return x_67;
}
}
block_115:
{
lean_dec(x_75);
lean_dec(x_74);
lean_dec_ref(x_73);
if (lean_obj_tag(x_72) == 0)
{
lean_object* x_81; uint8_t x_82; 
x_81 = lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___redArg(x_1, x_77);
lean_dec(x_1);
x_82 = !lean_is_exclusive(x_81);
if (x_82 == 0)
{
lean_object* x_83; uint8_t x_84; 
x_83 = lean_ctor_get(x_81, 0);
x_84 = lean_unbox(x_83);
lean_dec(x_83);
if (x_84 == 0)
{
lean_dec(x_79);
lean_dec_ref(x_78);
lean_dec(x_77);
lean_dec_ref(x_76);
lean_dec_ref(x_70);
lean_ctor_set(x_81, 0, x_71);
return x_81;
}
else
{
lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; uint8_t x_94; 
lean_free_object(x_81);
lean_dec(x_71);
x_85 = lp_aesop_Aesop_Check_name(x_70);
lean_dec_ref(x_70);
x_86 = l_Lean_MessageData_ofName(x_85);
x_87 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_88 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_88, 0, x_86);
lean_ctor_set(x_88, 1, x_87);
x_89 = lp_aesop_Aesop_normUnfold___closed__1;
x_90 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_90, 0, x_88);
lean_ctor_set(x_90, 1, x_89);
x_91 = lp_aesop_Aesop_checkSimp___closed__19;
x_92 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_92, 0, x_90);
lean_ctor_set(x_92, 1, x_91);
x_93 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_92, x_76, x_77, x_78, x_79);
lean_dec(x_79);
lean_dec_ref(x_78);
lean_dec(x_77);
lean_dec_ref(x_76);
x_94 = !lean_is_exclusive(x_93);
if (x_94 == 0)
{
return x_93;
}
else
{
lean_object* x_95; lean_object* x_96; 
x_95 = lean_ctor_get(x_93, 0);
lean_inc(x_95);
lean_dec(x_93);
x_96 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_96, 0, x_95);
return x_96;
}
}
}
else
{
lean_object* x_97; uint8_t x_98; 
x_97 = lean_ctor_get(x_81, 0);
lean_inc(x_97);
lean_dec(x_81);
x_98 = lean_unbox(x_97);
lean_dec(x_97);
if (x_98 == 0)
{
lean_object* x_99; 
lean_dec(x_79);
lean_dec_ref(x_78);
lean_dec(x_77);
lean_dec_ref(x_76);
lean_dec_ref(x_70);
x_99 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_99, 0, x_71);
return x_99;
}
else
{
lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; 
lean_dec(x_71);
x_100 = lp_aesop_Aesop_Check_name(x_70);
lean_dec_ref(x_70);
x_101 = l_Lean_MessageData_ofName(x_100);
x_102 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_103 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_103, 0, x_101);
lean_ctor_set(x_103, 1, x_102);
x_104 = lp_aesop_Aesop_normUnfold___closed__1;
x_105 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_105, 0, x_103);
lean_ctor_set(x_105, 1, x_104);
x_106 = lp_aesop_Aesop_checkSimp___closed__19;
x_107 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_107, 0, x_105);
lean_ctor_set(x_107, 1, x_106);
x_108 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_107, x_76, x_77, x_78, x_79);
lean_dec(x_79);
lean_dec_ref(x_78);
lean_dec(x_77);
lean_dec_ref(x_76);
x_109 = lean_ctor_get(x_108, 0);
lean_inc(x_109);
if (lean_is_exclusive(x_108)) {
 lean_ctor_release(x_108, 0);
 x_110 = x_108;
} else {
 lean_dec_ref(x_108);
 x_110 = lean_box(0);
}
if (lean_is_scalar(x_110)) {
 x_111 = lean_alloc_ctor(1, 1, 0);
} else {
 x_111 = x_110;
}
lean_ctor_set(x_111, 0, x_109);
return x_111;
}
}
}
else
{
uint8_t x_112; 
lean_dec(x_79);
lean_dec_ref(x_78);
lean_dec(x_77);
lean_dec_ref(x_76);
lean_dec_ref(x_70);
lean_dec(x_1);
x_112 = !lean_is_exclusive(x_72);
if (x_112 == 0)
{
lean_object* x_113; 
x_113 = lean_ctor_get(x_72, 0);
lean_dec(x_113);
lean_ctor_set_tag(x_72, 0);
lean_ctor_set(x_72, 0, x_71);
return x_72;
}
else
{
lean_object* x_114; 
lean_dec(x_72);
x_114 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_114, 0, x_71);
return x_114;
}
}
}
block_150:
{
uint8_t x_129; 
x_129 = l_Array_isEmpty___redArg(x_128);
lean_dec_ref(x_128);
if (x_129 == 0)
{
lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; size_t x_138; size_t x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; uint8_t x_147; 
lean_dec(x_127);
lean_dec(x_126);
lean_dec(x_124);
lean_dec_ref(x_123);
lean_dec(x_122);
lean_dec(x_1);
x_130 = lp_aesop_Aesop_Check_name(x_125);
lean_dec_ref(x_125);
x_131 = l_Lean_MessageData_ofName(x_130);
x_132 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_133 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_133, 0, x_131);
lean_ctor_set(x_133, 1, x_132);
x_134 = lp_aesop_Aesop_normUnfold___closed__1;
x_135 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_135, 0, x_133);
lean_ctor_set(x_135, 1, x_134);
x_136 = lp_aesop_Aesop_checkSimp___closed__21;
x_137 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_137, 0, x_135);
lean_ctor_set(x_137, 1, x_136);
x_138 = lean_array_size(x_121);
x_139 = 0;
x_140 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(x_138, x_139, x_121);
x_141 = lean_array_to_list(x_140);
x_142 = lean_box(0);
x_143 = lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(x_141, x_142);
x_144 = l_Lean_MessageData_ofList(x_143);
x_145 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_145, 0, x_137);
lean_ctor_set(x_145, 1, x_144);
x_146 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_145, x_119, x_116, x_118, x_120);
lean_dec(x_120);
lean_dec_ref(x_118);
lean_dec(x_116);
lean_dec_ref(x_119);
x_147 = !lean_is_exclusive(x_146);
if (x_147 == 0)
{
return x_146;
}
else
{
lean_object* x_148; lean_object* x_149; 
x_148 = lean_ctor_get(x_146, 0);
lean_inc(x_148);
lean_dec(x_146);
x_149 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_149, 0, x_148);
return x_149;
}
}
else
{
lean_dec_ref(x_121);
x_70 = x_125;
x_71 = x_126;
x_72 = x_127;
x_73 = x_123;
x_74 = x_124;
x_75 = x_122;
x_76 = x_119;
x_77 = x_116;
x_78 = x_118;
x_79 = x_120;
x_80 = lean_box(0);
goto block_115;
}
}
block_178:
{
lean_object* x_166; 
lean_inc(x_164);
lean_inc_ref(x_163);
lean_inc(x_162);
lean_inc_ref(x_161);
x_166 = lp_batteries_Lean_Meta_getAssignedExprMVars(x_157, x_151, x_161, x_162, x_163, x_164);
if (lean_obj_tag(x_166) == 0)
{
lean_object* x_167; lean_object* x_168; lean_object* x_169; uint8_t x_170; 
x_167 = lean_ctor_get(x_166, 0);
lean_inc(x_167);
lean_dec_ref(x_166);
x_168 = lean_array_get_size(x_167);
x_169 = lean_mk_empty_array_with_capacity(x_155);
x_170 = lean_nat_dec_lt(x_155, x_168);
if (x_170 == 0)
{
lean_dec(x_167);
x_116 = x_162;
x_117 = lean_box(0);
x_118 = x_163;
x_119 = x_161;
x_120 = x_164;
x_121 = x_152;
x_122 = x_160;
x_123 = x_158;
x_124 = x_159;
x_125 = x_153;
x_126 = x_154;
x_127 = x_156;
x_128 = x_169;
goto block_150;
}
else
{
uint8_t x_171; 
x_171 = lean_nat_dec_le(x_168, x_168);
if (x_171 == 0)
{
lean_dec(x_167);
x_116 = x_162;
x_117 = lean_box(0);
x_118 = x_163;
x_119 = x_161;
x_120 = x_164;
x_121 = x_152;
x_122 = x_160;
x_123 = x_158;
x_124 = x_159;
x_125 = x_153;
x_126 = x_154;
x_127 = x_156;
x_128 = x_169;
goto block_150;
}
else
{
size_t x_172; size_t x_173; lean_object* x_174; 
x_172 = 0;
x_173 = lean_usize_of_nat(x_168);
x_174 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__4(x_1, x_167, x_172, x_173, x_169);
lean_dec(x_167);
x_116 = x_162;
x_117 = lean_box(0);
x_118 = x_163;
x_119 = x_161;
x_120 = x_164;
x_121 = x_152;
x_122 = x_160;
x_123 = x_158;
x_124 = x_159;
x_125 = x_153;
x_126 = x_154;
x_127 = x_156;
x_128 = x_174;
goto block_150;
}
}
}
else
{
uint8_t x_175; 
lean_dec(x_164);
lean_dec_ref(x_163);
lean_dec(x_162);
lean_dec_ref(x_161);
lean_dec(x_160);
lean_dec(x_159);
lean_dec_ref(x_158);
lean_dec(x_156);
lean_dec(x_154);
lean_dec_ref(x_153);
lean_dec_ref(x_152);
lean_dec(x_1);
x_175 = !lean_is_exclusive(x_166);
if (x_175 == 0)
{
return x_166;
}
else
{
lean_object* x_176; lean_object* x_177; 
x_176 = lean_ctor_get(x_166, 0);
lean_inc(x_176);
lean_dec(x_166);
x_177 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_177, 0, x_176);
return x_177;
}
}
}
block_208:
{
uint8_t x_187; 
x_187 = l_Array_isEmpty___redArg(x_186);
if (x_187 == 0)
{
lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; size_t x_196; size_t x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; uint8_t x_205; 
lean_dec_ref(x_185);
lean_dec(x_183);
lean_dec(x_182);
lean_dec_ref(x_179);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_188 = lp_aesop_Aesop_Check_name(x_181);
lean_dec_ref(x_181);
x_189 = l_Lean_MessageData_ofName(x_188);
x_190 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_191 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_191, 0, x_189);
lean_ctor_set(x_191, 1, x_190);
x_192 = lp_aesop_Aesop_normUnfold___closed__1;
x_193 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_193, 0, x_191);
lean_ctor_set(x_193, 1, x_192);
x_194 = lp_aesop_Aesop_checkSimp___closed__34;
x_195 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_195, 0, x_193);
lean_ctor_set(x_195, 1, x_194);
x_196 = lean_array_size(x_186);
x_197 = 0;
x_198 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(x_196, x_197, x_186);
x_199 = lean_array_to_list(x_198);
x_200 = lean_box(0);
x_201 = lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(x_199, x_200);
x_202 = l_Lean_MessageData_ofList(x_201);
x_203 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_203, 0, x_195);
lean_ctor_set(x_203, 1, x_202);
x_204 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_203, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
x_205 = !lean_is_exclusive(x_204);
if (x_205 == 0)
{
return x_204;
}
else
{
lean_object* x_206; lean_object* x_207; 
x_206 = lean_ctor_get(x_204, 0);
lean_inc(x_206);
lean_dec(x_204);
x_207 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_207, 0, x_206);
return x_207;
}
}
else
{
x_151 = x_179;
x_152 = x_186;
x_153 = x_181;
x_154 = x_182;
x_155 = x_184;
x_156 = x_183;
x_157 = x_185;
x_158 = x_2;
x_159 = x_3;
x_160 = x_4;
x_161 = x_5;
x_162 = x_6;
x_163 = x_7;
x_164 = x_8;
x_165 = lean_box(0);
goto block_178;
}
}
block_232:
{
lean_object* x_214; 
x_214 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_214) == 0)
{
lean_object* x_215; lean_object* x_216; 
x_215 = lean_ctor_get(x_214, 0);
lean_inc(x_215);
lean_dec_ref(x_214);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_215);
lean_inc_ref(x_212);
x_216 = lp_batteries_Lean_Meta_getIntroducedExprMVars(x_212, x_215, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_216) == 0)
{
lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; uint8_t x_221; 
x_217 = lean_ctor_get(x_216, 0);
lean_inc(x_217);
lean_dec_ref(x_216);
x_218 = lean_unsigned_to_nat(0u);
x_219 = lean_array_get_size(x_217);
x_220 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_221 = lean_nat_dec_lt(x_218, x_219);
if (x_221 == 0)
{
lean_dec(x_217);
x_179 = x_215;
x_180 = lean_box(0);
x_181 = x_210;
x_182 = x_211;
x_183 = x_213;
x_184 = x_218;
x_185 = x_212;
x_186 = x_220;
goto block_208;
}
else
{
uint8_t x_222; 
x_222 = lean_nat_dec_le(x_219, x_219);
if (x_222 == 0)
{
lean_dec(x_217);
x_179 = x_215;
x_180 = lean_box(0);
x_181 = x_210;
x_182 = x_211;
x_183 = x_213;
x_184 = x_218;
x_185 = x_212;
x_186 = x_220;
goto block_208;
}
else
{
size_t x_223; size_t x_224; lean_object* x_225; 
x_223 = 0;
x_224 = lean_usize_of_nat(x_219);
x_225 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__5(x_213, x_217, x_223, x_224, x_220);
lean_dec(x_217);
x_179 = x_215;
x_180 = lean_box(0);
x_181 = x_210;
x_182 = x_211;
x_183 = x_213;
x_184 = x_218;
x_185 = x_212;
x_186 = x_225;
goto block_208;
}
}
}
else
{
uint8_t x_226; 
lean_dec(x_215);
lean_dec(x_213);
lean_dec_ref(x_212);
lean_dec(x_211);
lean_dec_ref(x_210);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_226 = !lean_is_exclusive(x_216);
if (x_226 == 0)
{
return x_216;
}
else
{
lean_object* x_227; lean_object* x_228; 
x_227 = lean_ctor_get(x_216, 0);
lean_inc(x_227);
lean_dec(x_216);
x_228 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_228, 0, x_227);
return x_228;
}
}
}
else
{
uint8_t x_229; 
lean_dec(x_213);
lean_dec_ref(x_212);
lean_dec(x_211);
lean_dec_ref(x_210);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_229 = !lean_is_exclusive(x_214);
if (x_229 == 0)
{
return x_214;
}
else
{
lean_object* x_230; lean_object* x_231; 
x_230 = lean_ctor_get(x_214, 0);
lean_inc(x_230);
lean_dec(x_214);
x_231 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_231, 0, x_230);
return x_231;
}
}
}
block_261:
{
lean_object* x_234; lean_object* x_235; lean_object* x_236; uint8_t x_237; 
x_234 = lp_aesop_Aesop_runNormRuleTac___closed__8;
x_235 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(x_234, x_7);
x_236 = lean_ctor_get(x_235, 0);
lean_inc(x_236);
lean_dec_ref(x_235);
x_237 = lean_unbox(x_236);
if (x_237 == 0)
{
lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; uint8_t x_242; lean_object* x_243; lean_object* x_244; 
lean_dec(x_236);
lean_dec(x_1);
x_238 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_239 = lean_ctor_get(x_238, 0);
lean_inc(x_239);
x_240 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimp___lam__1___boxed), 10, 2);
lean_closure_set(x_240, 0, x_11);
lean_closure_set(x_240, 1, x_238);
x_241 = lp_aesop_Aesop_normUnfold___closed__2;
x_242 = 1;
x_243 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_244 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_239, x_241, x_240, x_242, x_243, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_244;
}
else
{
lean_object* x_245; 
x_245 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_245) == 0)
{
lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; uint8_t x_252; lean_object* x_253; 
x_246 = lean_ctor_get(x_245, 0);
lean_inc(x_246);
lean_dec_ref(x_245);
x_247 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_248 = lean_ctor_get(x_247, 0);
lean_inc(x_248);
x_249 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimp___lam__1___boxed), 10, 2);
lean_closure_set(x_249, 0, x_11);
lean_closure_set(x_249, 1, x_247);
x_250 = lp_aesop_Aesop_normUnfold___closed__2;
x_251 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_252 = lean_unbox(x_236);
lean_dec(x_236);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_253 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_248, x_250, x_249, x_252, x_251, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_253) == 0)
{
lean_object* x_254; 
x_254 = lean_ctor_get(x_253, 0);
lean_inc(x_254);
lean_dec_ref(x_253);
if (lean_obj_tag(x_254) == 0)
{
lean_object* x_255; 
x_255 = lean_box(0);
x_209 = lean_box(0);
x_210 = x_234;
x_211 = x_254;
x_212 = x_246;
x_213 = x_255;
goto block_232;
}
else
{
lean_object* x_256; lean_object* x_257; 
x_256 = lean_ctor_get(x_254, 0);
x_257 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_256);
x_209 = lean_box(0);
x_210 = x_234;
x_211 = x_254;
x_212 = x_246;
x_213 = x_257;
goto block_232;
}
}
else
{
lean_dec(x_246);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_253;
}
}
else
{
uint8_t x_258; 
lean_dec(x_236);
lean_dec_ref(x_11);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_258 = !lean_is_exclusive(x_245);
if (x_258 == 0)
{
return x_245;
}
else
{
lean_object* x_259; lean_object* x_260; 
x_259 = lean_ctor_get(x_245, 0);
lean_inc(x_259);
lean_dec(x_245);
x_260 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_260, 0, x_259);
return x_260;
}
}
}
}
block_290:
{
lean_dec(x_269);
lean_dec(x_268);
lean_dec_ref(x_267);
if (lean_obj_tag(x_265) == 0)
{
lean_object* x_275; lean_object* x_276; uint8_t x_277; 
x_275 = lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___redArg(x_1, x_271);
lean_dec(x_1);
x_276 = lean_ctor_get(x_275, 0);
lean_inc(x_276);
lean_dec_ref(x_275);
x_277 = lean_unbox(x_276);
lean_dec(x_276);
if (x_277 == 0)
{
lean_dec(x_273);
lean_dec_ref(x_272);
lean_dec(x_271);
lean_dec_ref(x_270);
lean_dec_ref(x_264);
x_13 = x_263;
x_14 = x_266;
x_15 = x_262;
x_16 = lean_box(0);
goto block_64;
}
else
{
lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; uint8_t x_287; 
lean_dec(x_266);
lean_dec(x_262);
lean_dec(x_4);
x_278 = lp_aesop_Aesop_Check_name(x_264);
lean_dec_ref(x_264);
x_279 = l_Lean_MessageData_ofName(x_278);
x_280 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_281 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_281, 0, x_279);
lean_ctor_set(x_281, 1, x_280);
x_282 = lp_aesop_Aesop_normUnfold___closed__1;
x_283 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_283, 0, x_281);
lean_ctor_set(x_283, 1, x_282);
x_284 = lp_aesop_Aesop_checkSimp___closed__19;
x_285 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_285, 0, x_283);
lean_ctor_set(x_285, 1, x_284);
x_286 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_285, x_270, x_271, x_272, x_273);
lean_dec(x_273);
lean_dec_ref(x_272);
lean_dec(x_271);
lean_dec_ref(x_270);
x_287 = !lean_is_exclusive(x_286);
if (x_287 == 0)
{
return x_286;
}
else
{
lean_object* x_288; lean_object* x_289; 
x_288 = lean_ctor_get(x_286, 0);
lean_inc(x_288);
lean_dec(x_286);
x_289 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_289, 0, x_288);
return x_289;
}
}
}
else
{
lean_dec(x_273);
lean_dec_ref(x_272);
lean_dec(x_271);
lean_dec_ref(x_270);
lean_dec_ref(x_265);
lean_dec_ref(x_264);
lean_dec(x_1);
x_13 = x_263;
x_14 = x_266;
x_15 = x_262;
x_16 = lean_box(0);
goto block_64;
}
}
block_327:
{
uint8_t x_306; 
x_306 = l_Array_isEmpty___redArg(x_305);
lean_dec_ref(x_305);
if (x_306 == 0)
{
lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; lean_object* x_313; lean_object* x_314; size_t x_315; size_t x_316; lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_323; uint8_t x_324; 
lean_dec(x_299);
lean_dec(x_298);
lean_dec(x_296);
lean_dec(x_295);
lean_dec(x_294);
lean_dec_ref(x_293);
lean_dec(x_4);
lean_dec(x_1);
x_307 = lp_aesop_Aesop_Check_name(x_302);
lean_dec_ref(x_302);
x_308 = l_Lean_MessageData_ofName(x_307);
x_309 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_310 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_310, 0, x_308);
lean_ctor_set(x_310, 1, x_309);
x_311 = lp_aesop_Aesop_normUnfold___closed__1;
x_312 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_312, 0, x_310);
lean_ctor_set(x_312, 1, x_311);
x_313 = lp_aesop_Aesop_checkSimp___closed__21;
x_314 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_314, 0, x_312);
lean_ctor_set(x_314, 1, x_313);
x_315 = lean_array_size(x_300);
x_316 = 0;
x_317 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(x_315, x_316, x_300);
x_318 = lean_array_to_list(x_317);
x_319 = lean_box(0);
x_320 = lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(x_318, x_319);
x_321 = l_Lean_MessageData_ofList(x_320);
x_322 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_322, 0, x_314);
lean_ctor_set(x_322, 1, x_321);
x_323 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_322, x_292, x_291, x_303, x_297);
lean_dec(x_297);
lean_dec_ref(x_303);
lean_dec(x_291);
lean_dec_ref(x_292);
x_324 = !lean_is_exclusive(x_323);
if (x_324 == 0)
{
return x_323;
}
else
{
lean_object* x_325; lean_object* x_326; 
x_325 = lean_ctor_get(x_323, 0);
lean_inc(x_325);
lean_dec(x_323);
x_326 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_326, 0, x_325);
return x_326;
}
}
else
{
lean_dec_ref(x_300);
x_262 = x_299;
x_263 = x_301;
x_264 = x_302;
x_265 = x_295;
x_266 = x_296;
x_267 = x_293;
x_268 = x_298;
x_269 = x_294;
x_270 = x_292;
x_271 = x_291;
x_272 = x_303;
x_273 = x_297;
x_274 = lean_box(0);
goto block_290;
}
}
block_357:
{
lean_object* x_345; 
lean_inc(x_343);
lean_inc_ref(x_342);
lean_inc(x_341);
lean_inc_ref(x_340);
x_345 = lp_batteries_Lean_Meta_getAssignedExprMVars(x_334, x_332, x_340, x_341, x_342, x_343);
if (lean_obj_tag(x_345) == 0)
{
lean_object* x_346; lean_object* x_347; lean_object* x_348; uint8_t x_349; 
x_346 = lean_ctor_get(x_345, 0);
lean_inc(x_346);
lean_dec_ref(x_345);
x_347 = lean_array_get_size(x_346);
x_348 = lean_mk_empty_array_with_capacity(x_329);
x_349 = lean_nat_dec_lt(x_329, x_347);
if (x_349 == 0)
{
lean_dec(x_346);
x_291 = x_341;
x_292 = x_340;
x_293 = x_337;
x_294 = x_339;
x_295 = x_335;
x_296 = x_336;
x_297 = x_343;
x_298 = x_338;
x_299 = x_328;
x_300 = x_331;
x_301 = x_330;
x_302 = x_333;
x_303 = x_342;
x_304 = lean_box(0);
x_305 = x_348;
goto block_327;
}
else
{
uint8_t x_350; 
x_350 = lean_nat_dec_le(x_347, x_347);
if (x_350 == 0)
{
lean_dec(x_346);
x_291 = x_341;
x_292 = x_340;
x_293 = x_337;
x_294 = x_339;
x_295 = x_335;
x_296 = x_336;
x_297 = x_343;
x_298 = x_338;
x_299 = x_328;
x_300 = x_331;
x_301 = x_330;
x_302 = x_333;
x_303 = x_342;
x_304 = lean_box(0);
x_305 = x_348;
goto block_327;
}
else
{
size_t x_351; size_t x_352; lean_object* x_353; 
x_351 = 0;
x_352 = lean_usize_of_nat(x_347);
x_353 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__4(x_1, x_346, x_351, x_352, x_348);
lean_dec(x_346);
x_291 = x_341;
x_292 = x_340;
x_293 = x_337;
x_294 = x_339;
x_295 = x_335;
x_296 = x_336;
x_297 = x_343;
x_298 = x_338;
x_299 = x_328;
x_300 = x_331;
x_301 = x_330;
x_302 = x_333;
x_303 = x_342;
x_304 = lean_box(0);
x_305 = x_353;
goto block_327;
}
}
}
else
{
uint8_t x_354; 
lean_dec(x_343);
lean_dec_ref(x_342);
lean_dec(x_341);
lean_dec_ref(x_340);
lean_dec(x_339);
lean_dec(x_338);
lean_dec_ref(x_337);
lean_dec(x_336);
lean_dec(x_335);
lean_dec_ref(x_333);
lean_dec_ref(x_331);
lean_dec(x_328);
lean_dec(x_4);
lean_dec(x_1);
x_354 = !lean_is_exclusive(x_345);
if (x_354 == 0)
{
return x_345;
}
else
{
lean_object* x_355; lean_object* x_356; 
x_355 = lean_ctor_get(x_345, 0);
lean_inc(x_355);
lean_dec(x_345);
x_356 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_356, 0, x_355);
return x_356;
}
}
}
block_389:
{
uint8_t x_368; 
x_368 = l_Array_isEmpty___redArg(x_367);
if (x_368 == 0)
{
lean_object* x_369; lean_object* x_370; lean_object* x_371; lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; lean_object* x_376; size_t x_377; size_t x_378; lean_object* x_379; lean_object* x_380; lean_object* x_381; lean_object* x_382; lean_object* x_383; lean_object* x_384; lean_object* x_385; uint8_t x_386; 
lean_dec(x_366);
lean_dec(x_364);
lean_dec_ref(x_362);
lean_dec_ref(x_361);
lean_dec(x_358);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_369 = lp_aesop_Aesop_Check_name(x_363);
lean_dec_ref(x_363);
x_370 = l_Lean_MessageData_ofName(x_369);
x_371 = lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3;
x_372 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_372, 0, x_370);
lean_ctor_set(x_372, 1, x_371);
x_373 = lp_aesop_Aesop_normUnfold___closed__1;
x_374 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_374, 0, x_372);
lean_ctor_set(x_374, 1, x_373);
x_375 = lp_aesop_Aesop_checkSimp___closed__34;
x_376 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_376, 0, x_374);
lean_ctor_set(x_376, 1, x_375);
x_377 = lean_array_size(x_367);
x_378 = 0;
x_379 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_normSimp_spec__2(x_377, x_378, x_367);
x_380 = lean_array_to_list(x_379);
x_381 = lean_box(0);
x_382 = lp_aesop_List_mapTR_loop___at___00Aesop_normSimp_spec__3(x_380, x_381);
x_383 = l_Lean_MessageData_ofList(x_382);
x_384 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_384, 0, x_376);
lean_ctor_set(x_384, 1, x_383);
x_385 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_384, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
x_386 = !lean_is_exclusive(x_385);
if (x_386 == 0)
{
return x_385;
}
else
{
lean_object* x_387; lean_object* x_388; 
x_387 = lean_ctor_get(x_385, 0);
lean_inc(x_387);
lean_dec(x_385);
x_388 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_388, 0, x_387);
return x_388;
}
}
else
{
lean_inc(x_4);
x_328 = x_358;
x_329 = x_359;
x_330 = x_360;
x_331 = x_367;
x_332 = x_361;
x_333 = x_363;
x_334 = x_362;
x_335 = x_364;
x_336 = x_366;
x_337 = x_2;
x_338 = x_3;
x_339 = x_4;
x_340 = x_5;
x_341 = x_6;
x_342 = x_7;
x_343 = x_8;
x_344 = lean_box(0);
goto block_357;
}
}
block_415:
{
lean_object* x_397; 
x_397 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_397) == 0)
{
lean_object* x_398; lean_object* x_399; 
x_398 = lean_ctor_get(x_397, 0);
lean_inc(x_398);
lean_dec_ref(x_397);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_398);
lean_inc_ref(x_392);
x_399 = lp_batteries_Lean_Meta_getIntroducedExprMVars(x_392, x_398, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_399) == 0)
{
lean_object* x_400; lean_object* x_401; lean_object* x_402; lean_object* x_403; uint8_t x_404; 
x_400 = lean_ctor_get(x_399, 0);
lean_inc(x_400);
lean_dec_ref(x_399);
x_401 = lean_unsigned_to_nat(0u);
x_402 = lean_array_get_size(x_400);
x_403 = lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0;
x_404 = lean_nat_dec_lt(x_401, x_402);
if (x_404 == 0)
{
lean_dec(x_400);
x_358 = x_390;
x_359 = x_401;
x_360 = x_391;
x_361 = x_398;
x_362 = x_392;
x_363 = x_393;
x_364 = x_396;
x_365 = lean_box(0);
x_366 = x_395;
x_367 = x_403;
goto block_389;
}
else
{
uint8_t x_405; 
x_405 = lean_nat_dec_le(x_402, x_402);
if (x_405 == 0)
{
lean_dec(x_400);
x_358 = x_390;
x_359 = x_401;
x_360 = x_391;
x_361 = x_398;
x_362 = x_392;
x_363 = x_393;
x_364 = x_396;
x_365 = lean_box(0);
x_366 = x_395;
x_367 = x_403;
goto block_389;
}
else
{
size_t x_406; size_t x_407; lean_object* x_408; 
x_406 = 0;
x_407 = lean_usize_of_nat(x_402);
x_408 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_normSimp_spec__5(x_396, x_400, x_406, x_407, x_403);
lean_dec(x_400);
x_358 = x_390;
x_359 = x_401;
x_360 = x_391;
x_361 = x_398;
x_362 = x_392;
x_363 = x_393;
x_364 = x_396;
x_365 = lean_box(0);
x_366 = x_395;
x_367 = x_408;
goto block_389;
}
}
}
else
{
uint8_t x_409; 
lean_dec(x_398);
lean_dec(x_396);
lean_dec(x_395);
lean_dec_ref(x_393);
lean_dec_ref(x_392);
lean_dec(x_390);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_409 = !lean_is_exclusive(x_399);
if (x_409 == 0)
{
return x_399;
}
else
{
lean_object* x_410; lean_object* x_411; 
x_410 = lean_ctor_get(x_399, 0);
lean_inc(x_410);
lean_dec(x_399);
x_411 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_411, 0, x_410);
return x_411;
}
}
}
else
{
uint8_t x_412; 
lean_dec(x_396);
lean_dec(x_395);
lean_dec_ref(x_393);
lean_dec_ref(x_392);
lean_dec(x_390);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_412 = !lean_is_exclusive(x_397);
if (x_412 == 0)
{
return x_397;
}
else
{
lean_object* x_413; lean_object* x_414; 
x_413 = lean_ctor_get(x_397, 0);
lean_inc(x_413);
lean_dec(x_397);
x_414 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_414, 0, x_413);
return x_414;
}
}
}
block_445:
{
lean_object* x_418; lean_object* x_419; lean_object* x_420; lean_object* x_421; uint8_t x_422; 
x_418 = lean_io_mono_nanos_now();
x_419 = lp_aesop_Aesop_runNormRuleTac___closed__8;
x_420 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runNormRuleTac_spec__3___redArg(x_419, x_7);
x_421 = lean_ctor_get(x_420, 0);
lean_inc(x_421);
lean_dec_ref(x_420);
x_422 = lean_unbox(x_421);
if (x_422 == 0)
{
lean_object* x_423; lean_object* x_424; lean_object* x_425; lean_object* x_426; lean_object* x_427; lean_object* x_428; 
lean_dec(x_421);
lean_dec(x_1);
x_423 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_424 = lean_ctor_get(x_423, 0);
lean_inc(x_424);
x_425 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimp___lam__1___boxed), 10, 2);
lean_closure_set(x_425, 0, x_11);
lean_closure_set(x_425, 1, x_423);
x_426 = lp_aesop_Aesop_normUnfold___closed__2;
x_427 = lp_aesop_Aesop_withNormTraceNode___closed__43;
lean_inc(x_4);
x_428 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_424, x_426, x_425, x_416, x_427, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
x_65 = x_416;
x_66 = x_418;
x_67 = x_428;
goto block_69;
}
else
{
lean_object* x_429; 
x_429 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_429) == 0)
{
lean_object* x_430; lean_object* x_431; lean_object* x_432; lean_object* x_433; lean_object* x_434; lean_object* x_435; uint8_t x_436; lean_object* x_437; 
x_430 = lean_ctor_get(x_429, 0);
lean_inc(x_430);
lean_dec_ref(x_429);
x_431 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_432 = lean_ctor_get(x_431, 0);
lean_inc(x_432);
x_433 = lean_alloc_closure((void*)(lp_aesop_Aesop_normSimp___lam__1___boxed), 10, 2);
lean_closure_set(x_433, 0, x_11);
lean_closure_set(x_433, 1, x_431);
x_434 = lp_aesop_Aesop_normUnfold___closed__2;
x_435 = lp_aesop_Aesop_withNormTraceNode___closed__43;
x_436 = lean_unbox(x_421);
lean_dec(x_421);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_437 = lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg(x_432, x_434, x_433, x_436, x_435, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_437) == 0)
{
lean_object* x_438; 
x_438 = lean_ctor_get(x_437, 0);
lean_inc(x_438);
lean_dec_ref(x_437);
if (lean_obj_tag(x_438) == 0)
{
lean_object* x_439; 
x_439 = lean_box(0);
x_390 = x_438;
x_391 = x_416;
x_392 = x_430;
x_393 = x_419;
x_394 = lean_box(0);
x_395 = x_418;
x_396 = x_439;
goto block_415;
}
else
{
lean_object* x_440; lean_object* x_441; 
x_440 = lean_ctor_get(x_438, 0);
x_441 = lp_aesop_Aesop_NormRuleResult_newGoal_x3f(x_440);
x_390 = x_438;
x_391 = x_416;
x_392 = x_430;
x_393 = x_419;
x_394 = lean_box(0);
x_395 = x_418;
x_396 = x_441;
goto block_415;
}
}
else
{
lean_dec(x_430);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_65 = x_416;
x_66 = x_418;
x_67 = x_437;
goto block_69;
}
}
else
{
uint8_t x_442; 
lean_dec(x_421);
lean_dec(x_418);
lean_dec_ref(x_11);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_442 = !lean_is_exclusive(x_429);
if (x_442 == 0)
{
return x_429;
}
else
{
lean_object* x_443; lean_object* x_444; 
x_443 = lean_ctor_get(x_429, 0);
lean_inc(x_443);
lean_dec(x_429);
x_444 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_444, 0, x_443);
return x_444;
}
}
}
}
block_450:
{
lean_object* x_447; uint8_t x_448; 
x_447 = lean_ctor_get(x_446, 0);
lean_inc(x_447);
lean_dec_ref(x_446);
x_448 = lean_unbox(x_447);
if (x_448 == 0)
{
lean_dec(x_447);
x_233 = lean_box(0);
goto block_261;
}
else
{
uint8_t x_449; 
x_449 = lean_unbox(x_447);
lean_dec(x_447);
x_416 = x_449;
x_417 = lean_box(0);
goto block_445;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Lean_MVarId_isAssigned___at___00Aesop_normUnfold_spec__0___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normUnfold___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_normUnfold(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorIdx(lean_object* x_1) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
case 1:
{
lean_object* x_3; 
x_3 = lean_unsigned_to_nat(1u);
return x_3;
}
default: 
{
lean_object* x_4; 
x_4 = lean_unsigned_to_nat(2u);
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorIdx___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NormSeqResult_ctorIdx(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_2, x_3);
return x_4;
}
case 1:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_2(x_2, x_5, x_6);
return x_7;
}
default: 
{
return x_2;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_NormSeqResult_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_NormSeqResult_ctorElim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_proved_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_NormSeqResult_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_proved_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NormSeqResult_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_changed_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_NormSeqResult_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_changed_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NormSeqResult_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_unchanged_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_NormSeqResult_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSeqResult_unchanged_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NormSeqResult_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_NormRuleResult_toNormSeqResult___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormRuleResult_toNormSeqResult(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 0);
lean_ctor_set(x_2, 0, x_1);
x_5 = lp_aesop_Aesop_NormRuleResult_toNormSeqResult___closed__0;
x_6 = lean_array_push(x_5, x_2);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_4);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_8 = lean_ctor_get(x_2, 0);
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_2);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_1);
lean_ctor_set(x_10, 1, x_9);
x_11 = lp_aesop_Aesop_NormRuleResult_toNormSeqResult___closed__0;
x_12 = lean_array_push(x_11, x_10);
x_13 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_13, 0, x_8);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
else
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_2);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = lean_ctor_get(x_2, 0);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_1);
lean_ctor_set(x_16, 1, x_15);
x_17 = lp_aesop_Aesop_NormRuleResult_toNormSeqResult___closed__0;
x_18 = lean_array_push(x_17, x_16);
lean_ctor_set_tag(x_2, 0);
lean_ctor_set(x_2, 0, x_18);
return x_2;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_19 = lean_ctor_get(x_2, 0);
lean_inc(x_19);
lean_dec(x_2);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_1);
lean_ctor_set(x_20, 1, x_19);
x_21 = lp_aesop_Aesop_NormRuleResult_toNormSeqResult___closed__0;
x_22 = lean_array_push(x_21, x_20);
x_23 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_23, 0, x_22);
return x_23;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optNormRuleResultToNormSeqResult(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_box(2);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec(x_3);
x_6 = lp_aesop_Aesop_NormRuleResult_toNormSeqResult(x_4, x_5);
return x_6;
}
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_aesop_Aesop_runNormSteps___redArg___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_runNormSteps___redArg___closed__2;
x_2 = lp_aesop_Aesop_runNormSteps___redArg___closed__1;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_runNormSteps___redArg___closed__3;
x_2 = lp_aesop_Aesop_runNormSteps___redArg___closed__1;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_runNormSteps___redArg___closed__4;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormSteps_spec__0(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_11; 
x_11 = lean_usize_dec_lt(x_4, x_3);
if (x_11 == 0)
{
lean_dec(x_1);
return x_5;
}
else
{
uint8_t x_12; 
x_12 = !lean_is_exclusive(x_5);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_13 = lean_ctor_get(x_5, 0);
x_14 = lean_ctor_get(x_5, 1);
x_15 = lean_array_uget(x_2, x_4);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
x_17 = lean_ctor_get(x_16, 3);
lean_inc(x_17);
lean_dec(x_16);
lean_inc(x_1);
x_18 = lean_nat_to_int(x_1);
x_19 = lean_int_dec_lt(x_17, x_18);
lean_dec(x_18);
lean_dec(x_17);
if (x_19 == 0)
{
lean_object* x_20; 
x_20 = lean_array_push(x_14, x_15);
lean_ctor_set(x_5, 1, x_20);
x_6 = x_5;
goto block_10;
}
else
{
lean_object* x_21; 
x_21 = lean_array_push(x_13, x_15);
lean_ctor_set(x_5, 0, x_21);
x_6 = x_5;
goto block_10;
}
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; 
x_22 = lean_ctor_get(x_5, 0);
x_23 = lean_ctor_get(x_5, 1);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_5);
x_24 = lean_array_uget(x_2, x_4);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_25, 3);
lean_inc(x_26);
lean_dec(x_25);
lean_inc(x_1);
x_27 = lean_nat_to_int(x_1);
x_28 = lean_int_dec_lt(x_26, x_27);
lean_dec(x_27);
lean_dec(x_26);
if (x_28 == 0)
{
lean_object* x_29; lean_object* x_30; 
x_29 = lean_array_push(x_23, x_24);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_22);
lean_ctor_set(x_30, 1, x_29);
x_6 = x_30;
goto block_10;
}
else
{
lean_object* x_31; lean_object* x_32; 
x_31 = lean_array_push(x_22, x_24);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_23);
x_6 = x_32;
goto block_10;
}
}
}
block_10:
{
size_t x_7; size_t x_8; 
x_7 = 1;
x_8 = lean_usize_add(x_4, x_7);
x_4 = x_8;
x_5 = x_6;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; uint8_t x_37; uint8_t x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; 
x_17 = lean_ctor_get(x_8, 1);
lean_inc(x_17);
if (lean_is_exclusive(x_8)) {
 lean_ctor_release(x_8, 0);
 lean_ctor_release(x_8, 1);
 x_18 = x_8;
} else {
 lean_dec_ref(x_8);
 x_18 = lean_box(0);
}
x_19 = lean_ctor_get(x_17, 1);
lean_inc(x_19);
x_20 = lean_ctor_get(x_19, 1);
lean_inc(x_20);
x_21 = lean_ctor_get(x_20, 1);
lean_inc(x_21);
x_22 = lean_ctor_get(x_21, 1);
lean_inc(x_22);
x_23 = lean_ctor_get(x_22, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_17, 0);
lean_inc(x_24);
if (lean_is_exclusive(x_17)) {
 lean_ctor_release(x_17, 0);
 lean_ctor_release(x_17, 1);
 x_25 = x_17;
} else {
 lean_dec_ref(x_17);
 x_25 = lean_box(0);
}
x_26 = lean_ctor_get(x_19, 0);
lean_inc(x_26);
if (lean_is_exclusive(x_19)) {
 lean_ctor_release(x_19, 0);
 lean_ctor_release(x_19, 1);
 x_27 = x_19;
} else {
 lean_dec_ref(x_19);
 x_27 = lean_box(0);
}
x_28 = lean_ctor_get(x_20, 0);
lean_inc(x_28);
if (lean_is_exclusive(x_20)) {
 lean_ctor_release(x_20, 0);
 lean_ctor_release(x_20, 1);
 x_29 = x_20;
} else {
 lean_dec_ref(x_20);
 x_29 = lean_box(0);
}
x_30 = lean_ctor_get(x_21, 0);
lean_inc(x_30);
if (lean_is_exclusive(x_21)) {
 lean_ctor_release(x_21, 0);
 lean_ctor_release(x_21, 1);
 x_31 = x_21;
} else {
 lean_dec_ref(x_21);
 x_31 = lean_box(0);
}
x_32 = lean_ctor_get(x_22, 0);
lean_inc(x_32);
if (lean_is_exclusive(x_22)) {
 lean_ctor_release(x_22, 0);
 lean_ctor_release(x_22, 1);
 x_33 = x_22;
} else {
 lean_dec_ref(x_22);
 x_33 = lean_box(0);
}
x_34 = lean_ctor_get(x_23, 0);
lean_inc(x_34);
x_35 = lean_ctor_get(x_23, 1);
lean_inc(x_35);
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 lean_ctor_release(x_23, 1);
 x_36 = x_23;
} else {
 lean_dec_ref(x_23);
 x_36 = lean_box(0);
}
x_37 = lean_nat_dec_lt(x_28, x_5);
if (x_37 == 0)
{
lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; 
lean_dec(x_36);
lean_dec(x_33);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_27);
lean_dec(x_25);
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_2);
x_192 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_192, 0, x_34);
lean_ctor_set(x_192, 1, x_35);
x_193 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_193, 0, x_32);
lean_ctor_set(x_193, 1, x_192);
x_194 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_194, 0, x_30);
lean_ctor_set(x_194, 1, x_193);
x_195 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_195, 0, x_28);
lean_ctor_set(x_195, 1, x_194);
x_196 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_196, 0, x_26);
lean_ctor_set(x_196, 1, x_195);
x_197 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_197, 0, x_24);
lean_ctor_set(x_197, 1, x_196);
x_198 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_198, 0, x_3);
lean_ctor_set(x_198, 1, x_197);
x_199 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_199, 0, x_198);
return x_199;
}
else
{
uint8_t x_200; 
x_200 = lean_nat_dec_eq(x_35, x_6);
if (x_200 == 0)
{
uint8_t x_201; 
x_201 = lean_unbox(x_24);
lean_dec(x_24);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
x_38 = x_201;
x_39 = x_26;
x_40 = x_28;
x_41 = x_30;
x_42 = x_32;
x_43 = x_34;
x_44 = x_35;
x_45 = x_9;
x_46 = x_10;
x_47 = x_11;
x_48 = x_12;
x_49 = x_13;
x_50 = x_14;
x_51 = x_15;
x_52 = lean_box(0);
goto block_191;
}
else
{
lean_object* x_202; 
lean_dec(x_32);
lean_dec(x_30);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
lean_inc(x_11);
lean_inc(x_26);
x_202 = lp_aesop_Aesop_updateForwardState___redArg(x_26, x_10, x_11, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_202) == 0)
{
lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; 
lean_dec_ref(x_202);
x_203 = lean_st_ref_get(x_10);
x_204 = lean_ctor_get(x_7, 1);
x_205 = lean_ctor_get(x_203, 1);
lean_inc_ref(x_205);
lean_dec(x_203);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
lean_inc(x_11);
lean_inc(x_26);
lean_inc_ref(x_204);
x_206 = lp_aesop_Aesop_selectNormRules(x_204, x_205, x_26, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_205);
if (lean_obj_tag(x_206) == 0)
{
lean_object* x_207; lean_object* x_208; size_t x_209; size_t x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; uint8_t x_214; 
x_207 = lean_ctor_get(x_206, 0);
lean_inc(x_207);
lean_dec_ref(x_206);
x_208 = lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__1;
x_209 = lean_array_size(x_207);
x_210 = 0;
lean_inc(x_6);
x_211 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormSteps_spec__0(x_6, x_207, x_209, x_210, x_208);
lean_dec(x_207);
x_212 = lean_ctor_get(x_211, 0);
lean_inc(x_212);
x_213 = lean_ctor_get(x_211, 1);
lean_inc(x_213);
lean_dec_ref(x_211);
x_214 = lean_unbox(x_24);
lean_dec(x_24);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
x_38 = x_214;
x_39 = x_26;
x_40 = x_28;
x_41 = x_213;
x_42 = x_212;
x_43 = x_34;
x_44 = x_35;
x_45 = x_9;
x_46 = x_10;
x_47 = x_11;
x_48 = x_12;
x_49 = x_13;
x_50 = x_14;
x_51 = x_15;
x_52 = lean_box(0);
goto block_191;
}
else
{
uint8_t x_215; 
lean_dec(x_36);
lean_dec(x_35);
lean_dec(x_34);
lean_dec(x_33);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_26);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
x_215 = !lean_is_exclusive(x_206);
if (x_215 == 0)
{
return x_206;
}
else
{
lean_object* x_216; lean_object* x_217; 
x_216 = lean_ctor_get(x_206, 0);
lean_inc(x_216);
lean_dec(x_206);
x_217 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_217, 0, x_216);
return x_217;
}
}
}
else
{
uint8_t x_218; 
lean_dec(x_36);
lean_dec(x_35);
lean_dec(x_34);
lean_dec(x_33);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_26);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
x_218 = !lean_is_exclusive(x_202);
if (x_218 == 0)
{
return x_202;
}
else
{
lean_object* x_219; lean_object* x_220; 
x_219 = lean_ctor_get(x_202, 0);
lean_inc(x_219);
lean_dec(x_202);
x_220 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_220, 0, x_219);
return x_220;
}
}
}
}
block_191:
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_array_fget_borrowed(x_1, x_44);
lean_inc(x_53);
lean_inc_ref(x_41);
lean_inc_ref(x_42);
lean_inc(x_39);
x_54 = lean_apply_11(x_53, x_39, x_42, x_41, x_45, x_46, x_47, x_48, x_49, x_50, x_51, lean_box(0));
if (lean_obj_tag(x_54) == 0)
{
uint8_t x_55; 
x_55 = !lean_is_exclusive(x_54);
if (x_55 == 0)
{
lean_object* x_56; 
x_56 = lean_ctor_get(x_54, 0);
switch (lean_obj_tag(x_56)) {
case 0:
{
uint8_t x_57; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
x_57 = !lean_is_exclusive(x_56);
if (x_57 == 0)
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_58 = lean_ctor_get(x_56, 0);
x_59 = l_Array_append___redArg(x_43, x_58);
lean_dec_ref(x_58);
lean_inc_ref(x_59);
lean_ctor_set(x_56, 0, x_59);
x_60 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_60, 0, x_56);
if (lean_is_scalar(x_36)) {
 x_61 = lean_alloc_ctor(0, 2, 0);
} else {
 x_61 = x_36;
}
lean_ctor_set(x_61, 0, x_59);
lean_ctor_set(x_61, 1, x_44);
if (lean_is_scalar(x_33)) {
 x_62 = lean_alloc_ctor(0, 2, 0);
} else {
 x_62 = x_33;
}
lean_ctor_set(x_62, 0, x_42);
lean_ctor_set(x_62, 1, x_61);
if (lean_is_scalar(x_31)) {
 x_63 = lean_alloc_ctor(0, 2, 0);
} else {
 x_63 = x_31;
}
lean_ctor_set(x_63, 0, x_41);
lean_ctor_set(x_63, 1, x_62);
if (lean_is_scalar(x_29)) {
 x_64 = lean_alloc_ctor(0, 2, 0);
} else {
 x_64 = x_29;
}
lean_ctor_set(x_64, 0, x_40);
lean_ctor_set(x_64, 1, x_63);
if (lean_is_scalar(x_27)) {
 x_65 = lean_alloc_ctor(0, 2, 0);
} else {
 x_65 = x_27;
}
lean_ctor_set(x_65, 0, x_39);
lean_ctor_set(x_65, 1, x_64);
x_66 = lean_box(x_38);
if (lean_is_scalar(x_25)) {
 x_67 = lean_alloc_ctor(0, 2, 0);
} else {
 x_67 = x_25;
}
lean_ctor_set(x_67, 0, x_66);
lean_ctor_set(x_67, 1, x_65);
if (lean_is_scalar(x_18)) {
 x_68 = lean_alloc_ctor(0, 2, 0);
} else {
 x_68 = x_18;
}
lean_ctor_set(x_68, 0, x_60);
lean_ctor_set(x_68, 1, x_67);
lean_ctor_set(x_54, 0, x_68);
return x_54;
}
else
{
lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; 
x_69 = lean_ctor_get(x_56, 0);
lean_inc(x_69);
lean_dec(x_56);
x_70 = l_Array_append___redArg(x_43, x_69);
lean_dec_ref(x_69);
lean_inc_ref(x_70);
x_71 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_71, 0, x_70);
x_72 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_72, 0, x_71);
if (lean_is_scalar(x_36)) {
 x_73 = lean_alloc_ctor(0, 2, 0);
} else {
 x_73 = x_36;
}
lean_ctor_set(x_73, 0, x_70);
lean_ctor_set(x_73, 1, x_44);
if (lean_is_scalar(x_33)) {
 x_74 = lean_alloc_ctor(0, 2, 0);
} else {
 x_74 = x_33;
}
lean_ctor_set(x_74, 0, x_42);
lean_ctor_set(x_74, 1, x_73);
if (lean_is_scalar(x_31)) {
 x_75 = lean_alloc_ctor(0, 2, 0);
} else {
 x_75 = x_31;
}
lean_ctor_set(x_75, 0, x_41);
lean_ctor_set(x_75, 1, x_74);
if (lean_is_scalar(x_29)) {
 x_76 = lean_alloc_ctor(0, 2, 0);
} else {
 x_76 = x_29;
}
lean_ctor_set(x_76, 0, x_40);
lean_ctor_set(x_76, 1, x_75);
if (lean_is_scalar(x_27)) {
 x_77 = lean_alloc_ctor(0, 2, 0);
} else {
 x_77 = x_27;
}
lean_ctor_set(x_77, 0, x_39);
lean_ctor_set(x_77, 1, x_76);
x_78 = lean_box(x_38);
if (lean_is_scalar(x_25)) {
 x_79 = lean_alloc_ctor(0, 2, 0);
} else {
 x_79 = x_25;
}
lean_ctor_set(x_79, 0, x_78);
lean_ctor_set(x_79, 1, x_77);
if (lean_is_scalar(x_18)) {
 x_80 = lean_alloc_ctor(0, 2, 0);
} else {
 x_80 = x_18;
}
lean_ctor_set(x_80, 0, x_72);
lean_ctor_set(x_80, 1, x_79);
lean_ctor_set(x_54, 0, x_80);
return x_54;
}
}
case 1:
{
lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; 
lean_free_object(x_54);
lean_dec(x_44);
lean_dec(x_39);
x_81 = lean_ctor_get(x_56, 0);
lean_inc(x_81);
x_82 = lean_ctor_get(x_56, 1);
lean_inc_ref(x_82);
lean_dec_ref(x_56);
x_83 = l_Array_append___redArg(x_43, x_82);
lean_dec_ref(x_82);
x_84 = lean_unsigned_to_nat(1u);
x_85 = lean_nat_add(x_40, x_84);
lean_dec(x_40);
lean_inc(x_2);
if (lean_is_scalar(x_36)) {
 x_86 = lean_alloc_ctor(0, 2, 0);
} else {
 x_86 = x_36;
}
lean_ctor_set(x_86, 0, x_83);
lean_ctor_set(x_86, 1, x_2);
if (lean_is_scalar(x_33)) {
 x_87 = lean_alloc_ctor(0, 2, 0);
} else {
 x_87 = x_33;
}
lean_ctor_set(x_87, 0, x_42);
lean_ctor_set(x_87, 1, x_86);
if (lean_is_scalar(x_31)) {
 x_88 = lean_alloc_ctor(0, 2, 0);
} else {
 x_88 = x_31;
}
lean_ctor_set(x_88, 0, x_41);
lean_ctor_set(x_88, 1, x_87);
if (lean_is_scalar(x_29)) {
 x_89 = lean_alloc_ctor(0, 2, 0);
} else {
 x_89 = x_29;
}
lean_ctor_set(x_89, 0, x_85);
lean_ctor_set(x_89, 1, x_88);
if (lean_is_scalar(x_27)) {
 x_90 = lean_alloc_ctor(0, 2, 0);
} else {
 x_90 = x_27;
}
lean_ctor_set(x_90, 0, x_81);
lean_ctor_set(x_90, 1, x_89);
x_91 = lean_box(x_37);
if (lean_is_scalar(x_25)) {
 x_92 = lean_alloc_ctor(0, 2, 0);
} else {
 x_92 = x_25;
}
lean_ctor_set(x_92, 0, x_91);
lean_ctor_set(x_92, 1, x_90);
lean_inc(x_3);
if (lean_is_scalar(x_18)) {
 x_93 = lean_alloc_ctor(0, 2, 0);
} else {
 x_93 = x_18;
}
lean_ctor_set(x_93, 0, x_3);
lean_ctor_set(x_93, 1, x_92);
x_8 = x_93;
goto _start;
}
default: 
{
lean_object* x_95; lean_object* x_96; uint8_t x_97; 
x_95 = lean_unsigned_to_nat(1u);
x_96 = lean_nat_add(x_44, x_95);
x_97 = lean_nat_dec_lt(x_96, x_4);
if (x_97 == 0)
{
lean_dec(x_96);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
if (x_38 == 0)
{
lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; 
x_98 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_98, 0, x_56);
if (lean_is_scalar(x_36)) {
 x_99 = lean_alloc_ctor(0, 2, 0);
} else {
 x_99 = x_36;
}
lean_ctor_set(x_99, 0, x_43);
lean_ctor_set(x_99, 1, x_44);
if (lean_is_scalar(x_33)) {
 x_100 = lean_alloc_ctor(0, 2, 0);
} else {
 x_100 = x_33;
}
lean_ctor_set(x_100, 0, x_42);
lean_ctor_set(x_100, 1, x_99);
if (lean_is_scalar(x_31)) {
 x_101 = lean_alloc_ctor(0, 2, 0);
} else {
 x_101 = x_31;
}
lean_ctor_set(x_101, 0, x_41);
lean_ctor_set(x_101, 1, x_100);
if (lean_is_scalar(x_29)) {
 x_102 = lean_alloc_ctor(0, 2, 0);
} else {
 x_102 = x_29;
}
lean_ctor_set(x_102, 0, x_40);
lean_ctor_set(x_102, 1, x_101);
if (lean_is_scalar(x_27)) {
 x_103 = lean_alloc_ctor(0, 2, 0);
} else {
 x_103 = x_27;
}
lean_ctor_set(x_103, 0, x_39);
lean_ctor_set(x_103, 1, x_102);
x_104 = lean_box(x_38);
if (lean_is_scalar(x_25)) {
 x_105 = lean_alloc_ctor(0, 2, 0);
} else {
 x_105 = x_25;
}
lean_ctor_set(x_105, 0, x_104);
lean_ctor_set(x_105, 1, x_103);
if (lean_is_scalar(x_18)) {
 x_106 = lean_alloc_ctor(0, 2, 0);
} else {
 x_106 = x_18;
}
lean_ctor_set(x_106, 0, x_98);
lean_ctor_set(x_106, 1, x_105);
lean_ctor_set(x_54, 0, x_106);
return x_54;
}
else
{
lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; 
lean_inc_ref(x_43);
lean_inc(x_39);
x_107 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_107, 0, x_39);
lean_ctor_set(x_107, 1, x_43);
x_108 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_108, 0, x_107);
if (lean_is_scalar(x_36)) {
 x_109 = lean_alloc_ctor(0, 2, 0);
} else {
 x_109 = x_36;
}
lean_ctor_set(x_109, 0, x_43);
lean_ctor_set(x_109, 1, x_44);
if (lean_is_scalar(x_33)) {
 x_110 = lean_alloc_ctor(0, 2, 0);
} else {
 x_110 = x_33;
}
lean_ctor_set(x_110, 0, x_42);
lean_ctor_set(x_110, 1, x_109);
if (lean_is_scalar(x_31)) {
 x_111 = lean_alloc_ctor(0, 2, 0);
} else {
 x_111 = x_31;
}
lean_ctor_set(x_111, 0, x_41);
lean_ctor_set(x_111, 1, x_110);
if (lean_is_scalar(x_29)) {
 x_112 = lean_alloc_ctor(0, 2, 0);
} else {
 x_112 = x_29;
}
lean_ctor_set(x_112, 0, x_40);
lean_ctor_set(x_112, 1, x_111);
if (lean_is_scalar(x_27)) {
 x_113 = lean_alloc_ctor(0, 2, 0);
} else {
 x_113 = x_27;
}
lean_ctor_set(x_113, 0, x_39);
lean_ctor_set(x_113, 1, x_112);
x_114 = lean_box(x_38);
if (lean_is_scalar(x_25)) {
 x_115 = lean_alloc_ctor(0, 2, 0);
} else {
 x_115 = x_25;
}
lean_ctor_set(x_115, 0, x_114);
lean_ctor_set(x_115, 1, x_113);
if (lean_is_scalar(x_18)) {
 x_116 = lean_alloc_ctor(0, 2, 0);
} else {
 x_116 = x_18;
}
lean_ctor_set(x_116, 0, x_108);
lean_ctor_set(x_116, 1, x_115);
lean_ctor_set(x_54, 0, x_116);
return x_54;
}
}
else
{
lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; 
lean_free_object(x_54);
lean_dec(x_44);
if (lean_is_scalar(x_36)) {
 x_117 = lean_alloc_ctor(0, 2, 0);
} else {
 x_117 = x_36;
}
lean_ctor_set(x_117, 0, x_43);
lean_ctor_set(x_117, 1, x_96);
if (lean_is_scalar(x_33)) {
 x_118 = lean_alloc_ctor(0, 2, 0);
} else {
 x_118 = x_33;
}
lean_ctor_set(x_118, 0, x_42);
lean_ctor_set(x_118, 1, x_117);
if (lean_is_scalar(x_31)) {
 x_119 = lean_alloc_ctor(0, 2, 0);
} else {
 x_119 = x_31;
}
lean_ctor_set(x_119, 0, x_41);
lean_ctor_set(x_119, 1, x_118);
if (lean_is_scalar(x_29)) {
 x_120 = lean_alloc_ctor(0, 2, 0);
} else {
 x_120 = x_29;
}
lean_ctor_set(x_120, 0, x_40);
lean_ctor_set(x_120, 1, x_119);
if (lean_is_scalar(x_27)) {
 x_121 = lean_alloc_ctor(0, 2, 0);
} else {
 x_121 = x_27;
}
lean_ctor_set(x_121, 0, x_39);
lean_ctor_set(x_121, 1, x_120);
x_122 = lean_box(x_38);
if (lean_is_scalar(x_25)) {
 x_123 = lean_alloc_ctor(0, 2, 0);
} else {
 x_123 = x_25;
}
lean_ctor_set(x_123, 0, x_122);
lean_ctor_set(x_123, 1, x_121);
lean_inc(x_3);
if (lean_is_scalar(x_18)) {
 x_124 = lean_alloc_ctor(0, 2, 0);
} else {
 x_124 = x_18;
}
lean_ctor_set(x_124, 0, x_3);
lean_ctor_set(x_124, 1, x_123);
x_8 = x_124;
goto _start;
}
}
}
}
else
{
lean_object* x_126; 
x_126 = lean_ctor_get(x_54, 0);
lean_inc(x_126);
lean_dec(x_54);
switch (lean_obj_tag(x_126)) {
case 0:
{
lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
x_127 = lean_ctor_get(x_126, 0);
lean_inc_ref(x_127);
if (lean_is_exclusive(x_126)) {
 lean_ctor_release(x_126, 0);
 x_128 = x_126;
} else {
 lean_dec_ref(x_126);
 x_128 = lean_box(0);
}
x_129 = l_Array_append___redArg(x_43, x_127);
lean_dec_ref(x_127);
lean_inc_ref(x_129);
if (lean_is_scalar(x_128)) {
 x_130 = lean_alloc_ctor(0, 1, 0);
} else {
 x_130 = x_128;
}
lean_ctor_set(x_130, 0, x_129);
x_131 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_131, 0, x_130);
if (lean_is_scalar(x_36)) {
 x_132 = lean_alloc_ctor(0, 2, 0);
} else {
 x_132 = x_36;
}
lean_ctor_set(x_132, 0, x_129);
lean_ctor_set(x_132, 1, x_44);
if (lean_is_scalar(x_33)) {
 x_133 = lean_alloc_ctor(0, 2, 0);
} else {
 x_133 = x_33;
}
lean_ctor_set(x_133, 0, x_42);
lean_ctor_set(x_133, 1, x_132);
if (lean_is_scalar(x_31)) {
 x_134 = lean_alloc_ctor(0, 2, 0);
} else {
 x_134 = x_31;
}
lean_ctor_set(x_134, 0, x_41);
lean_ctor_set(x_134, 1, x_133);
if (lean_is_scalar(x_29)) {
 x_135 = lean_alloc_ctor(0, 2, 0);
} else {
 x_135 = x_29;
}
lean_ctor_set(x_135, 0, x_40);
lean_ctor_set(x_135, 1, x_134);
if (lean_is_scalar(x_27)) {
 x_136 = lean_alloc_ctor(0, 2, 0);
} else {
 x_136 = x_27;
}
lean_ctor_set(x_136, 0, x_39);
lean_ctor_set(x_136, 1, x_135);
x_137 = lean_box(x_38);
if (lean_is_scalar(x_25)) {
 x_138 = lean_alloc_ctor(0, 2, 0);
} else {
 x_138 = x_25;
}
lean_ctor_set(x_138, 0, x_137);
lean_ctor_set(x_138, 1, x_136);
if (lean_is_scalar(x_18)) {
 x_139 = lean_alloc_ctor(0, 2, 0);
} else {
 x_139 = x_18;
}
lean_ctor_set(x_139, 0, x_131);
lean_ctor_set(x_139, 1, x_138);
x_140 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_140, 0, x_139);
return x_140;
}
case 1:
{
lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; 
lean_dec(x_44);
lean_dec(x_39);
x_141 = lean_ctor_get(x_126, 0);
lean_inc(x_141);
x_142 = lean_ctor_get(x_126, 1);
lean_inc_ref(x_142);
lean_dec_ref(x_126);
x_143 = l_Array_append___redArg(x_43, x_142);
lean_dec_ref(x_142);
x_144 = lean_unsigned_to_nat(1u);
x_145 = lean_nat_add(x_40, x_144);
lean_dec(x_40);
lean_inc(x_2);
if (lean_is_scalar(x_36)) {
 x_146 = lean_alloc_ctor(0, 2, 0);
} else {
 x_146 = x_36;
}
lean_ctor_set(x_146, 0, x_143);
lean_ctor_set(x_146, 1, x_2);
if (lean_is_scalar(x_33)) {
 x_147 = lean_alloc_ctor(0, 2, 0);
} else {
 x_147 = x_33;
}
lean_ctor_set(x_147, 0, x_42);
lean_ctor_set(x_147, 1, x_146);
if (lean_is_scalar(x_31)) {
 x_148 = lean_alloc_ctor(0, 2, 0);
} else {
 x_148 = x_31;
}
lean_ctor_set(x_148, 0, x_41);
lean_ctor_set(x_148, 1, x_147);
if (lean_is_scalar(x_29)) {
 x_149 = lean_alloc_ctor(0, 2, 0);
} else {
 x_149 = x_29;
}
lean_ctor_set(x_149, 0, x_145);
lean_ctor_set(x_149, 1, x_148);
if (lean_is_scalar(x_27)) {
 x_150 = lean_alloc_ctor(0, 2, 0);
} else {
 x_150 = x_27;
}
lean_ctor_set(x_150, 0, x_141);
lean_ctor_set(x_150, 1, x_149);
x_151 = lean_box(x_37);
if (lean_is_scalar(x_25)) {
 x_152 = lean_alloc_ctor(0, 2, 0);
} else {
 x_152 = x_25;
}
lean_ctor_set(x_152, 0, x_151);
lean_ctor_set(x_152, 1, x_150);
lean_inc(x_3);
if (lean_is_scalar(x_18)) {
 x_153 = lean_alloc_ctor(0, 2, 0);
} else {
 x_153 = x_18;
}
lean_ctor_set(x_153, 0, x_3);
lean_ctor_set(x_153, 1, x_152);
x_8 = x_153;
goto _start;
}
default: 
{
lean_object* x_155; lean_object* x_156; uint8_t x_157; 
x_155 = lean_unsigned_to_nat(1u);
x_156 = lean_nat_add(x_44, x_155);
x_157 = lean_nat_dec_lt(x_156, x_4);
if (x_157 == 0)
{
lean_dec(x_156);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
if (x_38 == 0)
{
lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; 
x_158 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_158, 0, x_126);
if (lean_is_scalar(x_36)) {
 x_159 = lean_alloc_ctor(0, 2, 0);
} else {
 x_159 = x_36;
}
lean_ctor_set(x_159, 0, x_43);
lean_ctor_set(x_159, 1, x_44);
if (lean_is_scalar(x_33)) {
 x_160 = lean_alloc_ctor(0, 2, 0);
} else {
 x_160 = x_33;
}
lean_ctor_set(x_160, 0, x_42);
lean_ctor_set(x_160, 1, x_159);
if (lean_is_scalar(x_31)) {
 x_161 = lean_alloc_ctor(0, 2, 0);
} else {
 x_161 = x_31;
}
lean_ctor_set(x_161, 0, x_41);
lean_ctor_set(x_161, 1, x_160);
if (lean_is_scalar(x_29)) {
 x_162 = lean_alloc_ctor(0, 2, 0);
} else {
 x_162 = x_29;
}
lean_ctor_set(x_162, 0, x_40);
lean_ctor_set(x_162, 1, x_161);
if (lean_is_scalar(x_27)) {
 x_163 = lean_alloc_ctor(0, 2, 0);
} else {
 x_163 = x_27;
}
lean_ctor_set(x_163, 0, x_39);
lean_ctor_set(x_163, 1, x_162);
x_164 = lean_box(x_38);
if (lean_is_scalar(x_25)) {
 x_165 = lean_alloc_ctor(0, 2, 0);
} else {
 x_165 = x_25;
}
lean_ctor_set(x_165, 0, x_164);
lean_ctor_set(x_165, 1, x_163);
if (lean_is_scalar(x_18)) {
 x_166 = lean_alloc_ctor(0, 2, 0);
} else {
 x_166 = x_18;
}
lean_ctor_set(x_166, 0, x_158);
lean_ctor_set(x_166, 1, x_165);
x_167 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_167, 0, x_166);
return x_167;
}
else
{
lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; 
lean_inc_ref(x_43);
lean_inc(x_39);
x_168 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_168, 0, x_39);
lean_ctor_set(x_168, 1, x_43);
x_169 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_169, 0, x_168);
if (lean_is_scalar(x_36)) {
 x_170 = lean_alloc_ctor(0, 2, 0);
} else {
 x_170 = x_36;
}
lean_ctor_set(x_170, 0, x_43);
lean_ctor_set(x_170, 1, x_44);
if (lean_is_scalar(x_33)) {
 x_171 = lean_alloc_ctor(0, 2, 0);
} else {
 x_171 = x_33;
}
lean_ctor_set(x_171, 0, x_42);
lean_ctor_set(x_171, 1, x_170);
if (lean_is_scalar(x_31)) {
 x_172 = lean_alloc_ctor(0, 2, 0);
} else {
 x_172 = x_31;
}
lean_ctor_set(x_172, 0, x_41);
lean_ctor_set(x_172, 1, x_171);
if (lean_is_scalar(x_29)) {
 x_173 = lean_alloc_ctor(0, 2, 0);
} else {
 x_173 = x_29;
}
lean_ctor_set(x_173, 0, x_40);
lean_ctor_set(x_173, 1, x_172);
if (lean_is_scalar(x_27)) {
 x_174 = lean_alloc_ctor(0, 2, 0);
} else {
 x_174 = x_27;
}
lean_ctor_set(x_174, 0, x_39);
lean_ctor_set(x_174, 1, x_173);
x_175 = lean_box(x_38);
if (lean_is_scalar(x_25)) {
 x_176 = lean_alloc_ctor(0, 2, 0);
} else {
 x_176 = x_25;
}
lean_ctor_set(x_176, 0, x_175);
lean_ctor_set(x_176, 1, x_174);
if (lean_is_scalar(x_18)) {
 x_177 = lean_alloc_ctor(0, 2, 0);
} else {
 x_177 = x_18;
}
lean_ctor_set(x_177, 0, x_169);
lean_ctor_set(x_177, 1, x_176);
x_178 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_178, 0, x_177);
return x_178;
}
}
else
{
lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; 
lean_dec(x_44);
if (lean_is_scalar(x_36)) {
 x_179 = lean_alloc_ctor(0, 2, 0);
} else {
 x_179 = x_36;
}
lean_ctor_set(x_179, 0, x_43);
lean_ctor_set(x_179, 1, x_156);
if (lean_is_scalar(x_33)) {
 x_180 = lean_alloc_ctor(0, 2, 0);
} else {
 x_180 = x_33;
}
lean_ctor_set(x_180, 0, x_42);
lean_ctor_set(x_180, 1, x_179);
if (lean_is_scalar(x_31)) {
 x_181 = lean_alloc_ctor(0, 2, 0);
} else {
 x_181 = x_31;
}
lean_ctor_set(x_181, 0, x_41);
lean_ctor_set(x_181, 1, x_180);
if (lean_is_scalar(x_29)) {
 x_182 = lean_alloc_ctor(0, 2, 0);
} else {
 x_182 = x_29;
}
lean_ctor_set(x_182, 0, x_40);
lean_ctor_set(x_182, 1, x_181);
if (lean_is_scalar(x_27)) {
 x_183 = lean_alloc_ctor(0, 2, 0);
} else {
 x_183 = x_27;
}
lean_ctor_set(x_183, 0, x_39);
lean_ctor_set(x_183, 1, x_182);
x_184 = lean_box(x_38);
if (lean_is_scalar(x_25)) {
 x_185 = lean_alloc_ctor(0, 2, 0);
} else {
 x_185 = x_25;
}
lean_ctor_set(x_185, 0, x_184);
lean_ctor_set(x_185, 1, x_183);
lean_inc(x_3);
if (lean_is_scalar(x_18)) {
 x_186 = lean_alloc_ctor(0, 2, 0);
} else {
 x_186 = x_18;
}
lean_ctor_set(x_186, 0, x_3);
lean_ctor_set(x_186, 1, x_185);
x_8 = x_186;
goto _start;
}
}
}
}
}
else
{
uint8_t x_188; 
lean_dec(x_44);
lean_dec_ref(x_43);
lean_dec_ref(x_42);
lean_dec_ref(x_41);
lean_dec(x_40);
lean_dec(x_39);
lean_dec(x_36);
lean_dec(x_33);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_27);
lean_dec(x_25);
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
x_188 = !lean_is_exclusive(x_54);
if (x_188 == 0)
{
return x_54;
}
else
{
lean_object* x_189; lean_object* x_190; 
x_189 = lean_ctor_get(x_54, 0);
lean_inc(x_189);
lean_dec(x_54);
x_190 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_190, 0, x_189);
return x_190;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: exceeded maximum number of normalisation iterations (", 60, 60);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_runNormSteps___redArg___closed__6;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("). This means normalisation probably got stuck in an infinite loop.", 67, 67);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_runNormSteps___redArg___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_runNormSteps___redArg___closed__8;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormSteps___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_11 = lean_ctor_get(x_3, 0);
x_12 = lean_ctor_get(x_11, 0);
x_13 = lean_ctor_get(x_12, 3);
lean_inc(x_13);
x_14 = lean_unsigned_to_nat(0u);
x_15 = lean_array_get_size(x_2);
x_16 = 0;
x_17 = lean_box(0);
x_18 = lp_aesop_Aesop_runNormSteps___redArg___closed__5;
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_1);
lean_ctor_set(x_19, 1, x_18);
x_20 = lean_box(x_16);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_19);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_17);
lean_ctor_set(x_22, 1, x_21);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_3);
x_23 = lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1(x_2, x_14, x_17, x_15, x_13, x_14, x_3, x_22, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_23) == 0)
{
uint8_t x_24; 
x_24 = !lean_is_exclusive(x_23);
if (x_24 == 0)
{
lean_object* x_25; uint8_t x_26; 
x_25 = lean_ctor_get(x_23, 0);
x_26 = !lean_is_exclusive(x_25);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_25, 0);
x_28 = lean_ctor_get(x_25, 1);
lean_dec(x_28);
if (lean_obj_tag(x_27) == 0)
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
lean_free_object(x_23);
x_29 = lp_aesop_Aesop_runNormSteps___redArg___closed__7;
x_30 = l_Nat_reprFast(x_13);
x_31 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_31, 0, x_30);
x_32 = l_Lean_MessageData_ofFormat(x_31);
lean_ctor_set_tag(x_25, 7);
lean_ctor_set(x_25, 1, x_32);
lean_ctor_set(x_25, 0, x_29);
x_33 = lp_aesop_Aesop_runNormSteps___redArg___closed__9;
x_34 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_34, 0, x_25);
lean_ctor_set(x_34, 1, x_33);
x_35 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_34, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_35;
}
else
{
lean_object* x_36; 
lean_free_object(x_25);
lean_dec(x_13);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_36 = lean_ctor_get(x_27, 0);
lean_inc(x_36);
lean_dec_ref(x_27);
lean_ctor_set(x_23, 0, x_36);
return x_23;
}
}
else
{
lean_object* x_37; 
x_37 = lean_ctor_get(x_25, 0);
lean_inc(x_37);
lean_dec(x_25);
if (lean_obj_tag(x_37) == 0)
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; 
lean_free_object(x_23);
x_38 = lp_aesop_Aesop_runNormSteps___redArg___closed__7;
x_39 = l_Nat_reprFast(x_13);
x_40 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_40, 0, x_39);
x_41 = l_Lean_MessageData_ofFormat(x_40);
x_42 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_42, 0, x_38);
lean_ctor_set(x_42, 1, x_41);
x_43 = lp_aesop_Aesop_runNormSteps___redArg___closed__9;
x_44 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_44, 0, x_42);
lean_ctor_set(x_44, 1, x_43);
x_45 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_44, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_45;
}
else
{
lean_object* x_46; 
lean_dec(x_13);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_46 = lean_ctor_get(x_37, 0);
lean_inc(x_46);
lean_dec_ref(x_37);
lean_ctor_set(x_23, 0, x_46);
return x_23;
}
}
}
else
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_47 = lean_ctor_get(x_23, 0);
lean_inc(x_47);
lean_dec(x_23);
x_48 = lean_ctor_get(x_47, 0);
lean_inc(x_48);
if (lean_is_exclusive(x_47)) {
 lean_ctor_release(x_47, 0);
 lean_ctor_release(x_47, 1);
 x_49 = x_47;
} else {
 lean_dec_ref(x_47);
 x_49 = lean_box(0);
}
if (lean_obj_tag(x_48) == 0)
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_50 = lp_aesop_Aesop_runNormSteps___redArg___closed__7;
x_51 = l_Nat_reprFast(x_13);
x_52 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_52, 0, x_51);
x_53 = l_Lean_MessageData_ofFormat(x_52);
if (lean_is_scalar(x_49)) {
 x_54 = lean_alloc_ctor(7, 2, 0);
} else {
 x_54 = x_49;
 lean_ctor_set_tag(x_54, 7);
}
lean_ctor_set(x_54, 0, x_50);
lean_ctor_set(x_54, 1, x_53);
x_55 = lp_aesop_Aesop_runNormSteps___redArg___closed__9;
x_56 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_56, 0, x_54);
lean_ctor_set(x_56, 1, x_55);
x_57 = lp_aesop_Lean_throwError___at___00Aesop_normSimp_spec__0___redArg(x_56, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_57;
}
else
{
lean_object* x_58; lean_object* x_59; 
lean_dec(x_49);
lean_dec(x_13);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_58 = lean_ctor_get(x_48, 0);
lean_inc(x_58);
lean_dec_ref(x_48);
x_59 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_59, 0, x_58);
return x_59;
}
}
}
else
{
uint8_t x_60; 
lean_dec(x_13);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_60 = !lean_is_exclusive(x_23);
if (x_60 == 0)
{
return x_23;
}
else
{
lean_object* x_61; lean_object* x_62; 
x_61 = lean_ctor_get(x_23, 0);
lean_inc(x_61);
lean_dec(x_23);
x_62 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_62, 0, x_61);
return x_62;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormSteps(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_runNormSteps___redArg(x_1, x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormSteps___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_runNormSteps(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormSteps_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_8 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_runNormSteps_spec__0(x_1, x_2, x_6, x_7, x_5);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runNormSteps___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_runNormSteps___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_1);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPreSimpRules___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_runFirstNormRule(x_2, x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_12) == 0)
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_12, 0);
x_15 = lp_aesop_Aesop_optNormRuleResultToNormSeqResult(x_14);
lean_ctor_set(x_12, 0, x_15);
return x_12;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lean_ctor_get(x_12, 0);
lean_inc(x_16);
lean_dec(x_12);
x_17 = lp_aesop_Aesop_optNormRuleResultToNormSeqResult(x_16);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
else
{
uint8_t x_19; 
x_19 = !lean_is_exclusive(x_12);
if (x_19 == 0)
{
return x_12;
}
else
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_12, 0);
lean_inc(x_20);
lean_dec(x_12);
x_21 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPreSimpRules(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_NormStep_runPreSimpRules___redArg(x_1, x_2, x_3, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPreSimpRules___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_NormStep_runPreSimpRules(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPreSimpRules___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_NormStep_runPreSimpRules___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPostSimpRules___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_runFirstNormRule(x_2, x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_12) == 0)
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_12, 0);
x_15 = lp_aesop_Aesop_optNormRuleResultToNormSeqResult(x_14);
lean_ctor_set(x_12, 0, x_15);
return x_12;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lean_ctor_get(x_12, 0);
lean_inc(x_16);
lean_dec(x_12);
x_17 = lp_aesop_Aesop_optNormRuleResultToNormSeqResult(x_16);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
else
{
uint8_t x_19; 
x_19 = !lean_is_exclusive(x_12);
if (x_19 == 0)
{
return x_12;
}
else
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_12, 0);
lean_inc(x_20);
lean_dec(x_12);
x_21 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPostSimpRules(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_NormStep_runPostSimpRules___redArg(x_1, x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPostSimpRules___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_NormStep_runPostSimpRules(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_runPostSimpRules___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_NormStep_runPostSimpRules___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_3);
return x_12;
}
}
static lean_object* _init_lp_aesop_Aesop_NormStep_unfold___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("norm unfold is disabled (options := { ..., enableUnfold := false })", 67, 67);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NormStep_unfold___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_NormStep_unfold___redArg___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_unfold___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_14 = lean_ctor_get(x_2, 0);
x_15 = lean_ctor_get(x_14, 0);
x_16 = lean_ctor_get_uint8(x_15, sizeof(void*)*6 + 10);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; 
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_17 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_18 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_17, x_7);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lean_unbox(x_19);
lean_dec(x_19);
if (x_20 == 0)
{
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
x_10 = lean_box(0);
goto block_13;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_21 = lean_ctor_get(x_17, 0);
lean_inc(x_21);
x_22 = lp_aesop_Aesop_NormStep_unfold___redArg___closed__1;
x_23 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_21, x_22, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
if (lean_obj_tag(x_23) == 0)
{
lean_dec_ref(x_23);
x_10 = lean_box(0);
goto block_13;
}
else
{
uint8_t x_24; 
x_24 = !lean_is_exclusive(x_23);
if (x_24 == 0)
{
return x_23;
}
else
{
lean_object* x_25; lean_object* x_26; 
x_25 = lean_ctor_get(x_23, 0);
lean_inc(x_25);
lean_dec(x_23);
x_26 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_26, 0, x_25);
return x_26;
}
}
}
}
else
{
lean_object* x_27; 
x_27 = lp_aesop_Aesop_normUnfold(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_27) == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
if (lean_is_exclusive(x_27)) {
 lean_ctor_release(x_27, 0);
 x_29 = x_27;
} else {
 lean_dec_ref(x_27);
 x_29 = lean_box(0);
}
if (lean_obj_tag(x_28) == 0)
{
lean_object* x_34; 
x_34 = lean_box(0);
x_30 = x_34;
goto block_33;
}
else
{
uint8_t x_35; 
x_35 = !lean_is_exclusive(x_28);
if (x_35 == 0)
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_36 = lean_ctor_get(x_28, 0);
x_37 = lean_box(2);
x_38 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_36);
lean_ctor_set(x_28, 0, x_38);
x_30 = x_28;
goto block_33;
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_39 = lean_ctor_get(x_28, 0);
lean_inc(x_39);
lean_dec(x_28);
x_40 = lean_box(2);
x_41 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_39);
x_42 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_42, 0, x_41);
x_30 = x_42;
goto block_33;
}
}
block_33:
{
lean_object* x_31; lean_object* x_32; 
x_31 = lp_aesop_Aesop_optNormRuleResultToNormSeqResult(x_30);
if (lean_is_scalar(x_29)) {
 x_32 = lean_alloc_ctor(0, 1, 0);
} else {
 x_32 = x_29;
}
lean_ctor_set(x_32, 0, x_31);
return x_32;
}
}
else
{
uint8_t x_43; 
x_43 = !lean_is_exclusive(x_27);
if (x_43 == 0)
{
return x_27;
}
else
{
lean_object* x_44; lean_object* x_45; 
x_44 = lean_ctor_get(x_27, 0);
lean_inc(x_44);
lean_dec(x_27);
x_45 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_45, 0, x_44);
return x_45;
}
}
}
block_13:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_box(2);
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_unfold(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_NormStep_unfold___redArg(x_1, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_unfold___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_NormStep_unfold(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_unfold___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_NormStep_unfold___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
static lean_object* _init_lp_aesop_Aesop_NormStep_simp___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("norm simp is disabled (simp_options := { ..., enabled := false })", 65, 65);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NormStep_simp___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_NormStep_simp___redArg___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_simp___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_15; uint8_t x_16; 
x_15 = lean_ctor_get(x_3, 2);
x_16 = lean_ctor_get_uint8(x_15, sizeof(void*)*3);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; 
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_17 = lp_aesop_Aesop_withNormTraceNode___closed__29;
x_18 = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_runNormRuleTac_spec__0___redArg(x_17, x_8);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lean_unbox(x_19);
lean_dec(x_19);
if (x_20 == 0)
{
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_21 = lean_ctor_get(x_17, 0);
lean_inc(x_21);
x_22 = lp_aesop_Aesop_NormStep_simp___redArg___closed__1;
x_23 = lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg(x_21, x_22, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
if (lean_obj_tag(x_23) == 0)
{
lean_dec_ref(x_23);
x_11 = lean_box(0);
goto block_14;
}
else
{
uint8_t x_24; 
x_24 = !lean_is_exclusive(x_23);
if (x_24 == 0)
{
return x_23;
}
else
{
lean_object* x_25; lean_object* x_26; 
x_25 = lean_ctor_get(x_23, 0);
lean_inc(x_25);
lean_dec(x_23);
x_26 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_26, 0, x_25);
return x_26;
}
}
}
}
else
{
lean_object* x_27; 
x_27 = lp_aesop_Aesop_normSimp(x_2, x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_27) == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
if (lean_is_exclusive(x_27)) {
 lean_ctor_release(x_27, 0);
 x_29 = x_27;
} else {
 lean_dec_ref(x_27);
 x_29 = lean_box(0);
}
if (lean_obj_tag(x_28) == 0)
{
lean_object* x_34; 
x_34 = lean_box(0);
x_30 = x_34;
goto block_33;
}
else
{
uint8_t x_35; 
x_35 = !lean_is_exclusive(x_28);
if (x_35 == 0)
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_36 = lean_ctor_get(x_28, 0);
x_37 = lean_box(1);
x_38 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_36);
lean_ctor_set(x_28, 0, x_38);
x_30 = x_28;
goto block_33;
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_39 = lean_ctor_get(x_28, 0);
lean_inc(x_39);
lean_dec(x_28);
x_40 = lean_box(1);
x_41 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_39);
x_42 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_42, 0, x_41);
x_30 = x_42;
goto block_33;
}
}
block_33:
{
lean_object* x_31; lean_object* x_32; 
x_31 = lp_aesop_Aesop_optNormRuleResultToNormSeqResult(x_30);
if (lean_is_scalar(x_29)) {
 x_32 = lean_alloc_ctor(0, 1, 0);
} else {
 x_32 = x_29;
}
lean_ctor_set(x_32, 0, x_31);
return x_32;
}
}
else
{
uint8_t x_43; 
x_43 = !lean_is_exclusive(x_27);
if (x_43 == 0)
{
return x_27;
}
else
{
lean_object* x_44; lean_object* x_45; 
x_44 = lean_ctor_get(x_27, 0);
lean_inc(x_44);
lean_dec(x_27);
x_45 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_45, 0, x_44);
return x_45;
}
}
}
block_14:
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_box(2);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_simp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_NormStep_simp___redArg(x_1, x_2, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_simp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_NormStep_simp(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormStep_simp___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_NormStep_simp___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalMVar___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(4u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalMVar(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_11 = lp_aesop_Aesop_updateForwardState___redArg___closed__2;
x_12 = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_runNormRuleTac_spec__4(x_11, x_2);
lean_inc_ref(x_2);
x_13 = lean_alloc_closure((void*)(lp_aesop_Aesop_NormStep_runPreSimpRules___boxed), 12, 1);
lean_closure_set(x_13, 0, x_2);
x_14 = lean_alloc_closure((void*)(lp_aesop_Aesop_NormStep_unfold___boxed), 11, 0);
x_15 = lean_alloc_closure((void*)(lp_aesop_Aesop_NormStep_simp___boxed), 12, 1);
lean_closure_set(x_15, 0, x_12);
x_16 = lean_alloc_closure((void*)(lp_aesop_Aesop_NormStep_runPostSimpRules___boxed), 12, 1);
lean_closure_set(x_16, 0, x_2);
x_17 = lp_aesop_Aesop_normalizeGoalMVar___closed__0;
x_18 = lean_array_push(x_17, x_13);
x_19 = lean_array_push(x_18, x_14);
x_20 = lean_array_push(x_19, x_15);
x_21 = lean_array_push(x_20, x_16);
x_22 = lp_aesop_Aesop_runNormSteps___redArg(x_1, x_21, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_21);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalMVar___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_normalizeGoalMVar(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_treeImpl;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_2 = lp_aesop_Aesop_withNormTraceNode___closed__32;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__7;
x_2 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__1;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_withNormTraceNode___closed__6;
x_2 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__2;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__11;
x_2 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__3;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadFinallyEST___lam__0___boxed), 5, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__4;
x_2 = lean_alloc_closure((void*)(l_ReaderT_tryFinally___redArg___lam__1), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__5;
x_2 = lean_alloc_closure((void*)(l_ReaderT_tryFinally___redArg___lam__1), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__6;
x_2 = lean_alloc_closure((void*)(l_ReaderT_tryFinally___redArg___lam__1), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__7;
x_2 = lean_alloc_closure((void*)(l_ReaderT_tryFinally___redArg___lam__1), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__8;
x_2 = lean_alloc_closure((void*)(l_ReaderT_tryFinally___redArg___lam__1), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__9;
x_2 = lean_alloc_closure((void*)(l_ReaderT_tryFinally___redArg___lam__1), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__10;
x_2 = lean_alloc_closure((void*)(l_ReaderT_tryFinally___redArg___lam__1), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__13;
x_2 = lean_alloc_closure((void*)(l_ReaderT_tryFinally___redArg___lam__1), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_st_ref_get(x_6);
lean_dec(x_14);
x_15 = lean_st_mk_ref(x_1);
lean_inc(x_15);
x_16 = lp_aesop_Aesop_normalizeGoalMVar(x_2, x_3, x_4, x_15, x_8, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_16) == 0)
{
uint8_t x_17; 
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_18 = lean_ctor_get(x_16, 0);
x_19 = lean_st_ref_get(x_15);
lean_dec(x_15);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_18);
lean_ctor_set(x_20, 1, x_19);
lean_ctor_set(x_16, 0, x_20);
return x_16;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_21 = lean_ctor_get(x_16, 0);
lean_inc(x_21);
lean_dec(x_16);
x_22 = lean_st_ref_get(x_15);
lean_dec(x_15);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_24, 0, x_23);
return x_24;
}
}
else
{
uint8_t x_25; 
lean_dec(x_15);
x_25 = !lean_is_exclusive(x_16);
if (x_25 == 0)
{
return x_16;
}
else
{
lean_object* x_26; lean_object* x_27; 
x_26 = lean_ctor_get(x_16, 0);
lean_inc(x_26);
lean_dec(x_16);
x_27 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_27, 0, x_26);
return x_27;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_12 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_2);
x_13 = lean_st_ref_get(x_4);
lean_dec(x_13);
x_14 = lean_st_ref_get(x_1);
x_15 = lean_st_ref_get(x_4);
lean_dec(x_15);
lean_inc(x_14);
x_16 = lp_aesop_Aesop_Goal_isRoot(x_14);
x_17 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__0;
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
x_19 = lean_ctor_get(x_17, 1);
lean_inc_ref(x_19);
lean_inc_ref(x_19);
lean_inc(x_14);
x_20 = lean_apply_1(x_19, x_14);
if (x_16 == 0)
{
lean_object* x_21; 
x_21 = lean_ctor_get(x_20, 6);
lean_inc(x_21);
switch (lean_obj_tag(x_21)) {
case 0:
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_22 = lean_ctor_get(x_20, 5);
lean_inc(x_22);
x_23 = lean_ctor_get(x_20, 7);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_20, 8);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_20, 9);
lean_inc_ref(x_25);
lean_dec_ref(x_20);
x_26 = lean_ctor_get(x_3, 0);
x_27 = lean_ctor_get(x_3, 1);
x_28 = lean_ctor_get(x_3, 2);
x_29 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__12;
x_30 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__14;
lean_inc_ref(x_27);
lean_inc_ref(x_26);
lean_inc_ref(x_28);
x_31 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_31, 0, x_28);
lean_ctor_set(x_31, 1, x_26);
lean_ctor_set(x_31, 2, x_27);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_24);
lean_ctor_set(x_32, 1, x_25);
lean_inc(x_22);
x_33 = lean_alloc_closure((void*)(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___lam__0___boxed), 13, 4);
lean_closure_set(x_33, 0, x_32);
lean_closure_set(x_33, 1, x_22);
lean_closure_set(x_33, 2, x_23);
lean_closure_set(x_33, 3, x_31);
x_34 = lp_aesop_Aesop_Goal_runMetaMInParentState___redArg(x_12, x_29, x_30, x_33, x_14);
lean_inc(x_4);
x_35 = lean_apply_9(x_34, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_35) == 0)
{
uint8_t x_36; 
x_36 = !lean_is_exclusive(x_35);
if (x_36 == 0)
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_37 = lean_ctor_get(x_35, 0);
x_38 = lean_ctor_get(x_37, 0);
x_39 = lean_ctor_get(x_38, 1);
lean_inc(x_39);
x_40 = lean_ctor_get(x_38, 0);
switch (lean_obj_tag(x_40)) {
case 0:
{
uint8_t x_41; 
lean_inc_ref(x_40);
lean_dec(x_22);
x_41 = !lean_is_exclusive(x_39);
if (x_41 == 0)
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; uint8_t x_49; 
x_42 = lean_ctor_get(x_39, 1);
lean_dec(x_42);
x_43 = lean_ctor_get(x_39, 0);
lean_dec(x_43);
x_44 = lean_ctor_get(x_37, 1);
lean_inc(x_44);
lean_dec(x_37);
x_45 = lean_ctor_get(x_40, 0);
lean_inc_ref(x_45);
lean_dec_ref(x_40);
x_46 = lean_st_ref_get(x_4);
lean_dec(x_46);
x_47 = lean_st_ref_take(x_1);
x_48 = lean_apply_1(x_19, x_47);
x_49 = !lean_is_exclusive(x_48);
if (x_49 == 0)
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; uint8_t x_55; lean_object* x_56; 
x_50 = lean_ctor_get(x_48, 6);
lean_dec(x_50);
lean_ctor_set_tag(x_39, 2);
lean_ctor_set(x_39, 1, x_45);
lean_ctor_set(x_39, 0, x_44);
lean_ctor_set(x_48, 6, x_39);
x_51 = lean_apply_1(x_18, x_48);
x_52 = lean_st_ref_set(x_1, x_51);
x_53 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_53);
x_54 = lp_aesop_Aesop_GoalRef_markProvenByNormalization(x_1);
x_55 = 1;
x_56 = lean_box(x_55);
lean_ctor_set(x_35, 0, x_56);
return x_35;
}
else
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; uint8_t x_62; uint8_t x_63; uint8_t x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; double x_69; lean_object* x_70; lean_object* x_71; uint8_t x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; uint8_t x_80; lean_object* x_81; 
x_57 = lean_ctor_get(x_48, 0);
x_58 = lean_ctor_get(x_48, 1);
x_59 = lean_ctor_get(x_48, 2);
x_60 = lean_ctor_get(x_48, 3);
x_61 = lean_ctor_get(x_48, 4);
x_62 = lean_ctor_get_uint8(x_48, sizeof(void*)*14 + 8);
x_63 = lean_ctor_get_uint8(x_48, sizeof(void*)*14 + 9);
x_64 = lean_ctor_get_uint8(x_48, sizeof(void*)*14 + 10);
x_65 = lean_ctor_get(x_48, 5);
x_66 = lean_ctor_get(x_48, 7);
x_67 = lean_ctor_get(x_48, 8);
x_68 = lean_ctor_get(x_48, 9);
x_69 = lean_ctor_get_float(x_48, sizeof(void*)*14);
x_70 = lean_ctor_get(x_48, 10);
x_71 = lean_ctor_get(x_48, 11);
x_72 = lean_ctor_get_uint8(x_48, sizeof(void*)*14 + 11);
x_73 = lean_ctor_get(x_48, 12);
x_74 = lean_ctor_get(x_48, 13);
lean_inc(x_74);
lean_inc(x_73);
lean_inc(x_71);
lean_inc(x_70);
lean_inc(x_68);
lean_inc(x_67);
lean_inc(x_66);
lean_inc(x_65);
lean_inc(x_61);
lean_inc(x_60);
lean_inc(x_59);
lean_inc(x_58);
lean_inc(x_57);
lean_dec(x_48);
lean_ctor_set_tag(x_39, 2);
lean_ctor_set(x_39, 1, x_45);
lean_ctor_set(x_39, 0, x_44);
x_75 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_75, 0, x_57);
lean_ctor_set(x_75, 1, x_58);
lean_ctor_set(x_75, 2, x_59);
lean_ctor_set(x_75, 3, x_60);
lean_ctor_set(x_75, 4, x_61);
lean_ctor_set(x_75, 5, x_65);
lean_ctor_set(x_75, 6, x_39);
lean_ctor_set(x_75, 7, x_66);
lean_ctor_set(x_75, 8, x_67);
lean_ctor_set(x_75, 9, x_68);
lean_ctor_set(x_75, 10, x_70);
lean_ctor_set(x_75, 11, x_71);
lean_ctor_set(x_75, 12, x_73);
lean_ctor_set(x_75, 13, x_74);
lean_ctor_set_uint8(x_75, sizeof(void*)*14 + 8, x_62);
lean_ctor_set_uint8(x_75, sizeof(void*)*14 + 9, x_63);
lean_ctor_set_uint8(x_75, sizeof(void*)*14 + 10, x_64);
lean_ctor_set_float(x_75, sizeof(void*)*14, x_69);
lean_ctor_set_uint8(x_75, sizeof(void*)*14 + 11, x_72);
x_76 = lean_apply_1(x_18, x_75);
x_77 = lean_st_ref_set(x_1, x_76);
x_78 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_78);
x_79 = lp_aesop_Aesop_GoalRef_markProvenByNormalization(x_1);
x_80 = 1;
x_81 = lean_box(x_80);
lean_ctor_set(x_35, 0, x_81);
return x_35;
}
}
else
{
lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; uint8_t x_92; uint8_t x_93; uint8_t x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; double x_99; lean_object* x_100; lean_object* x_101; uint8_t x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; uint8_t x_112; lean_object* x_113; 
lean_dec(x_39);
x_82 = lean_ctor_get(x_37, 1);
lean_inc(x_82);
lean_dec(x_37);
x_83 = lean_ctor_get(x_40, 0);
lean_inc_ref(x_83);
lean_dec_ref(x_40);
x_84 = lean_st_ref_get(x_4);
lean_dec(x_84);
x_85 = lean_st_ref_take(x_1);
x_86 = lean_apply_1(x_19, x_85);
x_87 = lean_ctor_get(x_86, 0);
lean_inc(x_87);
x_88 = lean_ctor_get(x_86, 1);
lean_inc(x_88);
x_89 = lean_ctor_get(x_86, 2);
lean_inc_ref(x_89);
x_90 = lean_ctor_get(x_86, 3);
lean_inc(x_90);
x_91 = lean_ctor_get(x_86, 4);
lean_inc(x_91);
x_92 = lean_ctor_get_uint8(x_86, sizeof(void*)*14 + 8);
x_93 = lean_ctor_get_uint8(x_86, sizeof(void*)*14 + 9);
x_94 = lean_ctor_get_uint8(x_86, sizeof(void*)*14 + 10);
x_95 = lean_ctor_get(x_86, 5);
lean_inc(x_95);
x_96 = lean_ctor_get(x_86, 7);
lean_inc_ref(x_96);
x_97 = lean_ctor_get(x_86, 8);
lean_inc_ref(x_97);
x_98 = lean_ctor_get(x_86, 9);
lean_inc_ref(x_98);
x_99 = lean_ctor_get_float(x_86, sizeof(void*)*14);
x_100 = lean_ctor_get(x_86, 10);
lean_inc(x_100);
x_101 = lean_ctor_get(x_86, 11);
lean_inc(x_101);
x_102 = lean_ctor_get_uint8(x_86, sizeof(void*)*14 + 11);
x_103 = lean_ctor_get(x_86, 12);
lean_inc_ref(x_103);
x_104 = lean_ctor_get(x_86, 13);
lean_inc_ref(x_104);
if (lean_is_exclusive(x_86)) {
 lean_ctor_release(x_86, 0);
 lean_ctor_release(x_86, 1);
 lean_ctor_release(x_86, 2);
 lean_ctor_release(x_86, 3);
 lean_ctor_release(x_86, 4);
 lean_ctor_release(x_86, 5);
 lean_ctor_release(x_86, 6);
 lean_ctor_release(x_86, 7);
 lean_ctor_release(x_86, 8);
 lean_ctor_release(x_86, 9);
 lean_ctor_release(x_86, 10);
 lean_ctor_release(x_86, 11);
 lean_ctor_release(x_86, 12);
 lean_ctor_release(x_86, 13);
 x_105 = x_86;
} else {
 lean_dec_ref(x_86);
 x_105 = lean_box(0);
}
x_106 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_106, 0, x_82);
lean_ctor_set(x_106, 1, x_83);
if (lean_is_scalar(x_105)) {
 x_107 = lean_alloc_ctor(0, 14, 12);
} else {
 x_107 = x_105;
}
lean_ctor_set(x_107, 0, x_87);
lean_ctor_set(x_107, 1, x_88);
lean_ctor_set(x_107, 2, x_89);
lean_ctor_set(x_107, 3, x_90);
lean_ctor_set(x_107, 4, x_91);
lean_ctor_set(x_107, 5, x_95);
lean_ctor_set(x_107, 6, x_106);
lean_ctor_set(x_107, 7, x_96);
lean_ctor_set(x_107, 8, x_97);
lean_ctor_set(x_107, 9, x_98);
lean_ctor_set(x_107, 10, x_100);
lean_ctor_set(x_107, 11, x_101);
lean_ctor_set(x_107, 12, x_103);
lean_ctor_set(x_107, 13, x_104);
lean_ctor_set_uint8(x_107, sizeof(void*)*14 + 8, x_92);
lean_ctor_set_uint8(x_107, sizeof(void*)*14 + 9, x_93);
lean_ctor_set_uint8(x_107, sizeof(void*)*14 + 10, x_94);
lean_ctor_set_float(x_107, sizeof(void*)*14, x_99);
lean_ctor_set_uint8(x_107, sizeof(void*)*14 + 11, x_102);
x_108 = lean_apply_1(x_18, x_107);
x_109 = lean_st_ref_set(x_1, x_108);
x_110 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_110);
x_111 = lp_aesop_Aesop_GoalRef_markProvenByNormalization(x_1);
x_112 = 1;
x_113 = lean_box(x_112);
lean_ctor_set(x_35, 0, x_113);
return x_35;
}
}
case 1:
{
lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; uint8_t x_122; 
lean_inc_ref(x_40);
lean_dec(x_22);
x_114 = lean_ctor_get(x_37, 1);
lean_inc(x_114);
lean_dec(x_37);
x_115 = lean_ctor_get(x_39, 0);
lean_inc_ref(x_115);
x_116 = lean_ctor_get(x_39, 1);
lean_inc_ref(x_116);
lean_dec(x_39);
x_117 = lean_ctor_get(x_40, 0);
lean_inc(x_117);
x_118 = lean_ctor_get(x_40, 1);
lean_inc_ref(x_118);
lean_dec_ref(x_40);
x_119 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_119);
x_120 = lean_st_ref_take(x_1);
lean_inc_ref(x_19);
x_121 = lean_apply_1(x_19, x_120);
x_122 = !lean_is_exclusive(x_121);
if (x_122 == 0)
{
lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; uint8_t x_127; 
x_123 = lean_ctor_get(x_121, 6);
lean_dec(x_123);
x_124 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_124, 0, x_117);
lean_ctor_set(x_124, 1, x_114);
lean_ctor_set(x_124, 2, x_118);
lean_ctor_set(x_121, 6, x_124);
lean_inc(x_18);
x_125 = lean_apply_1(x_18, x_121);
lean_inc_ref(x_19);
x_126 = lean_apply_1(x_19, x_125);
x_127 = !lean_is_exclusive(x_126);
if (x_127 == 0)
{
lean_object* x_128; lean_object* x_129; lean_object* x_130; uint8_t x_131; 
x_128 = lean_ctor_get(x_126, 8);
lean_dec(x_128);
lean_ctor_set(x_126, 8, x_115);
lean_inc(x_18);
x_129 = lean_apply_1(x_18, x_126);
x_130 = lean_apply_1(x_19, x_129);
x_131 = !lean_is_exclusive(x_130);
if (x_131 == 0)
{
lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; 
x_132 = lean_ctor_get(x_130, 9);
lean_dec(x_132);
lean_ctor_set(x_130, 9, x_116);
x_133 = lean_apply_1(x_18, x_130);
x_134 = lean_st_ref_set(x_1, x_133);
x_135 = lean_box(x_16);
lean_ctor_set(x_35, 0, x_135);
return x_35;
}
else
{
lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; uint8_t x_141; uint8_t x_142; uint8_t x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; double x_148; lean_object* x_149; lean_object* x_150; uint8_t x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; 
x_136 = lean_ctor_get(x_130, 0);
x_137 = lean_ctor_get(x_130, 1);
x_138 = lean_ctor_get(x_130, 2);
x_139 = lean_ctor_get(x_130, 3);
x_140 = lean_ctor_get(x_130, 4);
x_141 = lean_ctor_get_uint8(x_130, sizeof(void*)*14 + 8);
x_142 = lean_ctor_get_uint8(x_130, sizeof(void*)*14 + 9);
x_143 = lean_ctor_get_uint8(x_130, sizeof(void*)*14 + 10);
x_144 = lean_ctor_get(x_130, 5);
x_145 = lean_ctor_get(x_130, 6);
x_146 = lean_ctor_get(x_130, 7);
x_147 = lean_ctor_get(x_130, 8);
x_148 = lean_ctor_get_float(x_130, sizeof(void*)*14);
x_149 = lean_ctor_get(x_130, 10);
x_150 = lean_ctor_get(x_130, 11);
x_151 = lean_ctor_get_uint8(x_130, sizeof(void*)*14 + 11);
x_152 = lean_ctor_get(x_130, 12);
x_153 = lean_ctor_get(x_130, 13);
lean_inc(x_153);
lean_inc(x_152);
lean_inc(x_150);
lean_inc(x_149);
lean_inc(x_147);
lean_inc(x_146);
lean_inc(x_145);
lean_inc(x_144);
lean_inc(x_140);
lean_inc(x_139);
lean_inc(x_138);
lean_inc(x_137);
lean_inc(x_136);
lean_dec(x_130);
x_154 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_154, 0, x_136);
lean_ctor_set(x_154, 1, x_137);
lean_ctor_set(x_154, 2, x_138);
lean_ctor_set(x_154, 3, x_139);
lean_ctor_set(x_154, 4, x_140);
lean_ctor_set(x_154, 5, x_144);
lean_ctor_set(x_154, 6, x_145);
lean_ctor_set(x_154, 7, x_146);
lean_ctor_set(x_154, 8, x_147);
lean_ctor_set(x_154, 9, x_116);
lean_ctor_set(x_154, 10, x_149);
lean_ctor_set(x_154, 11, x_150);
lean_ctor_set(x_154, 12, x_152);
lean_ctor_set(x_154, 13, x_153);
lean_ctor_set_uint8(x_154, sizeof(void*)*14 + 8, x_141);
lean_ctor_set_uint8(x_154, sizeof(void*)*14 + 9, x_142);
lean_ctor_set_uint8(x_154, sizeof(void*)*14 + 10, x_143);
lean_ctor_set_float(x_154, sizeof(void*)*14, x_148);
lean_ctor_set_uint8(x_154, sizeof(void*)*14 + 11, x_151);
x_155 = lean_apply_1(x_18, x_154);
x_156 = lean_st_ref_set(x_1, x_155);
x_157 = lean_box(x_16);
lean_ctor_set(x_35, 0, x_157);
return x_35;
}
}
else
{
lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; uint8_t x_163; uint8_t x_164; uint8_t x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; double x_170; lean_object* x_171; lean_object* x_172; uint8_t x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; uint8_t x_184; uint8_t x_185; uint8_t x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; double x_191; lean_object* x_192; lean_object* x_193; uint8_t x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; 
x_158 = lean_ctor_get(x_126, 0);
x_159 = lean_ctor_get(x_126, 1);
x_160 = lean_ctor_get(x_126, 2);
x_161 = lean_ctor_get(x_126, 3);
x_162 = lean_ctor_get(x_126, 4);
x_163 = lean_ctor_get_uint8(x_126, sizeof(void*)*14 + 8);
x_164 = lean_ctor_get_uint8(x_126, sizeof(void*)*14 + 9);
x_165 = lean_ctor_get_uint8(x_126, sizeof(void*)*14 + 10);
x_166 = lean_ctor_get(x_126, 5);
x_167 = lean_ctor_get(x_126, 6);
x_168 = lean_ctor_get(x_126, 7);
x_169 = lean_ctor_get(x_126, 9);
x_170 = lean_ctor_get_float(x_126, sizeof(void*)*14);
x_171 = lean_ctor_get(x_126, 10);
x_172 = lean_ctor_get(x_126, 11);
x_173 = lean_ctor_get_uint8(x_126, sizeof(void*)*14 + 11);
x_174 = lean_ctor_get(x_126, 12);
x_175 = lean_ctor_get(x_126, 13);
lean_inc(x_175);
lean_inc(x_174);
lean_inc(x_172);
lean_inc(x_171);
lean_inc(x_169);
lean_inc(x_168);
lean_inc(x_167);
lean_inc(x_166);
lean_inc(x_162);
lean_inc(x_161);
lean_inc(x_160);
lean_inc(x_159);
lean_inc(x_158);
lean_dec(x_126);
x_176 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_176, 0, x_158);
lean_ctor_set(x_176, 1, x_159);
lean_ctor_set(x_176, 2, x_160);
lean_ctor_set(x_176, 3, x_161);
lean_ctor_set(x_176, 4, x_162);
lean_ctor_set(x_176, 5, x_166);
lean_ctor_set(x_176, 6, x_167);
lean_ctor_set(x_176, 7, x_168);
lean_ctor_set(x_176, 8, x_115);
lean_ctor_set(x_176, 9, x_169);
lean_ctor_set(x_176, 10, x_171);
lean_ctor_set(x_176, 11, x_172);
lean_ctor_set(x_176, 12, x_174);
lean_ctor_set(x_176, 13, x_175);
lean_ctor_set_uint8(x_176, sizeof(void*)*14 + 8, x_163);
lean_ctor_set_uint8(x_176, sizeof(void*)*14 + 9, x_164);
lean_ctor_set_uint8(x_176, sizeof(void*)*14 + 10, x_165);
lean_ctor_set_float(x_176, sizeof(void*)*14, x_170);
lean_ctor_set_uint8(x_176, sizeof(void*)*14 + 11, x_173);
lean_inc(x_18);
x_177 = lean_apply_1(x_18, x_176);
x_178 = lean_apply_1(x_19, x_177);
x_179 = lean_ctor_get(x_178, 0);
lean_inc(x_179);
x_180 = lean_ctor_get(x_178, 1);
lean_inc(x_180);
x_181 = lean_ctor_get(x_178, 2);
lean_inc_ref(x_181);
x_182 = lean_ctor_get(x_178, 3);
lean_inc(x_182);
x_183 = lean_ctor_get(x_178, 4);
lean_inc(x_183);
x_184 = lean_ctor_get_uint8(x_178, sizeof(void*)*14 + 8);
x_185 = lean_ctor_get_uint8(x_178, sizeof(void*)*14 + 9);
x_186 = lean_ctor_get_uint8(x_178, sizeof(void*)*14 + 10);
x_187 = lean_ctor_get(x_178, 5);
lean_inc(x_187);
x_188 = lean_ctor_get(x_178, 6);
lean_inc(x_188);
x_189 = lean_ctor_get(x_178, 7);
lean_inc_ref(x_189);
x_190 = lean_ctor_get(x_178, 8);
lean_inc_ref(x_190);
x_191 = lean_ctor_get_float(x_178, sizeof(void*)*14);
x_192 = lean_ctor_get(x_178, 10);
lean_inc(x_192);
x_193 = lean_ctor_get(x_178, 11);
lean_inc(x_193);
x_194 = lean_ctor_get_uint8(x_178, sizeof(void*)*14 + 11);
x_195 = lean_ctor_get(x_178, 12);
lean_inc_ref(x_195);
x_196 = lean_ctor_get(x_178, 13);
lean_inc_ref(x_196);
if (lean_is_exclusive(x_178)) {
 lean_ctor_release(x_178, 0);
 lean_ctor_release(x_178, 1);
 lean_ctor_release(x_178, 2);
 lean_ctor_release(x_178, 3);
 lean_ctor_release(x_178, 4);
 lean_ctor_release(x_178, 5);
 lean_ctor_release(x_178, 6);
 lean_ctor_release(x_178, 7);
 lean_ctor_release(x_178, 8);
 lean_ctor_release(x_178, 9);
 lean_ctor_release(x_178, 10);
 lean_ctor_release(x_178, 11);
 lean_ctor_release(x_178, 12);
 lean_ctor_release(x_178, 13);
 x_197 = x_178;
} else {
 lean_dec_ref(x_178);
 x_197 = lean_box(0);
}
if (lean_is_scalar(x_197)) {
 x_198 = lean_alloc_ctor(0, 14, 12);
} else {
 x_198 = x_197;
}
lean_ctor_set(x_198, 0, x_179);
lean_ctor_set(x_198, 1, x_180);
lean_ctor_set(x_198, 2, x_181);
lean_ctor_set(x_198, 3, x_182);
lean_ctor_set(x_198, 4, x_183);
lean_ctor_set(x_198, 5, x_187);
lean_ctor_set(x_198, 6, x_188);
lean_ctor_set(x_198, 7, x_189);
lean_ctor_set(x_198, 8, x_190);
lean_ctor_set(x_198, 9, x_116);
lean_ctor_set(x_198, 10, x_192);
lean_ctor_set(x_198, 11, x_193);
lean_ctor_set(x_198, 12, x_195);
lean_ctor_set(x_198, 13, x_196);
lean_ctor_set_uint8(x_198, sizeof(void*)*14 + 8, x_184);
lean_ctor_set_uint8(x_198, sizeof(void*)*14 + 9, x_185);
lean_ctor_set_uint8(x_198, sizeof(void*)*14 + 10, x_186);
lean_ctor_set_float(x_198, sizeof(void*)*14, x_191);
lean_ctor_set_uint8(x_198, sizeof(void*)*14 + 11, x_194);
x_199 = lean_apply_1(x_18, x_198);
x_200 = lean_st_ref_set(x_1, x_199);
x_201 = lean_box(x_16);
lean_ctor_set(x_35, 0, x_201);
return x_35;
}
}
else
{
lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; uint8_t x_207; uint8_t x_208; uint8_t x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; double x_214; lean_object* x_215; lean_object* x_216; uint8_t x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; uint8_t x_229; uint8_t x_230; uint8_t x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; double x_236; lean_object* x_237; lean_object* x_238; uint8_t x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; lean_object* x_245; lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; uint8_t x_251; uint8_t x_252; uint8_t x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; double x_258; lean_object* x_259; lean_object* x_260; uint8_t x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; 
x_202 = lean_ctor_get(x_121, 0);
x_203 = lean_ctor_get(x_121, 1);
x_204 = lean_ctor_get(x_121, 2);
x_205 = lean_ctor_get(x_121, 3);
x_206 = lean_ctor_get(x_121, 4);
x_207 = lean_ctor_get_uint8(x_121, sizeof(void*)*14 + 8);
x_208 = lean_ctor_get_uint8(x_121, sizeof(void*)*14 + 9);
x_209 = lean_ctor_get_uint8(x_121, sizeof(void*)*14 + 10);
x_210 = lean_ctor_get(x_121, 5);
x_211 = lean_ctor_get(x_121, 7);
x_212 = lean_ctor_get(x_121, 8);
x_213 = lean_ctor_get(x_121, 9);
x_214 = lean_ctor_get_float(x_121, sizeof(void*)*14);
x_215 = lean_ctor_get(x_121, 10);
x_216 = lean_ctor_get(x_121, 11);
x_217 = lean_ctor_get_uint8(x_121, sizeof(void*)*14 + 11);
x_218 = lean_ctor_get(x_121, 12);
x_219 = lean_ctor_get(x_121, 13);
lean_inc(x_219);
lean_inc(x_218);
lean_inc(x_216);
lean_inc(x_215);
lean_inc(x_213);
lean_inc(x_212);
lean_inc(x_211);
lean_inc(x_210);
lean_inc(x_206);
lean_inc(x_205);
lean_inc(x_204);
lean_inc(x_203);
lean_inc(x_202);
lean_dec(x_121);
x_220 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_220, 0, x_117);
lean_ctor_set(x_220, 1, x_114);
lean_ctor_set(x_220, 2, x_118);
x_221 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_221, 0, x_202);
lean_ctor_set(x_221, 1, x_203);
lean_ctor_set(x_221, 2, x_204);
lean_ctor_set(x_221, 3, x_205);
lean_ctor_set(x_221, 4, x_206);
lean_ctor_set(x_221, 5, x_210);
lean_ctor_set(x_221, 6, x_220);
lean_ctor_set(x_221, 7, x_211);
lean_ctor_set(x_221, 8, x_212);
lean_ctor_set(x_221, 9, x_213);
lean_ctor_set(x_221, 10, x_215);
lean_ctor_set(x_221, 11, x_216);
lean_ctor_set(x_221, 12, x_218);
lean_ctor_set(x_221, 13, x_219);
lean_ctor_set_uint8(x_221, sizeof(void*)*14 + 8, x_207);
lean_ctor_set_uint8(x_221, sizeof(void*)*14 + 9, x_208);
lean_ctor_set_uint8(x_221, sizeof(void*)*14 + 10, x_209);
lean_ctor_set_float(x_221, sizeof(void*)*14, x_214);
lean_ctor_set_uint8(x_221, sizeof(void*)*14 + 11, x_217);
lean_inc(x_18);
x_222 = lean_apply_1(x_18, x_221);
lean_inc_ref(x_19);
x_223 = lean_apply_1(x_19, x_222);
x_224 = lean_ctor_get(x_223, 0);
lean_inc(x_224);
x_225 = lean_ctor_get(x_223, 1);
lean_inc(x_225);
x_226 = lean_ctor_get(x_223, 2);
lean_inc_ref(x_226);
x_227 = lean_ctor_get(x_223, 3);
lean_inc(x_227);
x_228 = lean_ctor_get(x_223, 4);
lean_inc(x_228);
x_229 = lean_ctor_get_uint8(x_223, sizeof(void*)*14 + 8);
x_230 = lean_ctor_get_uint8(x_223, sizeof(void*)*14 + 9);
x_231 = lean_ctor_get_uint8(x_223, sizeof(void*)*14 + 10);
x_232 = lean_ctor_get(x_223, 5);
lean_inc(x_232);
x_233 = lean_ctor_get(x_223, 6);
lean_inc(x_233);
x_234 = lean_ctor_get(x_223, 7);
lean_inc_ref(x_234);
x_235 = lean_ctor_get(x_223, 9);
lean_inc_ref(x_235);
x_236 = lean_ctor_get_float(x_223, sizeof(void*)*14);
x_237 = lean_ctor_get(x_223, 10);
lean_inc(x_237);
x_238 = lean_ctor_get(x_223, 11);
lean_inc(x_238);
x_239 = lean_ctor_get_uint8(x_223, sizeof(void*)*14 + 11);
x_240 = lean_ctor_get(x_223, 12);
lean_inc_ref(x_240);
x_241 = lean_ctor_get(x_223, 13);
lean_inc_ref(x_241);
if (lean_is_exclusive(x_223)) {
 lean_ctor_release(x_223, 0);
 lean_ctor_release(x_223, 1);
 lean_ctor_release(x_223, 2);
 lean_ctor_release(x_223, 3);
 lean_ctor_release(x_223, 4);
 lean_ctor_release(x_223, 5);
 lean_ctor_release(x_223, 6);
 lean_ctor_release(x_223, 7);
 lean_ctor_release(x_223, 8);
 lean_ctor_release(x_223, 9);
 lean_ctor_release(x_223, 10);
 lean_ctor_release(x_223, 11);
 lean_ctor_release(x_223, 12);
 lean_ctor_release(x_223, 13);
 x_242 = x_223;
} else {
 lean_dec_ref(x_223);
 x_242 = lean_box(0);
}
if (lean_is_scalar(x_242)) {
 x_243 = lean_alloc_ctor(0, 14, 12);
} else {
 x_243 = x_242;
}
lean_ctor_set(x_243, 0, x_224);
lean_ctor_set(x_243, 1, x_225);
lean_ctor_set(x_243, 2, x_226);
lean_ctor_set(x_243, 3, x_227);
lean_ctor_set(x_243, 4, x_228);
lean_ctor_set(x_243, 5, x_232);
lean_ctor_set(x_243, 6, x_233);
lean_ctor_set(x_243, 7, x_234);
lean_ctor_set(x_243, 8, x_115);
lean_ctor_set(x_243, 9, x_235);
lean_ctor_set(x_243, 10, x_237);
lean_ctor_set(x_243, 11, x_238);
lean_ctor_set(x_243, 12, x_240);
lean_ctor_set(x_243, 13, x_241);
lean_ctor_set_uint8(x_243, sizeof(void*)*14 + 8, x_229);
lean_ctor_set_uint8(x_243, sizeof(void*)*14 + 9, x_230);
lean_ctor_set_uint8(x_243, sizeof(void*)*14 + 10, x_231);
lean_ctor_set_float(x_243, sizeof(void*)*14, x_236);
lean_ctor_set_uint8(x_243, sizeof(void*)*14 + 11, x_239);
lean_inc(x_18);
x_244 = lean_apply_1(x_18, x_243);
x_245 = lean_apply_1(x_19, x_244);
x_246 = lean_ctor_get(x_245, 0);
lean_inc(x_246);
x_247 = lean_ctor_get(x_245, 1);
lean_inc(x_247);
x_248 = lean_ctor_get(x_245, 2);
lean_inc_ref(x_248);
x_249 = lean_ctor_get(x_245, 3);
lean_inc(x_249);
x_250 = lean_ctor_get(x_245, 4);
lean_inc(x_250);
x_251 = lean_ctor_get_uint8(x_245, sizeof(void*)*14 + 8);
x_252 = lean_ctor_get_uint8(x_245, sizeof(void*)*14 + 9);
x_253 = lean_ctor_get_uint8(x_245, sizeof(void*)*14 + 10);
x_254 = lean_ctor_get(x_245, 5);
lean_inc(x_254);
x_255 = lean_ctor_get(x_245, 6);
lean_inc(x_255);
x_256 = lean_ctor_get(x_245, 7);
lean_inc_ref(x_256);
x_257 = lean_ctor_get(x_245, 8);
lean_inc_ref(x_257);
x_258 = lean_ctor_get_float(x_245, sizeof(void*)*14);
x_259 = lean_ctor_get(x_245, 10);
lean_inc(x_259);
x_260 = lean_ctor_get(x_245, 11);
lean_inc(x_260);
x_261 = lean_ctor_get_uint8(x_245, sizeof(void*)*14 + 11);
x_262 = lean_ctor_get(x_245, 12);
lean_inc_ref(x_262);
x_263 = lean_ctor_get(x_245, 13);
lean_inc_ref(x_263);
if (lean_is_exclusive(x_245)) {
 lean_ctor_release(x_245, 0);
 lean_ctor_release(x_245, 1);
 lean_ctor_release(x_245, 2);
 lean_ctor_release(x_245, 3);
 lean_ctor_release(x_245, 4);
 lean_ctor_release(x_245, 5);
 lean_ctor_release(x_245, 6);
 lean_ctor_release(x_245, 7);
 lean_ctor_release(x_245, 8);
 lean_ctor_release(x_245, 9);
 lean_ctor_release(x_245, 10);
 lean_ctor_release(x_245, 11);
 lean_ctor_release(x_245, 12);
 lean_ctor_release(x_245, 13);
 x_264 = x_245;
} else {
 lean_dec_ref(x_245);
 x_264 = lean_box(0);
}
if (lean_is_scalar(x_264)) {
 x_265 = lean_alloc_ctor(0, 14, 12);
} else {
 x_265 = x_264;
}
lean_ctor_set(x_265, 0, x_246);
lean_ctor_set(x_265, 1, x_247);
lean_ctor_set(x_265, 2, x_248);
lean_ctor_set(x_265, 3, x_249);
lean_ctor_set(x_265, 4, x_250);
lean_ctor_set(x_265, 5, x_254);
lean_ctor_set(x_265, 6, x_255);
lean_ctor_set(x_265, 7, x_256);
lean_ctor_set(x_265, 8, x_257);
lean_ctor_set(x_265, 9, x_116);
lean_ctor_set(x_265, 10, x_259);
lean_ctor_set(x_265, 11, x_260);
lean_ctor_set(x_265, 12, x_262);
lean_ctor_set(x_265, 13, x_263);
lean_ctor_set_uint8(x_265, sizeof(void*)*14 + 8, x_251);
lean_ctor_set_uint8(x_265, sizeof(void*)*14 + 9, x_252);
lean_ctor_set_uint8(x_265, sizeof(void*)*14 + 10, x_253);
lean_ctor_set_float(x_265, sizeof(void*)*14, x_258);
lean_ctor_set_uint8(x_265, sizeof(void*)*14 + 11, x_261);
x_266 = lean_apply_1(x_18, x_265);
x_267 = lean_st_ref_set(x_1, x_266);
x_268 = lean_box(x_16);
lean_ctor_set(x_35, 0, x_268);
return x_35;
}
}
default: 
{
lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; uint8_t x_273; 
lean_dec(x_39);
x_269 = lean_ctor_get(x_37, 1);
lean_inc(x_269);
lean_dec(x_37);
x_270 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_270);
x_271 = lean_st_ref_take(x_1);
x_272 = lean_apply_1(x_19, x_271);
x_273 = !lean_is_exclusive(x_272);
if (x_273 == 0)
{
lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; lean_object* x_279; 
x_274 = lean_ctor_get(x_272, 6);
lean_dec(x_274);
x_275 = lp_aesop_Aesop_runNormSteps___redArg___closed__0;
x_276 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_276, 0, x_22);
lean_ctor_set(x_276, 1, x_269);
lean_ctor_set(x_276, 2, x_275);
lean_ctor_set(x_272, 6, x_276);
x_277 = lean_apply_1(x_18, x_272);
x_278 = lean_st_ref_set(x_1, x_277);
x_279 = lean_box(x_16);
lean_ctor_set(x_35, 0, x_279);
return x_35;
}
else
{
lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; uint8_t x_285; uint8_t x_286; uint8_t x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; double x_292; lean_object* x_293; lean_object* x_294; uint8_t x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; 
x_280 = lean_ctor_get(x_272, 0);
x_281 = lean_ctor_get(x_272, 1);
x_282 = lean_ctor_get(x_272, 2);
x_283 = lean_ctor_get(x_272, 3);
x_284 = lean_ctor_get(x_272, 4);
x_285 = lean_ctor_get_uint8(x_272, sizeof(void*)*14 + 8);
x_286 = lean_ctor_get_uint8(x_272, sizeof(void*)*14 + 9);
x_287 = lean_ctor_get_uint8(x_272, sizeof(void*)*14 + 10);
x_288 = lean_ctor_get(x_272, 5);
x_289 = lean_ctor_get(x_272, 7);
x_290 = lean_ctor_get(x_272, 8);
x_291 = lean_ctor_get(x_272, 9);
x_292 = lean_ctor_get_float(x_272, sizeof(void*)*14);
x_293 = lean_ctor_get(x_272, 10);
x_294 = lean_ctor_get(x_272, 11);
x_295 = lean_ctor_get_uint8(x_272, sizeof(void*)*14 + 11);
x_296 = lean_ctor_get(x_272, 12);
x_297 = lean_ctor_get(x_272, 13);
lean_inc(x_297);
lean_inc(x_296);
lean_inc(x_294);
lean_inc(x_293);
lean_inc(x_291);
lean_inc(x_290);
lean_inc(x_289);
lean_inc(x_288);
lean_inc(x_284);
lean_inc(x_283);
lean_inc(x_282);
lean_inc(x_281);
lean_inc(x_280);
lean_dec(x_272);
x_298 = lp_aesop_Aesop_runNormSteps___redArg___closed__0;
x_299 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_299, 0, x_22);
lean_ctor_set(x_299, 1, x_269);
lean_ctor_set(x_299, 2, x_298);
x_300 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_300, 0, x_280);
lean_ctor_set(x_300, 1, x_281);
lean_ctor_set(x_300, 2, x_282);
lean_ctor_set(x_300, 3, x_283);
lean_ctor_set(x_300, 4, x_284);
lean_ctor_set(x_300, 5, x_288);
lean_ctor_set(x_300, 6, x_299);
lean_ctor_set(x_300, 7, x_289);
lean_ctor_set(x_300, 8, x_290);
lean_ctor_set(x_300, 9, x_291);
lean_ctor_set(x_300, 10, x_293);
lean_ctor_set(x_300, 11, x_294);
lean_ctor_set(x_300, 12, x_296);
lean_ctor_set(x_300, 13, x_297);
lean_ctor_set_uint8(x_300, sizeof(void*)*14 + 8, x_285);
lean_ctor_set_uint8(x_300, sizeof(void*)*14 + 9, x_286);
lean_ctor_set_uint8(x_300, sizeof(void*)*14 + 10, x_287);
lean_ctor_set_float(x_300, sizeof(void*)*14, x_292);
lean_ctor_set_uint8(x_300, sizeof(void*)*14 + 11, x_295);
x_301 = lean_apply_1(x_18, x_300);
x_302 = lean_st_ref_set(x_1, x_301);
x_303 = lean_box(x_16);
lean_ctor_set(x_35, 0, x_303);
return x_35;
}
}
}
}
else
{
lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; 
x_304 = lean_ctor_get(x_35, 0);
lean_inc(x_304);
lean_dec(x_35);
x_305 = lean_ctor_get(x_304, 0);
x_306 = lean_ctor_get(x_305, 1);
lean_inc(x_306);
x_307 = lean_ctor_get(x_305, 0);
switch (lean_obj_tag(x_307)) {
case 0:
{
lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; lean_object* x_318; uint8_t x_319; uint8_t x_320; uint8_t x_321; lean_object* x_322; lean_object* x_323; lean_object* x_324; lean_object* x_325; double x_326; lean_object* x_327; lean_object* x_328; uint8_t x_329; lean_object* x_330; lean_object* x_331; lean_object* x_332; lean_object* x_333; lean_object* x_334; lean_object* x_335; lean_object* x_336; lean_object* x_337; lean_object* x_338; uint8_t x_339; lean_object* x_340; lean_object* x_341; 
lean_inc_ref(x_307);
lean_dec(x_22);
if (lean_is_exclusive(x_306)) {
 lean_ctor_release(x_306, 0);
 lean_ctor_release(x_306, 1);
 x_308 = x_306;
} else {
 lean_dec_ref(x_306);
 x_308 = lean_box(0);
}
x_309 = lean_ctor_get(x_304, 1);
lean_inc(x_309);
lean_dec(x_304);
x_310 = lean_ctor_get(x_307, 0);
lean_inc_ref(x_310);
lean_dec_ref(x_307);
x_311 = lean_st_ref_get(x_4);
lean_dec(x_311);
x_312 = lean_st_ref_take(x_1);
x_313 = lean_apply_1(x_19, x_312);
x_314 = lean_ctor_get(x_313, 0);
lean_inc(x_314);
x_315 = lean_ctor_get(x_313, 1);
lean_inc(x_315);
x_316 = lean_ctor_get(x_313, 2);
lean_inc_ref(x_316);
x_317 = lean_ctor_get(x_313, 3);
lean_inc(x_317);
x_318 = lean_ctor_get(x_313, 4);
lean_inc(x_318);
x_319 = lean_ctor_get_uint8(x_313, sizeof(void*)*14 + 8);
x_320 = lean_ctor_get_uint8(x_313, sizeof(void*)*14 + 9);
x_321 = lean_ctor_get_uint8(x_313, sizeof(void*)*14 + 10);
x_322 = lean_ctor_get(x_313, 5);
lean_inc(x_322);
x_323 = lean_ctor_get(x_313, 7);
lean_inc_ref(x_323);
x_324 = lean_ctor_get(x_313, 8);
lean_inc_ref(x_324);
x_325 = lean_ctor_get(x_313, 9);
lean_inc_ref(x_325);
x_326 = lean_ctor_get_float(x_313, sizeof(void*)*14);
x_327 = lean_ctor_get(x_313, 10);
lean_inc(x_327);
x_328 = lean_ctor_get(x_313, 11);
lean_inc(x_328);
x_329 = lean_ctor_get_uint8(x_313, sizeof(void*)*14 + 11);
x_330 = lean_ctor_get(x_313, 12);
lean_inc_ref(x_330);
x_331 = lean_ctor_get(x_313, 13);
lean_inc_ref(x_331);
if (lean_is_exclusive(x_313)) {
 lean_ctor_release(x_313, 0);
 lean_ctor_release(x_313, 1);
 lean_ctor_release(x_313, 2);
 lean_ctor_release(x_313, 3);
 lean_ctor_release(x_313, 4);
 lean_ctor_release(x_313, 5);
 lean_ctor_release(x_313, 6);
 lean_ctor_release(x_313, 7);
 lean_ctor_release(x_313, 8);
 lean_ctor_release(x_313, 9);
 lean_ctor_release(x_313, 10);
 lean_ctor_release(x_313, 11);
 lean_ctor_release(x_313, 12);
 lean_ctor_release(x_313, 13);
 x_332 = x_313;
} else {
 lean_dec_ref(x_313);
 x_332 = lean_box(0);
}
if (lean_is_scalar(x_308)) {
 x_333 = lean_alloc_ctor(2, 2, 0);
} else {
 x_333 = x_308;
 lean_ctor_set_tag(x_333, 2);
}
lean_ctor_set(x_333, 0, x_309);
lean_ctor_set(x_333, 1, x_310);
if (lean_is_scalar(x_332)) {
 x_334 = lean_alloc_ctor(0, 14, 12);
} else {
 x_334 = x_332;
}
lean_ctor_set(x_334, 0, x_314);
lean_ctor_set(x_334, 1, x_315);
lean_ctor_set(x_334, 2, x_316);
lean_ctor_set(x_334, 3, x_317);
lean_ctor_set(x_334, 4, x_318);
lean_ctor_set(x_334, 5, x_322);
lean_ctor_set(x_334, 6, x_333);
lean_ctor_set(x_334, 7, x_323);
lean_ctor_set(x_334, 8, x_324);
lean_ctor_set(x_334, 9, x_325);
lean_ctor_set(x_334, 10, x_327);
lean_ctor_set(x_334, 11, x_328);
lean_ctor_set(x_334, 12, x_330);
lean_ctor_set(x_334, 13, x_331);
lean_ctor_set_uint8(x_334, sizeof(void*)*14 + 8, x_319);
lean_ctor_set_uint8(x_334, sizeof(void*)*14 + 9, x_320);
lean_ctor_set_uint8(x_334, sizeof(void*)*14 + 10, x_321);
lean_ctor_set_float(x_334, sizeof(void*)*14, x_326);
lean_ctor_set_uint8(x_334, sizeof(void*)*14 + 11, x_329);
x_335 = lean_apply_1(x_18, x_334);
x_336 = lean_st_ref_set(x_1, x_335);
x_337 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_337);
x_338 = lp_aesop_Aesop_GoalRef_markProvenByNormalization(x_1);
x_339 = 1;
x_340 = lean_box(x_339);
x_341 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_341, 0, x_340);
return x_341;
}
case 1:
{
lean_object* x_342; lean_object* x_343; lean_object* x_344; lean_object* x_345; lean_object* x_346; lean_object* x_347; lean_object* x_348; lean_object* x_349; lean_object* x_350; lean_object* x_351; lean_object* x_352; lean_object* x_353; lean_object* x_354; uint8_t x_355; uint8_t x_356; uint8_t x_357; lean_object* x_358; lean_object* x_359; lean_object* x_360; lean_object* x_361; double x_362; lean_object* x_363; lean_object* x_364; uint8_t x_365; lean_object* x_366; lean_object* x_367; lean_object* x_368; lean_object* x_369; lean_object* x_370; lean_object* x_371; lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; lean_object* x_376; lean_object* x_377; uint8_t x_378; uint8_t x_379; uint8_t x_380; lean_object* x_381; lean_object* x_382; lean_object* x_383; lean_object* x_384; double x_385; lean_object* x_386; lean_object* x_387; uint8_t x_388; lean_object* x_389; lean_object* x_390; lean_object* x_391; lean_object* x_392; lean_object* x_393; lean_object* x_394; lean_object* x_395; lean_object* x_396; lean_object* x_397; lean_object* x_398; lean_object* x_399; uint8_t x_400; uint8_t x_401; uint8_t x_402; lean_object* x_403; lean_object* x_404; lean_object* x_405; lean_object* x_406; double x_407; lean_object* x_408; lean_object* x_409; uint8_t x_410; lean_object* x_411; lean_object* x_412; lean_object* x_413; lean_object* x_414; lean_object* x_415; lean_object* x_416; lean_object* x_417; lean_object* x_418; 
lean_inc_ref(x_307);
lean_dec(x_22);
x_342 = lean_ctor_get(x_304, 1);
lean_inc(x_342);
lean_dec(x_304);
x_343 = lean_ctor_get(x_306, 0);
lean_inc_ref(x_343);
x_344 = lean_ctor_get(x_306, 1);
lean_inc_ref(x_344);
lean_dec(x_306);
x_345 = lean_ctor_get(x_307, 0);
lean_inc(x_345);
x_346 = lean_ctor_get(x_307, 1);
lean_inc_ref(x_346);
lean_dec_ref(x_307);
x_347 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_347);
x_348 = lean_st_ref_take(x_1);
lean_inc_ref(x_19);
x_349 = lean_apply_1(x_19, x_348);
x_350 = lean_ctor_get(x_349, 0);
lean_inc(x_350);
x_351 = lean_ctor_get(x_349, 1);
lean_inc(x_351);
x_352 = lean_ctor_get(x_349, 2);
lean_inc_ref(x_352);
x_353 = lean_ctor_get(x_349, 3);
lean_inc(x_353);
x_354 = lean_ctor_get(x_349, 4);
lean_inc(x_354);
x_355 = lean_ctor_get_uint8(x_349, sizeof(void*)*14 + 8);
x_356 = lean_ctor_get_uint8(x_349, sizeof(void*)*14 + 9);
x_357 = lean_ctor_get_uint8(x_349, sizeof(void*)*14 + 10);
x_358 = lean_ctor_get(x_349, 5);
lean_inc(x_358);
x_359 = lean_ctor_get(x_349, 7);
lean_inc_ref(x_359);
x_360 = lean_ctor_get(x_349, 8);
lean_inc_ref(x_360);
x_361 = lean_ctor_get(x_349, 9);
lean_inc_ref(x_361);
x_362 = lean_ctor_get_float(x_349, sizeof(void*)*14);
x_363 = lean_ctor_get(x_349, 10);
lean_inc(x_363);
x_364 = lean_ctor_get(x_349, 11);
lean_inc(x_364);
x_365 = lean_ctor_get_uint8(x_349, sizeof(void*)*14 + 11);
x_366 = lean_ctor_get(x_349, 12);
lean_inc_ref(x_366);
x_367 = lean_ctor_get(x_349, 13);
lean_inc_ref(x_367);
if (lean_is_exclusive(x_349)) {
 lean_ctor_release(x_349, 0);
 lean_ctor_release(x_349, 1);
 lean_ctor_release(x_349, 2);
 lean_ctor_release(x_349, 3);
 lean_ctor_release(x_349, 4);
 lean_ctor_release(x_349, 5);
 lean_ctor_release(x_349, 6);
 lean_ctor_release(x_349, 7);
 lean_ctor_release(x_349, 8);
 lean_ctor_release(x_349, 9);
 lean_ctor_release(x_349, 10);
 lean_ctor_release(x_349, 11);
 lean_ctor_release(x_349, 12);
 lean_ctor_release(x_349, 13);
 x_368 = x_349;
} else {
 lean_dec_ref(x_349);
 x_368 = lean_box(0);
}
x_369 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_369, 0, x_345);
lean_ctor_set(x_369, 1, x_342);
lean_ctor_set(x_369, 2, x_346);
if (lean_is_scalar(x_368)) {
 x_370 = lean_alloc_ctor(0, 14, 12);
} else {
 x_370 = x_368;
}
lean_ctor_set(x_370, 0, x_350);
lean_ctor_set(x_370, 1, x_351);
lean_ctor_set(x_370, 2, x_352);
lean_ctor_set(x_370, 3, x_353);
lean_ctor_set(x_370, 4, x_354);
lean_ctor_set(x_370, 5, x_358);
lean_ctor_set(x_370, 6, x_369);
lean_ctor_set(x_370, 7, x_359);
lean_ctor_set(x_370, 8, x_360);
lean_ctor_set(x_370, 9, x_361);
lean_ctor_set(x_370, 10, x_363);
lean_ctor_set(x_370, 11, x_364);
lean_ctor_set(x_370, 12, x_366);
lean_ctor_set(x_370, 13, x_367);
lean_ctor_set_uint8(x_370, sizeof(void*)*14 + 8, x_355);
lean_ctor_set_uint8(x_370, sizeof(void*)*14 + 9, x_356);
lean_ctor_set_uint8(x_370, sizeof(void*)*14 + 10, x_357);
lean_ctor_set_float(x_370, sizeof(void*)*14, x_362);
lean_ctor_set_uint8(x_370, sizeof(void*)*14 + 11, x_365);
lean_inc(x_18);
x_371 = lean_apply_1(x_18, x_370);
lean_inc_ref(x_19);
x_372 = lean_apply_1(x_19, x_371);
x_373 = lean_ctor_get(x_372, 0);
lean_inc(x_373);
x_374 = lean_ctor_get(x_372, 1);
lean_inc(x_374);
x_375 = lean_ctor_get(x_372, 2);
lean_inc_ref(x_375);
x_376 = lean_ctor_get(x_372, 3);
lean_inc(x_376);
x_377 = lean_ctor_get(x_372, 4);
lean_inc(x_377);
x_378 = lean_ctor_get_uint8(x_372, sizeof(void*)*14 + 8);
x_379 = lean_ctor_get_uint8(x_372, sizeof(void*)*14 + 9);
x_380 = lean_ctor_get_uint8(x_372, sizeof(void*)*14 + 10);
x_381 = lean_ctor_get(x_372, 5);
lean_inc(x_381);
x_382 = lean_ctor_get(x_372, 6);
lean_inc(x_382);
x_383 = lean_ctor_get(x_372, 7);
lean_inc_ref(x_383);
x_384 = lean_ctor_get(x_372, 9);
lean_inc_ref(x_384);
x_385 = lean_ctor_get_float(x_372, sizeof(void*)*14);
x_386 = lean_ctor_get(x_372, 10);
lean_inc(x_386);
x_387 = lean_ctor_get(x_372, 11);
lean_inc(x_387);
x_388 = lean_ctor_get_uint8(x_372, sizeof(void*)*14 + 11);
x_389 = lean_ctor_get(x_372, 12);
lean_inc_ref(x_389);
x_390 = lean_ctor_get(x_372, 13);
lean_inc_ref(x_390);
if (lean_is_exclusive(x_372)) {
 lean_ctor_release(x_372, 0);
 lean_ctor_release(x_372, 1);
 lean_ctor_release(x_372, 2);
 lean_ctor_release(x_372, 3);
 lean_ctor_release(x_372, 4);
 lean_ctor_release(x_372, 5);
 lean_ctor_release(x_372, 6);
 lean_ctor_release(x_372, 7);
 lean_ctor_release(x_372, 8);
 lean_ctor_release(x_372, 9);
 lean_ctor_release(x_372, 10);
 lean_ctor_release(x_372, 11);
 lean_ctor_release(x_372, 12);
 lean_ctor_release(x_372, 13);
 x_391 = x_372;
} else {
 lean_dec_ref(x_372);
 x_391 = lean_box(0);
}
if (lean_is_scalar(x_391)) {
 x_392 = lean_alloc_ctor(0, 14, 12);
} else {
 x_392 = x_391;
}
lean_ctor_set(x_392, 0, x_373);
lean_ctor_set(x_392, 1, x_374);
lean_ctor_set(x_392, 2, x_375);
lean_ctor_set(x_392, 3, x_376);
lean_ctor_set(x_392, 4, x_377);
lean_ctor_set(x_392, 5, x_381);
lean_ctor_set(x_392, 6, x_382);
lean_ctor_set(x_392, 7, x_383);
lean_ctor_set(x_392, 8, x_343);
lean_ctor_set(x_392, 9, x_384);
lean_ctor_set(x_392, 10, x_386);
lean_ctor_set(x_392, 11, x_387);
lean_ctor_set(x_392, 12, x_389);
lean_ctor_set(x_392, 13, x_390);
lean_ctor_set_uint8(x_392, sizeof(void*)*14 + 8, x_378);
lean_ctor_set_uint8(x_392, sizeof(void*)*14 + 9, x_379);
lean_ctor_set_uint8(x_392, sizeof(void*)*14 + 10, x_380);
lean_ctor_set_float(x_392, sizeof(void*)*14, x_385);
lean_ctor_set_uint8(x_392, sizeof(void*)*14 + 11, x_388);
lean_inc(x_18);
x_393 = lean_apply_1(x_18, x_392);
x_394 = lean_apply_1(x_19, x_393);
x_395 = lean_ctor_get(x_394, 0);
lean_inc(x_395);
x_396 = lean_ctor_get(x_394, 1);
lean_inc(x_396);
x_397 = lean_ctor_get(x_394, 2);
lean_inc_ref(x_397);
x_398 = lean_ctor_get(x_394, 3);
lean_inc(x_398);
x_399 = lean_ctor_get(x_394, 4);
lean_inc(x_399);
x_400 = lean_ctor_get_uint8(x_394, sizeof(void*)*14 + 8);
x_401 = lean_ctor_get_uint8(x_394, sizeof(void*)*14 + 9);
x_402 = lean_ctor_get_uint8(x_394, sizeof(void*)*14 + 10);
x_403 = lean_ctor_get(x_394, 5);
lean_inc(x_403);
x_404 = lean_ctor_get(x_394, 6);
lean_inc(x_404);
x_405 = lean_ctor_get(x_394, 7);
lean_inc_ref(x_405);
x_406 = lean_ctor_get(x_394, 8);
lean_inc_ref(x_406);
x_407 = lean_ctor_get_float(x_394, sizeof(void*)*14);
x_408 = lean_ctor_get(x_394, 10);
lean_inc(x_408);
x_409 = lean_ctor_get(x_394, 11);
lean_inc(x_409);
x_410 = lean_ctor_get_uint8(x_394, sizeof(void*)*14 + 11);
x_411 = lean_ctor_get(x_394, 12);
lean_inc_ref(x_411);
x_412 = lean_ctor_get(x_394, 13);
lean_inc_ref(x_412);
if (lean_is_exclusive(x_394)) {
 lean_ctor_release(x_394, 0);
 lean_ctor_release(x_394, 1);
 lean_ctor_release(x_394, 2);
 lean_ctor_release(x_394, 3);
 lean_ctor_release(x_394, 4);
 lean_ctor_release(x_394, 5);
 lean_ctor_release(x_394, 6);
 lean_ctor_release(x_394, 7);
 lean_ctor_release(x_394, 8);
 lean_ctor_release(x_394, 9);
 lean_ctor_release(x_394, 10);
 lean_ctor_release(x_394, 11);
 lean_ctor_release(x_394, 12);
 lean_ctor_release(x_394, 13);
 x_413 = x_394;
} else {
 lean_dec_ref(x_394);
 x_413 = lean_box(0);
}
if (lean_is_scalar(x_413)) {
 x_414 = lean_alloc_ctor(0, 14, 12);
} else {
 x_414 = x_413;
}
lean_ctor_set(x_414, 0, x_395);
lean_ctor_set(x_414, 1, x_396);
lean_ctor_set(x_414, 2, x_397);
lean_ctor_set(x_414, 3, x_398);
lean_ctor_set(x_414, 4, x_399);
lean_ctor_set(x_414, 5, x_403);
lean_ctor_set(x_414, 6, x_404);
lean_ctor_set(x_414, 7, x_405);
lean_ctor_set(x_414, 8, x_406);
lean_ctor_set(x_414, 9, x_344);
lean_ctor_set(x_414, 10, x_408);
lean_ctor_set(x_414, 11, x_409);
lean_ctor_set(x_414, 12, x_411);
lean_ctor_set(x_414, 13, x_412);
lean_ctor_set_uint8(x_414, sizeof(void*)*14 + 8, x_400);
lean_ctor_set_uint8(x_414, sizeof(void*)*14 + 9, x_401);
lean_ctor_set_uint8(x_414, sizeof(void*)*14 + 10, x_402);
lean_ctor_set_float(x_414, sizeof(void*)*14, x_407);
lean_ctor_set_uint8(x_414, sizeof(void*)*14 + 11, x_410);
x_415 = lean_apply_1(x_18, x_414);
x_416 = lean_st_ref_set(x_1, x_415);
x_417 = lean_box(x_16);
x_418 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_418, 0, x_417);
return x_418;
}
default: 
{
lean_object* x_419; lean_object* x_420; lean_object* x_421; lean_object* x_422; lean_object* x_423; lean_object* x_424; lean_object* x_425; lean_object* x_426; lean_object* x_427; uint8_t x_428; uint8_t x_429; uint8_t x_430; lean_object* x_431; lean_object* x_432; lean_object* x_433; lean_object* x_434; double x_435; lean_object* x_436; lean_object* x_437; uint8_t x_438; lean_object* x_439; lean_object* x_440; lean_object* x_441; lean_object* x_442; lean_object* x_443; lean_object* x_444; lean_object* x_445; lean_object* x_446; lean_object* x_447; lean_object* x_448; 
lean_dec(x_306);
x_419 = lean_ctor_get(x_304, 1);
lean_inc(x_419);
lean_dec(x_304);
x_420 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_420);
x_421 = lean_st_ref_take(x_1);
x_422 = lean_apply_1(x_19, x_421);
x_423 = lean_ctor_get(x_422, 0);
lean_inc(x_423);
x_424 = lean_ctor_get(x_422, 1);
lean_inc(x_424);
x_425 = lean_ctor_get(x_422, 2);
lean_inc_ref(x_425);
x_426 = lean_ctor_get(x_422, 3);
lean_inc(x_426);
x_427 = lean_ctor_get(x_422, 4);
lean_inc(x_427);
x_428 = lean_ctor_get_uint8(x_422, sizeof(void*)*14 + 8);
x_429 = lean_ctor_get_uint8(x_422, sizeof(void*)*14 + 9);
x_430 = lean_ctor_get_uint8(x_422, sizeof(void*)*14 + 10);
x_431 = lean_ctor_get(x_422, 5);
lean_inc(x_431);
x_432 = lean_ctor_get(x_422, 7);
lean_inc_ref(x_432);
x_433 = lean_ctor_get(x_422, 8);
lean_inc_ref(x_433);
x_434 = lean_ctor_get(x_422, 9);
lean_inc_ref(x_434);
x_435 = lean_ctor_get_float(x_422, sizeof(void*)*14);
x_436 = lean_ctor_get(x_422, 10);
lean_inc(x_436);
x_437 = lean_ctor_get(x_422, 11);
lean_inc(x_437);
x_438 = lean_ctor_get_uint8(x_422, sizeof(void*)*14 + 11);
x_439 = lean_ctor_get(x_422, 12);
lean_inc_ref(x_439);
x_440 = lean_ctor_get(x_422, 13);
lean_inc_ref(x_440);
if (lean_is_exclusive(x_422)) {
 lean_ctor_release(x_422, 0);
 lean_ctor_release(x_422, 1);
 lean_ctor_release(x_422, 2);
 lean_ctor_release(x_422, 3);
 lean_ctor_release(x_422, 4);
 lean_ctor_release(x_422, 5);
 lean_ctor_release(x_422, 6);
 lean_ctor_release(x_422, 7);
 lean_ctor_release(x_422, 8);
 lean_ctor_release(x_422, 9);
 lean_ctor_release(x_422, 10);
 lean_ctor_release(x_422, 11);
 lean_ctor_release(x_422, 12);
 lean_ctor_release(x_422, 13);
 x_441 = x_422;
} else {
 lean_dec_ref(x_422);
 x_441 = lean_box(0);
}
x_442 = lp_aesop_Aesop_runNormSteps___redArg___closed__0;
x_443 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_443, 0, x_22);
lean_ctor_set(x_443, 1, x_419);
lean_ctor_set(x_443, 2, x_442);
if (lean_is_scalar(x_441)) {
 x_444 = lean_alloc_ctor(0, 14, 12);
} else {
 x_444 = x_441;
}
lean_ctor_set(x_444, 0, x_423);
lean_ctor_set(x_444, 1, x_424);
lean_ctor_set(x_444, 2, x_425);
lean_ctor_set(x_444, 3, x_426);
lean_ctor_set(x_444, 4, x_427);
lean_ctor_set(x_444, 5, x_431);
lean_ctor_set(x_444, 6, x_443);
lean_ctor_set(x_444, 7, x_432);
lean_ctor_set(x_444, 8, x_433);
lean_ctor_set(x_444, 9, x_434);
lean_ctor_set(x_444, 10, x_436);
lean_ctor_set(x_444, 11, x_437);
lean_ctor_set(x_444, 12, x_439);
lean_ctor_set(x_444, 13, x_440);
lean_ctor_set_uint8(x_444, sizeof(void*)*14 + 8, x_428);
lean_ctor_set_uint8(x_444, sizeof(void*)*14 + 9, x_429);
lean_ctor_set_uint8(x_444, sizeof(void*)*14 + 10, x_430);
lean_ctor_set_float(x_444, sizeof(void*)*14, x_435);
lean_ctor_set_uint8(x_444, sizeof(void*)*14 + 11, x_438);
x_445 = lean_apply_1(x_18, x_444);
x_446 = lean_st_ref_set(x_1, x_445);
x_447 = lean_box(x_16);
x_448 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_448, 0, x_447);
return x_448;
}
}
}
}
else
{
uint8_t x_449; 
lean_dec(x_22);
lean_dec_ref(x_19);
lean_dec(x_18);
lean_dec(x_4);
x_449 = !lean_is_exclusive(x_35);
if (x_449 == 0)
{
return x_35;
}
else
{
lean_object* x_450; lean_object* x_451; 
x_450 = lean_ctor_get(x_35, 0);
lean_inc(x_450);
lean_dec(x_35);
x_451 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_451, 0, x_450);
return x_451;
}
}
}
case 1:
{
lean_object* x_452; lean_object* x_453; 
lean_dec_ref(x_21);
lean_dec_ref(x_20);
lean_dec_ref(x_19);
lean_dec(x_18);
lean_dec(x_14);
lean_dec_ref(x_12);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_452 = lean_box(x_16);
x_453 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_453, 0, x_452);
return x_453;
}
default: 
{
uint8_t x_454; lean_object* x_455; lean_object* x_456; 
lean_dec_ref(x_21);
lean_dec_ref(x_20);
lean_dec_ref(x_19);
lean_dec(x_18);
lean_dec(x_14);
lean_dec_ref(x_12);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_454 = 1;
x_455 = lean_box(x_454);
x_456 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_456, 0, x_455);
return x_456;
}
}
}
else
{
lean_object* x_457; lean_object* x_458; lean_object* x_459; 
lean_dec(x_14);
lean_dec_ref(x_12);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_3);
x_457 = lean_ctor_get(x_20, 5);
lean_inc(x_457);
lean_dec_ref(x_20);
x_458 = lean_st_ref_get(x_4);
lean_dec(x_458);
x_459 = lp_aesop_Aesop_getRootMetaState___redArg(x_5);
lean_dec(x_5);
if (lean_obj_tag(x_459) == 0)
{
uint8_t x_460; 
x_460 = !lean_is_exclusive(x_459);
if (x_460 == 0)
{
lean_object* x_461; lean_object* x_462; lean_object* x_463; lean_object* x_464; uint8_t x_465; 
x_461 = lean_ctor_get(x_459, 0);
x_462 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_462);
x_463 = lean_st_ref_take(x_1);
x_464 = lean_apply_1(x_19, x_463);
x_465 = !lean_is_exclusive(x_464);
if (x_465 == 0)
{
lean_object* x_466; lean_object* x_467; lean_object* x_468; lean_object* x_469; lean_object* x_470; uint8_t x_471; lean_object* x_472; 
x_466 = lean_ctor_get(x_464, 6);
lean_dec(x_466);
x_467 = lp_aesop_Aesop_runNormSteps___redArg___closed__0;
x_468 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_468, 0, x_457);
lean_ctor_set(x_468, 1, x_461);
lean_ctor_set(x_468, 2, x_467);
lean_ctor_set(x_464, 6, x_468);
x_469 = lean_apply_1(x_18, x_464);
x_470 = lean_st_ref_set(x_1, x_469);
x_471 = 0;
x_472 = lean_box(x_471);
lean_ctor_set(x_459, 0, x_472);
return x_459;
}
else
{
lean_object* x_473; lean_object* x_474; lean_object* x_475; lean_object* x_476; lean_object* x_477; uint8_t x_478; uint8_t x_479; uint8_t x_480; lean_object* x_481; lean_object* x_482; lean_object* x_483; lean_object* x_484; double x_485; lean_object* x_486; lean_object* x_487; uint8_t x_488; lean_object* x_489; lean_object* x_490; lean_object* x_491; lean_object* x_492; lean_object* x_493; lean_object* x_494; lean_object* x_495; uint8_t x_496; lean_object* x_497; 
x_473 = lean_ctor_get(x_464, 0);
x_474 = lean_ctor_get(x_464, 1);
x_475 = lean_ctor_get(x_464, 2);
x_476 = lean_ctor_get(x_464, 3);
x_477 = lean_ctor_get(x_464, 4);
x_478 = lean_ctor_get_uint8(x_464, sizeof(void*)*14 + 8);
x_479 = lean_ctor_get_uint8(x_464, sizeof(void*)*14 + 9);
x_480 = lean_ctor_get_uint8(x_464, sizeof(void*)*14 + 10);
x_481 = lean_ctor_get(x_464, 5);
x_482 = lean_ctor_get(x_464, 7);
x_483 = lean_ctor_get(x_464, 8);
x_484 = lean_ctor_get(x_464, 9);
x_485 = lean_ctor_get_float(x_464, sizeof(void*)*14);
x_486 = lean_ctor_get(x_464, 10);
x_487 = lean_ctor_get(x_464, 11);
x_488 = lean_ctor_get_uint8(x_464, sizeof(void*)*14 + 11);
x_489 = lean_ctor_get(x_464, 12);
x_490 = lean_ctor_get(x_464, 13);
lean_inc(x_490);
lean_inc(x_489);
lean_inc(x_487);
lean_inc(x_486);
lean_inc(x_484);
lean_inc(x_483);
lean_inc(x_482);
lean_inc(x_481);
lean_inc(x_477);
lean_inc(x_476);
lean_inc(x_475);
lean_inc(x_474);
lean_inc(x_473);
lean_dec(x_464);
x_491 = lp_aesop_Aesop_runNormSteps___redArg___closed__0;
x_492 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_492, 0, x_457);
lean_ctor_set(x_492, 1, x_461);
lean_ctor_set(x_492, 2, x_491);
x_493 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_493, 0, x_473);
lean_ctor_set(x_493, 1, x_474);
lean_ctor_set(x_493, 2, x_475);
lean_ctor_set(x_493, 3, x_476);
lean_ctor_set(x_493, 4, x_477);
lean_ctor_set(x_493, 5, x_481);
lean_ctor_set(x_493, 6, x_492);
lean_ctor_set(x_493, 7, x_482);
lean_ctor_set(x_493, 8, x_483);
lean_ctor_set(x_493, 9, x_484);
lean_ctor_set(x_493, 10, x_486);
lean_ctor_set(x_493, 11, x_487);
lean_ctor_set(x_493, 12, x_489);
lean_ctor_set(x_493, 13, x_490);
lean_ctor_set_uint8(x_493, sizeof(void*)*14 + 8, x_478);
lean_ctor_set_uint8(x_493, sizeof(void*)*14 + 9, x_479);
lean_ctor_set_uint8(x_493, sizeof(void*)*14 + 10, x_480);
lean_ctor_set_float(x_493, sizeof(void*)*14, x_485);
lean_ctor_set_uint8(x_493, sizeof(void*)*14 + 11, x_488);
x_494 = lean_apply_1(x_18, x_493);
x_495 = lean_st_ref_set(x_1, x_494);
x_496 = 0;
x_497 = lean_box(x_496);
lean_ctor_set(x_459, 0, x_497);
return x_459;
}
}
else
{
lean_object* x_498; lean_object* x_499; lean_object* x_500; lean_object* x_501; lean_object* x_502; lean_object* x_503; lean_object* x_504; lean_object* x_505; lean_object* x_506; uint8_t x_507; uint8_t x_508; uint8_t x_509; lean_object* x_510; lean_object* x_511; lean_object* x_512; lean_object* x_513; double x_514; lean_object* x_515; lean_object* x_516; uint8_t x_517; lean_object* x_518; lean_object* x_519; lean_object* x_520; lean_object* x_521; lean_object* x_522; lean_object* x_523; lean_object* x_524; lean_object* x_525; uint8_t x_526; lean_object* x_527; lean_object* x_528; 
x_498 = lean_ctor_get(x_459, 0);
lean_inc(x_498);
lean_dec(x_459);
x_499 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_499);
x_500 = lean_st_ref_take(x_1);
x_501 = lean_apply_1(x_19, x_500);
x_502 = lean_ctor_get(x_501, 0);
lean_inc(x_502);
x_503 = lean_ctor_get(x_501, 1);
lean_inc(x_503);
x_504 = lean_ctor_get(x_501, 2);
lean_inc_ref(x_504);
x_505 = lean_ctor_get(x_501, 3);
lean_inc(x_505);
x_506 = lean_ctor_get(x_501, 4);
lean_inc(x_506);
x_507 = lean_ctor_get_uint8(x_501, sizeof(void*)*14 + 8);
x_508 = lean_ctor_get_uint8(x_501, sizeof(void*)*14 + 9);
x_509 = lean_ctor_get_uint8(x_501, sizeof(void*)*14 + 10);
x_510 = lean_ctor_get(x_501, 5);
lean_inc(x_510);
x_511 = lean_ctor_get(x_501, 7);
lean_inc_ref(x_511);
x_512 = lean_ctor_get(x_501, 8);
lean_inc_ref(x_512);
x_513 = lean_ctor_get(x_501, 9);
lean_inc_ref(x_513);
x_514 = lean_ctor_get_float(x_501, sizeof(void*)*14);
x_515 = lean_ctor_get(x_501, 10);
lean_inc(x_515);
x_516 = lean_ctor_get(x_501, 11);
lean_inc(x_516);
x_517 = lean_ctor_get_uint8(x_501, sizeof(void*)*14 + 11);
x_518 = lean_ctor_get(x_501, 12);
lean_inc_ref(x_518);
x_519 = lean_ctor_get(x_501, 13);
lean_inc_ref(x_519);
if (lean_is_exclusive(x_501)) {
 lean_ctor_release(x_501, 0);
 lean_ctor_release(x_501, 1);
 lean_ctor_release(x_501, 2);
 lean_ctor_release(x_501, 3);
 lean_ctor_release(x_501, 4);
 lean_ctor_release(x_501, 5);
 lean_ctor_release(x_501, 6);
 lean_ctor_release(x_501, 7);
 lean_ctor_release(x_501, 8);
 lean_ctor_release(x_501, 9);
 lean_ctor_release(x_501, 10);
 lean_ctor_release(x_501, 11);
 lean_ctor_release(x_501, 12);
 lean_ctor_release(x_501, 13);
 x_520 = x_501;
} else {
 lean_dec_ref(x_501);
 x_520 = lean_box(0);
}
x_521 = lp_aesop_Aesop_runNormSteps___redArg___closed__0;
x_522 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_522, 0, x_457);
lean_ctor_set(x_522, 1, x_498);
lean_ctor_set(x_522, 2, x_521);
if (lean_is_scalar(x_520)) {
 x_523 = lean_alloc_ctor(0, 14, 12);
} else {
 x_523 = x_520;
}
lean_ctor_set(x_523, 0, x_502);
lean_ctor_set(x_523, 1, x_503);
lean_ctor_set(x_523, 2, x_504);
lean_ctor_set(x_523, 3, x_505);
lean_ctor_set(x_523, 4, x_506);
lean_ctor_set(x_523, 5, x_510);
lean_ctor_set(x_523, 6, x_522);
lean_ctor_set(x_523, 7, x_511);
lean_ctor_set(x_523, 8, x_512);
lean_ctor_set(x_523, 9, x_513);
lean_ctor_set(x_523, 10, x_515);
lean_ctor_set(x_523, 11, x_516);
lean_ctor_set(x_523, 12, x_518);
lean_ctor_set(x_523, 13, x_519);
lean_ctor_set_uint8(x_523, sizeof(void*)*14 + 8, x_507);
lean_ctor_set_uint8(x_523, sizeof(void*)*14 + 9, x_508);
lean_ctor_set_uint8(x_523, sizeof(void*)*14 + 10, x_509);
lean_ctor_set_float(x_523, sizeof(void*)*14, x_514);
lean_ctor_set_uint8(x_523, sizeof(void*)*14 + 11, x_517);
x_524 = lean_apply_1(x_18, x_523);
x_525 = lean_st_ref_set(x_1, x_524);
x_526 = 0;
x_527 = lean_box(x_526);
x_528 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_528, 0, x_527);
return x_528;
}
}
else
{
uint8_t x_529; 
lean_dec(x_457);
lean_dec_ref(x_19);
lean_dec(x_18);
lean_dec(x_4);
x_529 = !lean_is_exclusive(x_459);
if (x_529 == 0)
{
return x_459;
}
else
{
lean_object* x_530; lean_object* x_531; 
x_530 = lean_ctor_get(x_459, 0);
lean_inc(x_530);
lean_dec(x_459);
x_531 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_531, 0, x_530);
return x_531;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_normalizeGoalIfNecessary(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_3);
lean_dec(x_2);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_12;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Expansion_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Expansion_Simp(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_RuleSelection(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_SearchM(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_State(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_HashSet(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_Expansion_Norm(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Expansion_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Expansion_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_RuleSelection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_SearchM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_HashSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_NormM_instInhabitedState_default___closed__0 = _init_lp_aesop_Aesop_NormM_instInhabitedState_default___closed__0();
lean_mark_persistent(lp_aesop_Aesop_NormM_instInhabitedState_default___closed__0);
lp_aesop_Aesop_NormM_instInhabitedState_default___closed__1 = _init_lp_aesop_Aesop_NormM_instInhabitedState_default___closed__1();
lean_mark_persistent(lp_aesop_Aesop_NormM_instInhabitedState_default___closed__1);
lp_aesop_Aesop_NormM_instInhabitedState_default___closed__2 = _init_lp_aesop_Aesop_NormM_instInhabitedState_default___closed__2();
lean_mark_persistent(lp_aesop_Aesop_NormM_instInhabitedState_default___closed__2);
lp_aesop_Aesop_NormM_instInhabitedState_default = _init_lp_aesop_Aesop_NormM_instInhabitedState_default();
lean_mark_persistent(lp_aesop_Aesop_NormM_instInhabitedState_default);
lp_aesop_Aesop_NormM_instInhabitedState = _init_lp_aesop_Aesop_NormM_instInhabitedState();
lean_mark_persistent(lp_aesop_Aesop_NormM_instInhabitedState);
lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__0 = _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__0();
lean_mark_persistent(lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__0);
lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__1 = _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__1();
lean_mark_persistent(lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__0___closed__1);
lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__0 = _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__0();
lean_mark_persistent(lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__0);
lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__1 = _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__1();
lean_mark_persistent(lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__1___closed__1);
lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__0 = _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__0();
lean_mark_persistent(lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__0);
lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__1 = _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__1();
lean_mark_persistent(lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_getResetForwardState_spec__2___closed__1);
lp_aesop_Aesop_modifyForwardState___redArg___closed__0 = _init_lp_aesop_Aesop_modifyForwardState___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_modifyForwardState___redArg___closed__0);
lp_aesop_Aesop_updateForwardState___redArg___closed__0 = _init_lp_aesop_Aesop_updateForwardState___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_updateForwardState___redArg___closed__0);
lp_aesop_Aesop_updateForwardState___redArg___closed__1 = _init_lp_aesop_Aesop_updateForwardState___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_updateForwardState___redArg___closed__1);
lp_aesop_Aesop_updateForwardState___redArg___closed__2 = _init_lp_aesop_Aesop_updateForwardState___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_updateForwardState___redArg___closed__2);
lp_aesop_Aesop_optNormRuleResultEmoji___closed__0 = _init_lp_aesop_Aesop_optNormRuleResultEmoji___closed__0();
lean_mark_persistent(lp_aesop_Aesop_optNormRuleResultEmoji___closed__0);
lp_aesop_Aesop_optNormRuleResultEmoji___closed__1 = _init_lp_aesop_Aesop_optNormRuleResultEmoji___closed__1();
lean_mark_persistent(lp_aesop_Aesop_optNormRuleResultEmoji___closed__1);
lp_aesop_Aesop_optNormRuleResultEmoji___closed__2 = _init_lp_aesop_Aesop_optNormRuleResultEmoji___closed__2();
lean_mark_persistent(lp_aesop_Aesop_optNormRuleResultEmoji___closed__2);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__0 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__0);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__1 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__1);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__2 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__2);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__3 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__3();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__3);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__4 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__4();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__4);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__5 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__5();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__5);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__6 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__6();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__6);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__7 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__7();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__7);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__8 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__8();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__8);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__9 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__9();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__9);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__10 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__10();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__10);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__11 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__11();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__11);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__12 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__12();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__12);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__13 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__13();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__13);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__14 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__14();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__14);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__15 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__15();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__15);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__16 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__16();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__16);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__17 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__17();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__17);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__18 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__18();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_withNormTraceNode_fmt___redArg___closed__18);
lp_aesop_Aesop_withNormTraceNode___closed__0 = _init_lp_aesop_Aesop_withNormTraceNode___closed__0();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__0);
lp_aesop_Aesop_withNormTraceNode___closed__1 = _init_lp_aesop_Aesop_withNormTraceNode___closed__1();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__1);
lp_aesop_Aesop_withNormTraceNode___closed__2 = _init_lp_aesop_Aesop_withNormTraceNode___closed__2();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__2);
lp_aesop_Aesop_withNormTraceNode___closed__3 = _init_lp_aesop_Aesop_withNormTraceNode___closed__3();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__3);
lp_aesop_Aesop_withNormTraceNode___closed__4 = _init_lp_aesop_Aesop_withNormTraceNode___closed__4();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__4);
lp_aesop_Aesop_withNormTraceNode___closed__5 = _init_lp_aesop_Aesop_withNormTraceNode___closed__5();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__5);
lp_aesop_Aesop_withNormTraceNode___closed__6 = _init_lp_aesop_Aesop_withNormTraceNode___closed__6();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__6);
lp_aesop_Aesop_withNormTraceNode___closed__7 = _init_lp_aesop_Aesop_withNormTraceNode___closed__7();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__7);
lp_aesop_Aesop_withNormTraceNode___closed__8 = _init_lp_aesop_Aesop_withNormTraceNode___closed__8();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__8);
lp_aesop_Aesop_withNormTraceNode___closed__9 = _init_lp_aesop_Aesop_withNormTraceNode___closed__9();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__9);
lp_aesop_Aesop_withNormTraceNode___closed__10 = _init_lp_aesop_Aesop_withNormTraceNode___closed__10();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__10);
lp_aesop_Aesop_withNormTraceNode___closed__11 = _init_lp_aesop_Aesop_withNormTraceNode___closed__11();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__11);
lp_aesop_Aesop_withNormTraceNode___closed__12 = _init_lp_aesop_Aesop_withNormTraceNode___closed__12();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__12);
lp_aesop_Aesop_withNormTraceNode___closed__13 = _init_lp_aesop_Aesop_withNormTraceNode___closed__13();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__13);
lp_aesop_Aesop_withNormTraceNode___closed__14 = _init_lp_aesop_Aesop_withNormTraceNode___closed__14();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__14);
lp_aesop_Aesop_withNormTraceNode___closed__15 = _init_lp_aesop_Aesop_withNormTraceNode___closed__15();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__15);
lp_aesop_Aesop_withNormTraceNode___closed__16 = _init_lp_aesop_Aesop_withNormTraceNode___closed__16();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__16);
lp_aesop_Aesop_withNormTraceNode___closed__17 = _init_lp_aesop_Aesop_withNormTraceNode___closed__17();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__17);
lp_aesop_Aesop_withNormTraceNode___closed__18 = _init_lp_aesop_Aesop_withNormTraceNode___closed__18();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__18);
lp_aesop_Aesop_withNormTraceNode___closed__19 = _init_lp_aesop_Aesop_withNormTraceNode___closed__19();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__19);
lp_aesop_Aesop_withNormTraceNode___closed__20 = _init_lp_aesop_Aesop_withNormTraceNode___closed__20();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__20);
lp_aesop_Aesop_withNormTraceNode___closed__21 = _init_lp_aesop_Aesop_withNormTraceNode___closed__21();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__21);
lp_aesop_Aesop_withNormTraceNode___closed__22 = _init_lp_aesop_Aesop_withNormTraceNode___closed__22();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__22);
lp_aesop_Aesop_withNormTraceNode___closed__23 = _init_lp_aesop_Aesop_withNormTraceNode___closed__23();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__23);
lp_aesop_Aesop_withNormTraceNode___closed__24 = _init_lp_aesop_Aesop_withNormTraceNode___closed__24();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__24);
lp_aesop_Aesop_withNormTraceNode___closed__25 = _init_lp_aesop_Aesop_withNormTraceNode___closed__25();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__25);
lp_aesop_Aesop_withNormTraceNode___closed__26 = _init_lp_aesop_Aesop_withNormTraceNode___closed__26();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__26);
lp_aesop_Aesop_withNormTraceNode___closed__27 = _init_lp_aesop_Aesop_withNormTraceNode___closed__27();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__27);
lp_aesop_Aesop_withNormTraceNode___closed__28 = _init_lp_aesop_Aesop_withNormTraceNode___closed__28();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__28);
lp_aesop_Aesop_withNormTraceNode___closed__29 = _init_lp_aesop_Aesop_withNormTraceNode___closed__29();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__29);
lp_aesop_Aesop_withNormTraceNode___closed__30 = _init_lp_aesop_Aesop_withNormTraceNode___closed__30();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__30);
lp_aesop_Aesop_withNormTraceNode___closed__31 = _init_lp_aesop_Aesop_withNormTraceNode___closed__31();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__31);
lp_aesop_Aesop_withNormTraceNode___closed__32 = _init_lp_aesop_Aesop_withNormTraceNode___closed__32();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__32);
lp_aesop_Aesop_withNormTraceNode___closed__33 = _init_lp_aesop_Aesop_withNormTraceNode___closed__33();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__33);
lp_aesop_Aesop_withNormTraceNode___closed__34 = _init_lp_aesop_Aesop_withNormTraceNode___closed__34();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__34);
lp_aesop_Aesop_withNormTraceNode___closed__35 = _init_lp_aesop_Aesop_withNormTraceNode___closed__35();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__35);
lp_aesop_Aesop_withNormTraceNode___closed__36 = _init_lp_aesop_Aesop_withNormTraceNode___closed__36();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__36);
lp_aesop_Aesop_withNormTraceNode___closed__37 = _init_lp_aesop_Aesop_withNormTraceNode___closed__37();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__37);
lp_aesop_Aesop_withNormTraceNode___closed__38 = _init_lp_aesop_Aesop_withNormTraceNode___closed__38();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__38);
lp_aesop_Aesop_withNormTraceNode___closed__39 = _init_lp_aesop_Aesop_withNormTraceNode___closed__39();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__39);
lp_aesop_Aesop_withNormTraceNode___closed__40 = _init_lp_aesop_Aesop_withNormTraceNode___closed__40();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__40);
lp_aesop_Aesop_withNormTraceNode___closed__41 = _init_lp_aesop_Aesop_withNormTraceNode___closed__41();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__41);
lp_aesop_Aesop_withNormTraceNode___closed__42 = _init_lp_aesop_Aesop_withNormTraceNode___closed__42();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__42);
lp_aesop_Aesop_withNormTraceNode___closed__43 = _init_lp_aesop_Aesop_withNormTraceNode___closed__43();
lean_mark_persistent(lp_aesop_Aesop_withNormTraceNode___closed__43);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__0 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__0);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__1 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__1);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__2 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__2);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__3);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__4 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__4();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__4);
lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__5 = _init_lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__5();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Expansion_Norm_0__Aesop_runNormRuleTac_err___redArg___closed__5);
lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0 = _init_lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__0();
lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__1 = _init_lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__1();
lean_mark_persistent(lp_aesop_Lean_addTrace___at___00Aesop_runNormRuleTac_spec__2___redArg___closed__1);
lp_aesop_Aesop_runNormRuleTac___closed__0 = _init_lp_aesop_Aesop_runNormRuleTac___closed__0();
lean_mark_persistent(lp_aesop_Aesop_runNormRuleTac___closed__0);
lp_aesop_Aesop_runNormRuleTac___closed__1 = _init_lp_aesop_Aesop_runNormRuleTac___closed__1();
lean_mark_persistent(lp_aesop_Aesop_runNormRuleTac___closed__1);
lp_aesop_Aesop_runNormRuleTac___closed__2 = _init_lp_aesop_Aesop_runNormRuleTac___closed__2();
lean_mark_persistent(lp_aesop_Aesop_runNormRuleTac___closed__2);
lp_aesop_Aesop_runNormRuleTac___closed__3 = _init_lp_aesop_Aesop_runNormRuleTac___closed__3();
lean_mark_persistent(lp_aesop_Aesop_runNormRuleTac___closed__3);
lp_aesop_Aesop_runNormRuleTac___closed__4 = _init_lp_aesop_Aesop_runNormRuleTac___closed__4();
lean_mark_persistent(lp_aesop_Aesop_runNormRuleTac___closed__4);
lp_aesop_Aesop_runNormRuleTac___closed__5 = _init_lp_aesop_Aesop_runNormRuleTac___closed__5();
lean_mark_persistent(lp_aesop_Aesop_runNormRuleTac___closed__5);
lp_aesop_Aesop_runNormRuleTac___closed__6 = _init_lp_aesop_Aesop_runNormRuleTac___closed__6();
lean_mark_persistent(lp_aesop_Aesop_runNormRuleTac___closed__6);
lp_aesop_Aesop_runNormRuleTac___closed__7 = _init_lp_aesop_Aesop_runNormRuleTac___closed__7();
lean_mark_persistent(lp_aesop_Aesop_runNormRuleTac___closed__7);
lp_aesop_Aesop_runNormRuleTac___closed__8 = _init_lp_aesop_Aesop_runNormRuleTac___closed__8();
lean_mark_persistent(lp_aesop_Aesop_runNormRuleTac___closed__8);
lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__0 = _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__0();
lean_mark_persistent(lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__0);
lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__1 = _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__1();
lean_mark_persistent(lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__1);
lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__2 = _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__2();
lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__3 = _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__3();
lean_mark_persistent(lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__3);
lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__4 = _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__4();
lean_mark_persistent(lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__4);
lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__5 = _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__5();
lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__0 = _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__0();
lean_mark_persistent(lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__0);
lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__1 = _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__1();
lean_mark_persistent(lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__1);
lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__2 = _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__2();
lean_mark_persistent(lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_withTraceNode___at___00Aesop_runNormRule_spec__1_spec__2___redArg___closed__2);
lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__6 = _init_lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__6();
lean_mark_persistent(lp_aesop_Lean_withTraceNode___at___00Aesop_runNormRule_spec__1___redArg___closed__6);
lp_aesop_Aesop_runNormRule___closed__0 = _init_lp_aesop_Aesop_runNormRule___closed__0();
lean_mark_persistent(lp_aesop_Aesop_runNormRule___closed__0);
lp_aesop_Aesop_runNormRule___closed__1 = _init_lp_aesop_Aesop_runNormRule___closed__1();
lean_mark_persistent(lp_aesop_Aesop_runNormRule___closed__1);
lp_aesop_Aesop_runNormRule___closed__2 = _init_lp_aesop_Aesop_runNormRule___closed__2();
lean_mark_persistent(lp_aesop_Aesop_runNormRule___closed__2);
lp_aesop_Aesop_runFirstNormRule___closed__0 = _init_lp_aesop_Aesop_runFirstNormRule___closed__0();
lean_mark_persistent(lp_aesop_Aesop_runFirstNormRule___closed__0);
lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0 = _init_lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__0);
lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__1 = _init_lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__1);
lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__2 = _init_lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__2);
lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__3 = _init_lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_mkNormSimpScriptStep___redArg___closed__3);
lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__0 = _init_lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__0();
lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__1 = _init_lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_normSimpCore_spec__0_spec__0_spec__0___redArg___closed__1();
lp_aesop_Aesop_normSimpCore___lam__0___closed__0 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__0();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__0);
lp_aesop_Aesop_normSimpCore___lam__0___closed__1 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__1();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__1);
lp_aesop_Aesop_normSimpCore___lam__0___closed__2 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__2();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__2);
lp_aesop_Aesop_normSimpCore___lam__0___closed__3 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__3();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__3);
lp_aesop_Aesop_normSimpCore___lam__0___closed__4 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__4();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__4);
lp_aesop_Aesop_normSimpCore___lam__0___closed__8 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__8();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__8);
lp_aesop_Aesop_normSimpCore___lam__0___closed__9 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__9();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__9);
lp_aesop_Aesop_normSimpCore___lam__0___closed__10 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__10();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__10);
lp_aesop_Aesop_normSimpCore___lam__0___closed__5 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__5();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__5);
lp_aesop_Aesop_normSimpCore___lam__0___closed__6 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__6();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__6);
lp_aesop_Aesop_normSimpCore___lam__0___closed__11 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__11();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__11);
lp_aesop_Aesop_normSimpCore___lam__0___closed__7 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__7();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__7);
lp_aesop_Aesop_normSimpCore___lam__0___closed__12 = _init_lp_aesop_Aesop_normSimpCore___lam__0___closed__12();
lean_mark_persistent(lp_aesop_Aesop_normSimpCore___lam__0___closed__12);
lp_aesop_Aesop_checkSimp___lam__1___closed__0 = _init_lp_aesop_Aesop_checkSimp___lam__1___closed__0();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___lam__1___closed__0);
lp_aesop_Aesop_checkSimp___closed__0 = _init_lp_aesop_Aesop_checkSimp___closed__0();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__0);
lp_aesop_Aesop_checkSimp___closed__1 = _init_lp_aesop_Aesop_checkSimp___closed__1();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__1);
lp_aesop_Aesop_checkSimp___closed__2 = _init_lp_aesop_Aesop_checkSimp___closed__2();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__2);
lp_aesop_Aesop_checkSimp___closed__3 = _init_lp_aesop_Aesop_checkSimp___closed__3();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__3);
lp_aesop_Aesop_checkSimp___closed__4 = _init_lp_aesop_Aesop_checkSimp___closed__4();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__4);
lp_aesop_Aesop_checkSimp___closed__5 = _init_lp_aesop_Aesop_checkSimp___closed__5();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__5);
lp_aesop_Aesop_checkSimp___closed__6 = _init_lp_aesop_Aesop_checkSimp___closed__6();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__6);
lp_aesop_Aesop_checkSimp___closed__7 = _init_lp_aesop_Aesop_checkSimp___closed__7();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__7);
lp_aesop_Aesop_checkSimp___closed__8 = _init_lp_aesop_Aesop_checkSimp___closed__8();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__8);
lp_aesop_Aesop_checkSimp___closed__9 = _init_lp_aesop_Aesop_checkSimp___closed__9();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__9);
lp_aesop_Aesop_checkSimp___closed__10 = _init_lp_aesop_Aesop_checkSimp___closed__10();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__10);
lp_aesop_Aesop_checkSimp___closed__11 = _init_lp_aesop_Aesop_checkSimp___closed__11();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__11);
lp_aesop_Aesop_checkSimp___closed__12 = _init_lp_aesop_Aesop_checkSimp___closed__12();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__12);
lp_aesop_Aesop_checkSimp___closed__13 = _init_lp_aesop_Aesop_checkSimp___closed__13();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__13);
lp_aesop_Aesop_checkSimp___closed__14 = _init_lp_aesop_Aesop_checkSimp___closed__14();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__14);
lp_aesop_Aesop_checkSimp___closed__15 = _init_lp_aesop_Aesop_checkSimp___closed__15();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__15);
lp_aesop_Aesop_checkSimp___closed__16 = _init_lp_aesop_Aesop_checkSimp___closed__16();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__16);
lp_aesop_Aesop_checkSimp___closed__17 = _init_lp_aesop_Aesop_checkSimp___closed__17();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__17);
lp_aesop_Aesop_checkSimp___closed__18 = _init_lp_aesop_Aesop_checkSimp___closed__18();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__18);
lp_aesop_Aesop_checkSimp___closed__19 = _init_lp_aesop_Aesop_checkSimp___closed__19();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__19);
lp_aesop_Aesop_checkSimp___closed__20 = _init_lp_aesop_Aesop_checkSimp___closed__20();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__20);
lp_aesop_Aesop_checkSimp___closed__21 = _init_lp_aesop_Aesop_checkSimp___closed__21();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__21);
lp_aesop_Aesop_checkSimp___closed__22 = _init_lp_aesop_Aesop_checkSimp___closed__22();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__22);
lp_aesop_Aesop_checkSimp___closed__23 = _init_lp_aesop_Aesop_checkSimp___closed__23();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__23);
lp_aesop_Aesop_checkSimp___closed__24 = _init_lp_aesop_Aesop_checkSimp___closed__24();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__24);
lp_aesop_Aesop_checkSimp___closed__25 = _init_lp_aesop_Aesop_checkSimp___closed__25();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__25);
lp_aesop_Aesop_checkSimp___closed__26 = _init_lp_aesop_Aesop_checkSimp___closed__26();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__26);
lp_aesop_Aesop_checkSimp___closed__27 = _init_lp_aesop_Aesop_checkSimp___closed__27();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__27);
lp_aesop_Aesop_checkSimp___closed__28 = _init_lp_aesop_Aesop_checkSimp___closed__28();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__28);
lp_aesop_Aesop_checkSimp___closed__29 = _init_lp_aesop_Aesop_checkSimp___closed__29();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__29);
lp_aesop_Aesop_checkSimp___closed__30 = _init_lp_aesop_Aesop_checkSimp___closed__30();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__30);
lp_aesop_Aesop_checkSimp___closed__31 = _init_lp_aesop_Aesop_checkSimp___closed__31();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__31);
lp_aesop_Aesop_checkSimp___closed__32 = _init_lp_aesop_Aesop_checkSimp___closed__32();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__32);
lp_aesop_Aesop_checkSimp___closed__33 = _init_lp_aesop_Aesop_checkSimp___closed__33();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__33);
lp_aesop_Aesop_checkSimp___closed__34 = _init_lp_aesop_Aesop_checkSimp___closed__34();
lean_mark_persistent(lp_aesop_Aesop_checkSimp___closed__34);
lp_aesop_Aesop_normSimp___lam__0___closed__0 = _init_lp_aesop_Aesop_normSimp___lam__0___closed__0();
lean_mark_persistent(lp_aesop_Aesop_normSimp___lam__0___closed__0);
lp_aesop_Aesop_normSimp___lam__0___closed__1 = _init_lp_aesop_Aesop_normSimp___lam__0___closed__1();
lean_mark_persistent(lp_aesop_Aesop_normSimp___lam__0___closed__1);
lp_aesop_Aesop_normSimp___closed__0 = _init_lp_aesop_Aesop_normSimp___closed__0();
lean_mark_persistent(lp_aesop_Aesop_normSimp___closed__0);
lp_aesop_Aesop_normSimp___closed__1 = _init_lp_aesop_Aesop_normSimp___closed__1();
lean_mark_persistent(lp_aesop_Aesop_normSimp___closed__1);
lp_aesop_Aesop_normSimp___closed__2 = _init_lp_aesop_Aesop_normSimp___closed__2();
lean_mark_persistent(lp_aesop_Aesop_normSimp___closed__2);
lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg___closed__0 = _init_lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_ScriptT_run___at___00Aesop_normUnfoldCore_spec__3___redArg___closed__0);
lp_aesop_Aesop_normUnfoldCore___closed__0 = _init_lp_aesop_Aesop_normUnfoldCore___closed__0();
lean_mark_persistent(lp_aesop_Aesop_normUnfoldCore___closed__0);
lp_aesop_Aesop_normUnfoldCore___closed__1 = _init_lp_aesop_Aesop_normUnfoldCore___closed__1();
lean_mark_persistent(lp_aesop_Aesop_normUnfoldCore___closed__1);
lp_aesop_Aesop_normUnfold___lam__0___closed__0 = _init_lp_aesop_Aesop_normUnfold___lam__0___closed__0();
lean_mark_persistent(lp_aesop_Aesop_normUnfold___lam__0___closed__0);
lp_aesop_Aesop_normUnfold___lam__0___closed__1 = _init_lp_aesop_Aesop_normUnfold___lam__0___closed__1();
lean_mark_persistent(lp_aesop_Aesop_normUnfold___lam__0___closed__1);
lp_aesop_Aesop_normUnfold___closed__0 = _init_lp_aesop_Aesop_normUnfold___closed__0();
lean_mark_persistent(lp_aesop_Aesop_normUnfold___closed__0);
lp_aesop_Aesop_normUnfold___closed__1 = _init_lp_aesop_Aesop_normUnfold___closed__1();
lean_mark_persistent(lp_aesop_Aesop_normUnfold___closed__1);
lp_aesop_Aesop_normUnfold___closed__2 = _init_lp_aesop_Aesop_normUnfold___closed__2();
lean_mark_persistent(lp_aesop_Aesop_normUnfold___closed__2);
lp_aesop_Aesop_NormRuleResult_toNormSeqResult___closed__0 = _init_lp_aesop_Aesop_NormRuleResult_toNormSeqResult___closed__0();
lean_mark_persistent(lp_aesop_Aesop_NormRuleResult_toNormSeqResult___closed__0);
lp_aesop_Aesop_runNormSteps___redArg___closed__0 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__0);
lp_aesop_Aesop_runNormSteps___redArg___closed__2 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__2);
lp_aesop_Aesop_runNormSteps___redArg___closed__1 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__1);
lp_aesop_Aesop_runNormSteps___redArg___closed__3 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__3);
lp_aesop_Aesop_runNormSteps___redArg___closed__4 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__4);
lp_aesop_Aesop_runNormSteps___redArg___closed__5 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__5);
lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__0 = _init_lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__0();
lean_mark_persistent(lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__0);
lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__1 = _init_lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__1();
lean_mark_persistent(lp_aesop___private_Init_While_0__Lean_Loop_forIn_loop___at___00Aesop_runNormSteps_spec__1___closed__1);
lp_aesop_Aesop_runNormSteps___redArg___closed__6 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__6();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__6);
lp_aesop_Aesop_runNormSteps___redArg___closed__7 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__7();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__7);
lp_aesop_Aesop_runNormSteps___redArg___closed__8 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__8();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__8);
lp_aesop_Aesop_runNormSteps___redArg___closed__9 = _init_lp_aesop_Aesop_runNormSteps___redArg___closed__9();
lean_mark_persistent(lp_aesop_Aesop_runNormSteps___redArg___closed__9);
lp_aesop_Aesop_NormStep_unfold___redArg___closed__0 = _init_lp_aesop_Aesop_NormStep_unfold___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_NormStep_unfold___redArg___closed__0);
lp_aesop_Aesop_NormStep_unfold___redArg___closed__1 = _init_lp_aesop_Aesop_NormStep_unfold___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_NormStep_unfold___redArg___closed__1);
lp_aesop_Aesop_NormStep_simp___redArg___closed__0 = _init_lp_aesop_Aesop_NormStep_simp___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_NormStep_simp___redArg___closed__0);
lp_aesop_Aesop_NormStep_simp___redArg___closed__1 = _init_lp_aesop_Aesop_NormStep_simp___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_NormStep_simp___redArg___closed__1);
lp_aesop_Aesop_normalizeGoalMVar___closed__0 = _init_lp_aesop_Aesop_normalizeGoalMVar___closed__0();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalMVar___closed__0);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__0 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__0);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__11 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__11();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__11);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__1 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__1);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__2 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__2);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__3 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__3);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__12 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__12();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__12);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__4 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__4);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__5 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__5);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__6 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__6();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__6);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__7 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__7();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__7);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__8 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__8();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__8);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__9 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__9();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__9);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__10 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__10();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__10);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__13 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__13();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__13);
lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__14 = _init_lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__14();
lean_mark_persistent(lp_aesop_Aesop_normalizeGoalIfNecessary___redArg___closed__14);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
