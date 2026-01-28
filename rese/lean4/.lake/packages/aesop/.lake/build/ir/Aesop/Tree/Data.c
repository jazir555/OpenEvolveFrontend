// Lean compiler output
// Module: Aesop.Tree.Data
// Imports: public import Init public import Aesop.Constants public import Aesop.Script.Step public import Aesop.Tracing public import Aesop.Tree.Data.ForwardRuleMatches public import Aesop.Tree.UnsafeQueue public import Aesop.Forward.State
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parent(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setSuccessProbability(double, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeRulesSelected___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_id(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedNodeState_default;
LEAN_EXPORT double lp_aesop_Aesop_Goal_priority(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RappId_instDecidableRelLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normal_elim(lean_object*, lean_object*, lean_object*, lean_object*);
double lean_float_mul(double, double);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasSafeRapp(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isUnprovable(uint8_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_lastExpandedInIteration(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsForcedUnprovable___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_goals(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormalizationState;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqGoalId_decEq(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormalizationState_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_zero;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_preNormGoal(lean_object*);
uint8_t l_Array_isEmpty___redArg(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasProvableRapp___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasMVar___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIsIrrelevant(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setState___boxed(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_nodeUnknownEmoji;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isProven___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqGoalState_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isIrrelevant___boxed(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setState___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setForwardRuleMatches(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_GoalOrigin_toString___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqRappId_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isIrrelevant(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parentPostNormMetaState___boxed(lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
LEAN_EXPORT double lp_aesop_Aesop_Rapp_successProbability(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_state___boxed(lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__5(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqRappId___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isNormal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isForcedUnprovable___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isUnprovable(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setScriptSteps_x3f(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProvenByRuleApplication(uint8_t);
static lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProvenByRuleApplication___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isUnprovable___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isRoot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeQueue(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_depth(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toEmoji(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_instToString;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeRulesSelected(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setState(uint8_t, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__2(lean_object*);
static lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toEmoji___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_addedInIteration(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setState(uint8_t, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isUnknown(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_depth___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqRappId_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_metaState(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isExhausted(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_one;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_mk(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormalizationState_isProvenByNormalization(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_instToString;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_copied_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_state(lean_object*);
static lean_object* lp_aesop_Aesop_Goal_safeRapps___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_originalSubgoals(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isUnknown___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_successProbability___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_postNormGoalAndMetaState_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqNodeState;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_instHashable;
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isIrrelevant(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_succ___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorIdx___boxed(lean_object*);
LEAN_EXPORT uint64_t lp_aesop_Aesop_Rapp_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isIrrelevant___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedGoalState;
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_dummy;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isIrrelevant(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorIdx___boxed(lean_object*);
static lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProven(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_instHashable;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqNodeState_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData_default(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0(lean_object*);
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isIrrelevant(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isActive___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instBEq___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalOrigin;
lean_object* lean_st_ref_take(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setLastExpandedInIteration(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isIrrelevant___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isIrrelevant(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProvenByNormalization(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_provenByNormalization_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIsIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRappId_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentMetaState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instBEq;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_successProbability___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toCtorIdx(uint8_t);
lean_object* l_Array_empty(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_elim(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toEmoji___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setId(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setParent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_elim(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg(lean_object*);
LEAN_EXPORT double lp_aesop_Aesop_Goal_successProbability(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setSuccessProbability___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_instLT;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsIrrelevant(uint8_t, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_state(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_one;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isUnprovable___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_introducesMVar(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_postNormGoal_x3f(lean_object*);
static lean_object* lp_aesop_Aesop_GoalOrigin_toString___closed__0;
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parent(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instToString;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeQueue(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_assignedMVars(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqGoalId(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_state___boxed(lean_object*);
static lean_object* lp_aesop_Aesop_instBEqGoalState___closed__0;
double lean_float_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProven___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isNormal___boxed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_notNormal_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIntroducedMVars(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setNormalizationState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setSuccessProbability(double, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instHashable___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalId_default;
static lean_object* lp_aesop_Aesop_NodeState_toEmoji___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parentPostNormMetaState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_parent_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_failedRapps(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setForwardState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__5___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_toString(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqNodeState_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqGoalId___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLt___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_GoalId_instToString___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instLT;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_none;
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_mk(lean_object*);
lean_object* l_Lean_DeclNameGenerator_mkChild(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setParent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___redArg(lean_object*);
double pow(double, double);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeQueue_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setParent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__4(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_succ(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_UInt64_ofNat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasSafeRapp___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setPreNormGoal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_instToString;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setAddedInIteration(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isIrrelevant___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_depth(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_children(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasMVar(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isIrrelevant___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProvenByNormalization___boxed(lean_object*);
static lean_object* lp_aesop_Aesop_NodeState_toEmoji___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setState(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint64_t lp_aesop_Aesop_Goal_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_forwardRuleMatches(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_zero;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalId;
extern lean_object* lp_aesop_Aesop_nodeUnprovableEmoji;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_elim(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_isProvenByNormalization___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isForcedUnprovable(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_originalGoalId(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instHashable;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isRoot(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_subgoal_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instHashable;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_introducesMVar___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqGoalState;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_state(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_instToString;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_succ(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqGoalId_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setAssignedMVars(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLe___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_mk(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeRulesSelected___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isUnknown___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat(lean_object*);
static lean_object* lp_aesop_Aesop_GoalOrigin_toString___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setDepth(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentMetaState___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat___boxed(lean_object*);
static double lp_aesop_Aesop_Goal_priority___closed__0;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedNodeState;
static lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_mvars(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqRappId(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0___boxed(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setMetaState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_normalizationState(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setAppliedRule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_origin(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isUnsafeExhausted___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_appliedRule(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setChildren(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_toNodeState(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setChildren(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_introducedMVars(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_succ___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isProven(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toNodeState___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLe(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_provenGoal_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_children(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setIsIrrelevant(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_instLT;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_subgoal_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_notNormal_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instBEqNodeState___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqGoalState_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setMVars(lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setSuccessProbability___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instBEq___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setState___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_succ___boxed(lean_object*);
static lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_one;
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_provenByNormalization_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setOriginalSubgoals(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isSafe(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_id(lean_object*);
static lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRappId;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instBEq;
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0(lean_object*, size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
size_t lean_array_size(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsForcedUnprovable(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_NodeState_toEmoji___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_scriptSteps_x3f(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_copied_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_instDecidableRelLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isSafe___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_succ(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_isNormal___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setFailedRapps(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toCtorIdx(uint8_t);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isUnknown(uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normal_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_modify(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_priority___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_droppedMVar_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalOrigin_default;
lean_object* lean_array_get_size(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_state___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_unsafeRulesSelected(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_GoalId_instHashable___closed__0;
uint8_t lean_usize_dec_lt(size_t, size_t);
uint8_t lp_aesop_Aesop_RegularRule_isSafe(lean_object*);
static lean_object* lp_aesop_Aesop_GoalOrigin_toString___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIteration;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isExhausted___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isActive(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_safeRapps___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setGoals(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isUnsafeExhausted(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_droppedMVar_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_provenGoal_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setOrigin(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasProvableRapp(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_dummy;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setIsIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_modify(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setId(lean_object*, lean_object*);
extern double lp_aesop_Aesop_unificationGoalPenalty;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instLE;
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalId_instDecidableRelLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__3___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_safeRapps(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toEmoji(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instHashable___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_modify(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_forwardState(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedGoalState_default;
extern lean_object* lp_aesop_Aesop_nodeProvedEmoji;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_instDecidableRelLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormalizationState_isNormal(lean_object*);
static lean_object* _init_lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("no ", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("yes", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo(uint8_t x_1) {
_start:
{
if (x_1 == 0)
{
lean_object* x_2; 
x_2 = lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1;
return x_2;
}
else
{
lean_object* x_3; 
x_3 = lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalId_default() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalId() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqGoalId_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqGoalId_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqGoalId_decEq(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqGoalId(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqGoalId___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqGoalId(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_zero() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_one() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(1u);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_succ(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_nat_add(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_succ___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalId_succ(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_dummy() {
_start:
{
lean_object* x_1; 
x_1 = lean_cstr_to_nat("1000000000000000");
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_instLT() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalId_instDecidableRelLt(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_lt(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_instDecidableRelLt___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_GoalId_instDecidableRelLt(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_instToString___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_reprFast), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_instToString() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_GoalId_instToString___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_instHashable___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_UInt64_ofNat___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_instHashable() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_GoalId_instHashable___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRappId_default() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRappId() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqRappId_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqRappId_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqRappId_decEq(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqRappId(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqRappId___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_instDecidableEqRappId(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_zero() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_succ(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_nat_add(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_succ___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_RappId_succ(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_one() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(1u);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_dummy() {
_start:
{
lean_object* x_1; 
x_1 = lean_cstr_to_nat("1000000000000000");
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_instLT() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RappId_instDecidableRelLt(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_lt(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_instDecidableRelLt___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_RappId_instDecidableRelLt(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_instToString() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_GoalId_instToString___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_instHashable() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_GoalId_instHashable___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIteration() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_Iteration_one() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(1u);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_succ(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_nat_add(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_succ___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_Iteration_succ(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_Iteration_none() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableEq(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_eq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Iteration_instDecidableEq(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_Iteration_instToString() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_GoalId_instToString___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_Iteration_instLT() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_Iteration_instLE() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLt(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_lt(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLt___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Iteration_instDecidableRelLt(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLe(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_le(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLe___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Iteration_instDecidableRelLe(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorIdx(uint8_t x_1) {
_start:
{
switch (x_1) {
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorIdx___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_NodeState_ctorIdx(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toCtorIdx(uint8_t x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NodeState_ctorIdx(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toCtorIdx___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_NodeState_toCtorIdx(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lean_unbox(x_3);
x_7 = lp_aesop_Aesop_NodeState_ctorElim(x_1, x_2, x_6, x_4, x_5);
lean_dec(x_5);
lean_dec(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NodeState_ctorElim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_aesop_Aesop_NodeState_unknown_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NodeState_unknown_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_aesop_Aesop_NodeState_proven_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NodeState_proven_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_aesop_Aesop_NodeState_unprovable_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NodeState_unprovable_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedNodeState_default() {
_start:
{
uint8_t x_1; 
x_1 = 0;
return x_1;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedNodeState() {
_start:
{
uint8_t x_1; 
x_1 = 0;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqNodeState_beq(uint8_t x_1, uint8_t x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lp_aesop_Aesop_NodeState_ctorIdx(x_1);
x_4 = lp_aesop_Aesop_NodeState_ctorIdx(x_2);
x_5 = lean_nat_dec_eq(x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqNodeState_beq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_aesop_Aesop_instBEqNodeState_beq(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqNodeState___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instBEqNodeState_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqNodeState() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instBEqNodeState___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("unknown", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("proven", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("unprovable", 10, 10);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0(uint8_t x_1) {
_start:
{
switch (x_1) {
case 0:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0;
return x_2;
}
case 1:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1;
return x_3;
}
default: 
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2;
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_NodeState_instToString___lam__0(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_NodeState_instToString() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_NodeState_instToString___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isUnknown(uint8_t x_1) {
_start:
{
if (x_1 == 0)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isUnknown___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_NodeState_isUnknown(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isProven(uint8_t x_1) {
_start:
{
if (x_1 == 1)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isProven___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_NodeState_isProven(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isUnprovable(uint8_t x_1) {
_start:
{
if (x_1 == 2)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isUnprovable___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_NodeState_isUnprovable(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isIrrelevant(uint8_t x_1) {
_start:
{
if (x_1 == 0)
{
uint8_t x_2; 
x_2 = 0;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 1;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isIrrelevant___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_NodeState_isIrrelevant(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_NodeState_toEmoji___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_nodeUnknownEmoji;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NodeState_toEmoji___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_nodeProvedEmoji;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_NodeState_toEmoji___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_nodeUnprovableEmoji;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toEmoji(uint8_t x_1) {
_start:
{
switch (x_1) {
case 0:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NodeState_toEmoji___closed__0;
return x_2;
}
case 1:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NodeState_toEmoji___closed__1;
return x_3;
}
default: 
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_NodeState_toEmoji___closed__2;
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toEmoji___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_NodeState_toEmoji(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorIdx(uint8_t x_1) {
_start:
{
switch (x_1) {
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
case 2:
{
lean_object* x_4; 
x_4 = lean_unsigned_to_nat(2u);
return x_4;
}
default: 
{
lean_object* x_5; 
x_5 = lean_unsigned_to_nat(3u);
return x_5;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorIdx___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_ctorIdx(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toCtorIdx(uint8_t x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalState_ctorIdx(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toCtorIdx___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_toCtorIdx(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lean_unbox(x_3);
x_7 = lp_aesop_Aesop_GoalState_ctorElim(x_1, x_2, x_6, x_4, x_5);
lean_dec(x_5);
lean_dec(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalState_ctorElim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_aesop_Aesop_GoalState_unknown_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalState_unknown_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_aesop_Aesop_GoalState_provenByRuleApplication_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_aesop_Aesop_GoalState_provenByNormalization_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_aesop_Aesop_GoalState_unprovable_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalState_unprovable_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedGoalState_default() {
_start:
{
uint8_t x_1; 
x_1 = 0;
return x_1;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedGoalState() {
_start:
{
uint8_t x_1; 
x_1 = 0;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqGoalState_beq(uint8_t x_1, uint8_t x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lp_aesop_Aesop_GoalState_ctorIdx(x_1);
x_4 = lp_aesop_Aesop_GoalState_ctorIdx(x_2);
x_5 = lean_nat_dec_eq(x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqGoalState_beq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_aesop_Aesop_instBEqGoalState_beq(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqGoalState___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instBEqGoalState_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instBEqGoalState() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instBEqGoalState___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("provenByRuleApplication", 23, 23);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("provenByNormalization", 21, 21);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0(uint8_t x_1) {
_start:
{
switch (x_1) {
case 0:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0;
return x_2;
}
case 1:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0;
return x_3;
}
case 2:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1;
return x_4;
}
default: 
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2;
return x_5;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_instToString___lam__0(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalState_instToString() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalState_instToString___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProvenByRuleApplication(uint8_t x_1) {
_start:
{
if (x_1 == 1)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProvenByRuleApplication___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_isProvenByRuleApplication(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProvenByNormalization(uint8_t x_1) {
_start:
{
if (x_1 == 2)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProvenByNormalization___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_isProvenByNormalization(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProven(uint8_t x_1) {
_start:
{
switch (x_1) {
case 1:
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
case 2:
{
uint8_t x_3; 
x_3 = 1;
return x_3;
}
default: 
{
uint8_t x_4; 
x_4 = 0;
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProven___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_isProven(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isUnprovable(uint8_t x_1) {
_start:
{
if (x_1 == 3)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isUnprovable___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_isUnprovable(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isUnknown(uint8_t x_1) {
_start:
{
if (x_1 == 0)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isUnknown___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_isUnknown(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_toNodeState(uint8_t x_1) {
_start:
{
switch (x_1) {
case 0:
{
uint8_t x_2; 
x_2 = 0;
return x_2;
}
case 3:
{
uint8_t x_3; 
x_3 = 2;
return x_3;
}
default: 
{
uint8_t x_4; 
x_4 = 1;
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toNodeState___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_toNodeState(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isIrrelevant(uint8_t x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; 
x_2 = lp_aesop_Aesop_GoalState_toNodeState(x_1);
x_3 = lp_aesop_Aesop_NodeState_isIrrelevant(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isIrrelevant___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_isIrrelevant(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toEmoji(uint8_t x_1) {
_start:
{
switch (x_1) {
case 0:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NodeState_toEmoji___closed__0;
return x_2;
}
case 3:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NodeState_toEmoji___closed__2;
return x_3;
}
default: 
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_NodeState_toEmoji___closed__1;
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toEmoji___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_aesop_Aesop_GoalState_toEmoji(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorIdx(lean_object* x_1) {
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorIdx___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NormalizationState_ctorIdx(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
return x_2;
}
case 1:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_3(x_2, x_3, x_4, x_5);
return x_6;
}
default: 
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_8);
lean_dec_ref(x_1);
x_9 = lean_apply_2(x_2, x_7, x_8);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_NormalizationState_ctorElim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_notNormal_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_notNormal_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normal_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normal_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_provenByNormalization_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_provenByNormalization_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormalizationState_default() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormalizationState() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormalizationState_isNormal(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_2; 
x_2 = 0;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 1;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_isNormal___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_NormalizationState_isNormal(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormalizationState_isProvenByNormalization(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 2)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_isProvenByNormalization___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_NormalizationState_isProvenByNormalization(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 1)
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorIdx(lean_object* x_1) {
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorIdx___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalOrigin_ctorIdx(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 1)
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
lean_dec(x_1);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_GoalOrigin_ctorElim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_subgoal_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_subgoal_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_copied_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_copied_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_droppedMVar_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_droppedMVar_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalOrigin_default() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalOrigin() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 1)
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 1);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalOrigin_toString___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("subgoal", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalOrigin_toString___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("copy of ", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalOrigin_toString___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(", originally ", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalOrigin_toString___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("dropped mvar", 12, 12);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_toString(lean_object* x_1) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_GoalOrigin_toString___closed__0;
return x_2;
}
case 1:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lp_aesop_Aesop_GoalOrigin_toString___closed__1;
x_6 = l_Nat_reprFast(x_3);
x_7 = lean_string_append(x_5, x_6);
lean_dec_ref(x_6);
x_8 = lp_aesop_Aesop_GoalOrigin_toString___closed__2;
x_9 = lean_string_append(x_7, x_8);
x_10 = l_Nat_reprFast(x_4);
x_11 = lean_string_append(x_9, x_10);
lean_dec_ref(x_10);
return x_11;
}
default: 
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_GoalOrigin_toString___closed__3;
return x_12;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1() {
_start:
{
uint8_t x_1; uint8_t x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = 0;
x_2 = 0;
x_3 = lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0;
x_4 = lean_box(0);
x_5 = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set_uint8(x_5, sizeof(void*)*2, x_2);
lean_ctor_set_uint8(x_5, sizeof(void*)*2 + 1, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData_default(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1;
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedMVarClusterData_default(lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__2(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__4(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__3(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__5(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__1___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_treeImpl___lam__1(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__3___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_treeImpl___lam__3(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__5___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_treeImpl___lam__5(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_treeImpl() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_treeImpl___lam__0), 1, 0);
x_2 = lean_alloc_closure((void*)(lp_aesop_Aesop_treeImpl___lam__1___boxed), 1, 0);
x_3 = lean_alloc_closure((void*)(lp_aesop_Aesop_treeImpl___lam__2), 1, 0);
x_4 = lean_alloc_closure((void*)(lp_aesop_Aesop_treeImpl___lam__3___boxed), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_aesop_Aesop_treeImpl___lam__4), 1, 0);
x_6 = lean_alloc_closure((void*)(lp_aesop_Aesop_treeImpl___lam__5___boxed), 1, 0);
x_7 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_7, 0, x_1);
lean_ctor_set(x_7, 1, x_2);
lean_ctor_set(x_7, 2, x_3);
lean_ctor_set(x_7, 3, x_4);
lean_ctor_set(x_7, 4, x_5);
lean_ctor_set(x_7, 5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_mk(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 4);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_elim(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 5);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_modify(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 4);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 5);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = lean_apply_1(x_1, x_6);
x_8 = lean_apply_1(x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_parent_x3f(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 5);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setParent(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 4);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 5);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 0);
lean_dec(x_8);
lean_ctor_set(x_6, 0, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; uint8_t x_11; uint8_t x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get_uint8(x_6, sizeof(void*)*2);
x_12 = lean_ctor_get_uint8(x_6, sizeof(void*)*2 + 1);
lean_inc(x_10);
lean_dec(x_6);
x_13 = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(x_13, 0, x_1);
lean_ctor_set(x_13, 1, x_10);
lean_ctor_set_uint8(x_13, sizeof(void*)*2, x_11);
lean_ctor_set_uint8(x_13, sizeof(void*)*2 + 1, x_12);
x_14 = lean_apply_1(x_4, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_goals(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 5);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setGoals(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 4);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 5);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 1);
lean_dec(x_8);
lean_ctor_set(x_6, 1, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; uint8_t x_11; uint8_t x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get_uint8(x_6, sizeof(void*)*2);
x_12 = lean_ctor_get_uint8(x_6, sizeof(void*)*2 + 1);
lean_inc(x_10);
lean_dec(x_6);
x_13 = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(x_13, 0, x_10);
lean_ctor_set(x_13, 1, x_1);
lean_ctor_set_uint8(x_13, sizeof(void*)*2, x_11);
lean_ctor_set_uint8(x_13, sizeof(void*)*2 + 1, x_12);
x_14 = lean_apply_1(x_4, x_13);
return x_14;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isIrrelevant(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 5);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*2);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isIrrelevant___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_MVarCluster_isIrrelevant(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setIsIrrelevant(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 4);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 5);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_uint8(x_6, sizeof(void*)*2, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get_uint8(x_6, sizeof(void*)*2 + 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_12 = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(x_12, 0, x_9);
lean_ctor_set(x_12, 1, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*2, x_1);
lean_ctor_set_uint8(x_12, sizeof(void*)*2 + 1, x_11);
x_13 = lean_apply_1(x_4, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setIsIrrelevant___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_aesop_Aesop_MVarCluster_setIsIrrelevant(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_state(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 5);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*2 + 1);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_state___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_MVarCluster_state(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setState(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 4);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 5);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 1, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get_uint8(x_6, sizeof(void*)*2);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_12 = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(x_12, 0, x_9);
lean_ctor_set(x_12, 1, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*2, x_11);
lean_ctor_set_uint8(x_12, sizeof(void*)*2 + 1, x_1);
x_13 = lean_apply_1(x_4, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setState___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_aesop_Aesop_MVarCluster_setState(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_mk(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_elim(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_modify(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = lean_apply_1(x_1, x_6);
x_8 = lean_apply_1(x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_id(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parent(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_children(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 2);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_origin(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 3);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_depth(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 4);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_state(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*14 + 8);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_state___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_state(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isIrrelevant(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*14 + 9);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isIrrelevant___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_isIrrelevant(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isForcedUnprovable(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*14 + 10);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isForcedUnprovable___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_isForcedUnprovable(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_preNormGoal(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 5);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_normalizationState(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 6);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_mvars(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 7);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_forwardState(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 8);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_forwardRuleMatches(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 9);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT double lp_aesop_Aesop_Goal_successProbability(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; double x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_float(x_4, sizeof(void*)*14);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_successProbability___boxed(lean_object* x_1) {
_start:
{
double x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_successProbability(x_1);
x_3 = lean_box_float(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_addedInIteration(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 10);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_lastExpandedInIteration(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 11);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_failedRapps(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 13);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_unsafeRulesSelected(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*14 + 11);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeRulesSelected___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_unsafeRulesSelected(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeQueue(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 12);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeQueue_x3f(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*14 + 11);
if (x_5 == 0)
{
lean_object* x_6; 
lean_dec_ref(x_4);
x_6 = lean_box(0);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_4, 12);
lean_inc_ref(x_7);
lean_dec_ref(x_4);
x_8 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setId(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 0);
lean_dec(x_8);
lean_ctor_set(x_6, 0, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get(x_6, 3);
x_13 = lean_ctor_get(x_6, 4);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_1);
lean_ctor_set(x_28, 1, x_10);
lean_ctor_set(x_28, 2, x_11);
lean_ctor_set(x_28, 3, x_12);
lean_ctor_set(x_28, 4, x_13);
lean_ctor_set(x_28, 5, x_17);
lean_ctor_set(x_28, 6, x_18);
lean_ctor_set(x_28, 7, x_19);
lean_ctor_set(x_28, 8, x_20);
lean_ctor_set(x_28, 9, x_21);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_14);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_16);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setParent(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 1);
lean_dec(x_8);
lean_ctor_set(x_6, 1, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get(x_6, 3);
x_13 = lean_ctor_get(x_6, 4);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_1);
lean_ctor_set(x_28, 2, x_11);
lean_ctor_set(x_28, 3, x_12);
lean_ctor_set(x_28, 4, x_13);
lean_ctor_set(x_28, 5, x_17);
lean_ctor_set(x_28, 6, x_18);
lean_ctor_set(x_28, 7, x_19);
lean_ctor_set(x_28, 8, x_20);
lean_ctor_set(x_28, 9, x_21);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_14);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_16);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setChildren(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 2);
lean_dec(x_8);
lean_ctor_set(x_6, 2, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 3);
x_13 = lean_ctor_get(x_6, 4);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_1);
lean_ctor_set(x_28, 3, x_12);
lean_ctor_set(x_28, 4, x_13);
lean_ctor_set(x_28, 5, x_17);
lean_ctor_set(x_28, 6, x_18);
lean_ctor_set(x_28, 7, x_19);
lean_ctor_set(x_28, 8, x_20);
lean_ctor_set(x_28, 9, x_21);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_14);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_16);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setOrigin(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 3);
lean_dec(x_8);
lean_ctor_set(x_6, 3, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 4);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_1);
lean_ctor_set(x_28, 4, x_13);
lean_ctor_set(x_28, 5, x_17);
lean_ctor_set(x_28, 6, x_18);
lean_ctor_set(x_28, 7, x_19);
lean_ctor_set(x_28, 8, x_20);
lean_ctor_set(x_28, 9, x_21);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_14);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_16);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setDepth(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 4);
lean_dec(x_8);
lean_ctor_set(x_6, 4, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_1);
lean_ctor_set(x_28, 5, x_17);
lean_ctor_set(x_28, 6, x_18);
lean_ctor_set(x_28, 7, x_19);
lean_ctor_set(x_28, 8, x_20);
lean_ctor_set(x_28, 9, x_21);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_14);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_16);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsIrrelevant(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_uint8(x_6, sizeof(void*)*14 + 9, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; double x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get(x_6, 3);
x_13 = lean_ctor_get(x_6, 4);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_16 = lean_ctor_get(x_6, 5);
x_17 = lean_ctor_get(x_6, 6);
x_18 = lean_ctor_get(x_6, 7);
x_19 = lean_ctor_get(x_6, 8);
x_20 = lean_ctor_get(x_6, 9);
x_21 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_22 = lean_ctor_get(x_6, 10);
x_23 = lean_ctor_get(x_6, 11);
x_24 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_25 = lean_ctor_get(x_6, 12);
x_26 = lean_ctor_get(x_6, 13);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_27 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_27, 0, x_9);
lean_ctor_set(x_27, 1, x_10);
lean_ctor_set(x_27, 2, x_11);
lean_ctor_set(x_27, 3, x_12);
lean_ctor_set(x_27, 4, x_13);
lean_ctor_set(x_27, 5, x_16);
lean_ctor_set(x_27, 6, x_17);
lean_ctor_set(x_27, 7, x_18);
lean_ctor_set(x_27, 8, x_19);
lean_ctor_set(x_27, 9, x_20);
lean_ctor_set(x_27, 10, x_22);
lean_ctor_set(x_27, 11, x_23);
lean_ctor_set(x_27, 12, x_25);
lean_ctor_set(x_27, 13, x_26);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 8, x_14);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 9, x_1);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 10, x_15);
lean_ctor_set_float(x_27, sizeof(void*)*14, x_21);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 11, x_24);
x_28 = lean_apply_1(x_4, x_27);
return x_28;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsIrrelevant___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_aesop_Aesop_Goal_setIsIrrelevant(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsForcedUnprovable(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_uint8(x_6, sizeof(void*)*14 + 10, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; double x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get(x_6, 3);
x_13 = lean_ctor_get(x_6, 4);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_16 = lean_ctor_get(x_6, 5);
x_17 = lean_ctor_get(x_6, 6);
x_18 = lean_ctor_get(x_6, 7);
x_19 = lean_ctor_get(x_6, 8);
x_20 = lean_ctor_get(x_6, 9);
x_21 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_22 = lean_ctor_get(x_6, 10);
x_23 = lean_ctor_get(x_6, 11);
x_24 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_25 = lean_ctor_get(x_6, 12);
x_26 = lean_ctor_get(x_6, 13);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_27 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_27, 0, x_9);
lean_ctor_set(x_27, 1, x_10);
lean_ctor_set(x_27, 2, x_11);
lean_ctor_set(x_27, 3, x_12);
lean_ctor_set(x_27, 4, x_13);
lean_ctor_set(x_27, 5, x_16);
lean_ctor_set(x_27, 6, x_17);
lean_ctor_set(x_27, 7, x_18);
lean_ctor_set(x_27, 8, x_19);
lean_ctor_set(x_27, 9, x_20);
lean_ctor_set(x_27, 10, x_22);
lean_ctor_set(x_27, 11, x_23);
lean_ctor_set(x_27, 12, x_25);
lean_ctor_set(x_27, 13, x_26);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 8, x_14);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 9, x_15);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 10, x_1);
lean_ctor_set_float(x_27, sizeof(void*)*14, x_21);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 11, x_24);
x_28 = lean_apply_1(x_4, x_27);
return x_28;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsForcedUnprovable___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_aesop_Aesop_Goal_setIsForcedUnprovable(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setPreNormGoal(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 5);
lean_dec(x_8);
lean_ctor_set(x_6, 5, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; uint8_t x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_14);
lean_ctor_set(x_28, 5, x_1);
lean_ctor_set(x_28, 6, x_18);
lean_ctor_set(x_28, 7, x_19);
lean_ctor_set(x_28, 8, x_20);
lean_ctor_set(x_28, 9, x_21);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_16);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_17);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setNormalizationState(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 6);
lean_dec(x_8);
lean_ctor_set(x_6, 6, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; uint8_t x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_18 = lean_ctor_get(x_6, 5);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_14);
lean_ctor_set(x_28, 5, x_18);
lean_ctor_set(x_28, 6, x_1);
lean_ctor_set(x_28, 7, x_19);
lean_ctor_set(x_28, 8, x_20);
lean_ctor_set(x_28, 9, x_21);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_16);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_17);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setMVars(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 7);
lean_dec(x_8);
lean_ctor_set(x_6, 7, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; uint8_t x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_18 = lean_ctor_get(x_6, 5);
x_19 = lean_ctor_get(x_6, 6);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_14);
lean_ctor_set(x_28, 5, x_18);
lean_ctor_set(x_28, 6, x_19);
lean_ctor_set(x_28, 7, x_1);
lean_ctor_set(x_28, 8, x_20);
lean_ctor_set(x_28, 9, x_21);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_16);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_17);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setForwardState(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 8);
lean_dec(x_8);
lean_ctor_set(x_6, 8, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; uint8_t x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_18 = lean_ctor_get(x_6, 5);
x_19 = lean_ctor_get(x_6, 6);
x_20 = lean_ctor_get(x_6, 7);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_14);
lean_ctor_set(x_28, 5, x_18);
lean_ctor_set(x_28, 6, x_19);
lean_ctor_set(x_28, 7, x_20);
lean_ctor_set(x_28, 8, x_1);
lean_ctor_set(x_28, 9, x_21);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_16);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_17);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setForwardRuleMatches(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 9);
lean_dec(x_8);
lean_ctor_set(x_6, 9, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; uint8_t x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_18 = lean_ctor_get(x_6, 5);
x_19 = lean_ctor_get(x_6, 6);
x_20 = lean_ctor_get(x_6, 7);
x_21 = lean_ctor_get(x_6, 8);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_14);
lean_ctor_set(x_28, 5, x_18);
lean_ctor_set(x_28, 6, x_19);
lean_ctor_set(x_28, 7, x_20);
lean_ctor_set(x_28, 8, x_21);
lean_ctor_set(x_28, 9, x_1);
lean_ctor_set(x_28, 10, x_23);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_16);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_17);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setSuccessProbability(double x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_float(x_6, sizeof(void*)*14, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get(x_6, 3);
x_13 = lean_ctor_get(x_6, 4);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get(x_6, 10);
x_23 = lean_ctor_get(x_6, 11);
x_24 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_25 = lean_ctor_get(x_6, 12);
x_26 = lean_ctor_get(x_6, 13);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_27 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_27, 0, x_9);
lean_ctor_set(x_27, 1, x_10);
lean_ctor_set(x_27, 2, x_11);
lean_ctor_set(x_27, 3, x_12);
lean_ctor_set(x_27, 4, x_13);
lean_ctor_set(x_27, 5, x_17);
lean_ctor_set(x_27, 6, x_18);
lean_ctor_set(x_27, 7, x_19);
lean_ctor_set(x_27, 8, x_20);
lean_ctor_set(x_27, 9, x_21);
lean_ctor_set(x_27, 10, x_22);
lean_ctor_set(x_27, 11, x_23);
lean_ctor_set(x_27, 12, x_25);
lean_ctor_set(x_27, 13, x_26);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 8, x_14);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 9, x_15);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 10, x_16);
lean_ctor_set_float(x_27, sizeof(void*)*14, x_1);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 11, x_24);
x_28 = lean_apply_1(x_4, x_27);
return x_28;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setSuccessProbability___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
double x_3; lean_object* x_4; 
x_3 = lean_unbox_float(x_1);
lean_dec_ref(x_1);
x_4 = lp_aesop_Aesop_Goal_setSuccessProbability(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setAddedInIteration(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 10);
lean_dec(x_8);
lean_ctor_set(x_6, 10, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; uint8_t x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; double x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_18 = lean_ctor_get(x_6, 5);
x_19 = lean_ctor_get(x_6, 6);
x_20 = lean_ctor_get(x_6, 7);
x_21 = lean_ctor_get(x_6, 8);
x_22 = lean_ctor_get(x_6, 9);
x_23 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_14);
lean_ctor_set(x_28, 5, x_18);
lean_ctor_set(x_28, 6, x_19);
lean_ctor_set(x_28, 7, x_20);
lean_ctor_set(x_28, 8, x_21);
lean_ctor_set(x_28, 9, x_22);
lean_ctor_set(x_28, 10, x_1);
lean_ctor_set(x_28, 11, x_24);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_16);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_17);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_23);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setLastExpandedInIteration(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 11);
lean_dec(x_8);
lean_ctor_set(x_6, 11, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; uint8_t x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; double x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_18 = lean_ctor_get(x_6, 5);
x_19 = lean_ctor_get(x_6, 6);
x_20 = lean_ctor_get(x_6, 7);
x_21 = lean_ctor_get(x_6, 8);
x_22 = lean_ctor_get(x_6, 9);
x_23 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_24 = lean_ctor_get(x_6, 10);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_26 = lean_ctor_get(x_6, 12);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_24);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_14);
lean_ctor_set(x_28, 5, x_18);
lean_ctor_set(x_28, 6, x_19);
lean_ctor_set(x_28, 7, x_20);
lean_ctor_set(x_28, 8, x_21);
lean_ctor_set(x_28, 9, x_22);
lean_ctor_set(x_28, 10, x_24);
lean_ctor_set(x_28, 11, x_1);
lean_ctor_set(x_28, 12, x_26);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_16);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_17);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_23);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_25);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeRulesSelected(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_uint8(x_6, sizeof(void*)*14 + 11, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; double x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get(x_6, 3);
x_13 = lean_ctor_get(x_6, 4);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
x_21 = lean_ctor_get(x_6, 9);
x_22 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_23 = lean_ctor_get(x_6, 10);
x_24 = lean_ctor_get(x_6, 11);
x_25 = lean_ctor_get(x_6, 12);
x_26 = lean_ctor_get(x_6, 13);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_27 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_27, 0, x_9);
lean_ctor_set(x_27, 1, x_10);
lean_ctor_set(x_27, 2, x_11);
lean_ctor_set(x_27, 3, x_12);
lean_ctor_set(x_27, 4, x_13);
lean_ctor_set(x_27, 5, x_17);
lean_ctor_set(x_27, 6, x_18);
lean_ctor_set(x_27, 7, x_19);
lean_ctor_set(x_27, 8, x_20);
lean_ctor_set(x_27, 9, x_21);
lean_ctor_set(x_27, 10, x_23);
lean_ctor_set(x_27, 11, x_24);
lean_ctor_set(x_27, 12, x_25);
lean_ctor_set(x_27, 13, x_26);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 8, x_14);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 9, x_15);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 10, x_16);
lean_ctor_set_float(x_27, sizeof(void*)*14, x_22);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 11, x_1);
x_28 = lean_apply_1(x_4, x_27);
return x_28;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeRulesSelected___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_aesop_Aesop_Goal_setUnsafeRulesSelected(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeQueue(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 12);
lean_dec(x_8);
lean_ctor_set(x_6, 12, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; uint8_t x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; double x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_18 = lean_ctor_get(x_6, 5);
x_19 = lean_ctor_get(x_6, 6);
x_20 = lean_ctor_get(x_6, 7);
x_21 = lean_ctor_get(x_6, 8);
x_22 = lean_ctor_get(x_6, 9);
x_23 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_24 = lean_ctor_get(x_6, 10);
x_25 = lean_ctor_get(x_6, 11);
x_26 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_27 = lean_ctor_get(x_6, 13);
lean_inc(x_27);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_14);
lean_ctor_set(x_28, 5, x_18);
lean_ctor_set(x_28, 6, x_19);
lean_ctor_set(x_28, 7, x_20);
lean_ctor_set(x_28, 8, x_21);
lean_ctor_set(x_28, 9, x_22);
lean_ctor_set(x_28, 10, x_24);
lean_ctor_set(x_28, 11, x_25);
lean_ctor_set(x_28, 12, x_1);
lean_ctor_set(x_28, 13, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_16);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_17);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_23);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_26);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setState(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_uint8(x_6, sizeof(void*)*14 + 8, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; double x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get(x_6, 3);
x_13 = lean_ctor_get(x_6, 4);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_16 = lean_ctor_get(x_6, 5);
x_17 = lean_ctor_get(x_6, 6);
x_18 = lean_ctor_get(x_6, 7);
x_19 = lean_ctor_get(x_6, 8);
x_20 = lean_ctor_get(x_6, 9);
x_21 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_22 = lean_ctor_get(x_6, 10);
x_23 = lean_ctor_get(x_6, 11);
x_24 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_25 = lean_ctor_get(x_6, 12);
x_26 = lean_ctor_get(x_6, 13);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_27 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_27, 0, x_9);
lean_ctor_set(x_27, 1, x_10);
lean_ctor_set(x_27, 2, x_11);
lean_ctor_set(x_27, 3, x_12);
lean_ctor_set(x_27, 4, x_13);
lean_ctor_set(x_27, 5, x_16);
lean_ctor_set(x_27, 6, x_17);
lean_ctor_set(x_27, 7, x_18);
lean_ctor_set(x_27, 8, x_19);
lean_ctor_set(x_27, 9, x_20);
lean_ctor_set(x_27, 10, x_22);
lean_ctor_set(x_27, 11, x_23);
lean_ctor_set(x_27, 12, x_25);
lean_ctor_set(x_27, 13, x_26);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 8, x_1);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 9, x_14);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 10, x_15);
lean_ctor_set_float(x_27, sizeof(void*)*14, x_21);
lean_ctor_set_uint8(x_27, sizeof(void*)*14 + 11, x_24);
x_28 = lean_apply_1(x_4, x_27);
return x_28;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setState___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_aesop_Aesop_Goal_setState(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setFailedRapps(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 13);
lean_dec(x_8);
lean_ctor_set(x_6, 13, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; uint8_t x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; double x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 8);
x_16 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 9);
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 10);
x_18 = lean_ctor_get(x_6, 5);
x_19 = lean_ctor_get(x_6, 6);
x_20 = lean_ctor_get(x_6, 7);
x_21 = lean_ctor_get(x_6, 8);
x_22 = lean_ctor_get(x_6, 9);
x_23 = lean_ctor_get_float(x_6, sizeof(void*)*14);
x_24 = lean_ctor_get(x_6, 10);
x_25 = lean_ctor_get(x_6, 11);
x_26 = lean_ctor_get_uint8(x_6, sizeof(void*)*14 + 11);
x_27 = lean_ctor_get(x_6, 12);
lean_inc(x_27);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_28 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_28, 0, x_10);
lean_ctor_set(x_28, 1, x_11);
lean_ctor_set(x_28, 2, x_12);
lean_ctor_set(x_28, 3, x_13);
lean_ctor_set(x_28, 4, x_14);
lean_ctor_set(x_28, 5, x_18);
lean_ctor_set(x_28, 6, x_19);
lean_ctor_set(x_28, 7, x_20);
lean_ctor_set(x_28, 8, x_21);
lean_ctor_set(x_28, 9, x_22);
lean_ctor_set(x_28, 10, x_24);
lean_ctor_set(x_28, 11, x_25);
lean_ctor_set(x_28, 12, x_27);
lean_ctor_set(x_28, 13, x_1);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 8, x_15);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 9, x_16);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 10, x_17);
lean_ctor_set_float(x_28, sizeof(void*)*14, x_23);
lean_ctor_set_uint8(x_28, sizeof(void*)*14 + 11, x_26);
x_29 = lean_apply_1(x_4, x_28);
return x_29;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_instBEq___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_4, x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_1(x_4, x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_nat_dec_eq(x_6, x_8);
lean_dec(x_8);
lean_dec(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instBEq___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Goal_instBEq___lam__0(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_instBEq() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_instBEq___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Goal_instHashable___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint64_t x_6; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_uint64_of_nat(x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instHashable___lam__0___boxed(lean_object* x_1) {
_start:
{
uint64_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_instHashable___lam__0(x_1);
x_3 = lean_box_uint64(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_instHashable() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_instHashable___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_mk(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 2);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_elim(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_modify(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = lean_apply_1(x_1, x_6);
x_8 = lean_apply_1(x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_id(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parent(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_children(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 2);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_state(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*9 + 8);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_state___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Rapp_state(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isIrrelevant(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*9 + 9);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isIrrelevant___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Rapp_isIrrelevant(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_appliedRule(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 3);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_scriptSteps_x3f(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 4);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_originalSubgoals(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 5);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT double lp_aesop_Aesop_Rapp_successProbability(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; double x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_float(x_4, sizeof(void*)*9);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_successProbability___boxed(lean_object* x_1) {
_start:
{
double x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Rapp_successProbability(x_1);
x_3 = lean_box_float(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_metaState(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 6);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_introducedMVars(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 7);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_assignedMVars(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 8);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setId(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 0);
lean_dec(x_8);
lean_ctor_set(x_6, 0, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; double x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_14 = lean_ctor_get(x_6, 3);
x_15 = lean_ctor_get(x_6, 4);
x_16 = lean_ctor_get(x_6, 5);
x_17 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_21 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_21, 0, x_1);
lean_ctor_set(x_21, 1, x_10);
lean_ctor_set(x_21, 2, x_11);
lean_ctor_set(x_21, 3, x_14);
lean_ctor_set(x_21, 4, x_15);
lean_ctor_set(x_21, 5, x_16);
lean_ctor_set(x_21, 6, x_18);
lean_ctor_set(x_21, 7, x_19);
lean_ctor_set(x_21, 8, x_20);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 8, x_12);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 9, x_13);
lean_ctor_set_float(x_21, sizeof(void*)*9, x_17);
x_22 = lean_apply_1(x_4, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setParent(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 1);
lean_dec(x_8);
lean_ctor_set(x_6, 1, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; double x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_14 = lean_ctor_get(x_6, 3);
x_15 = lean_ctor_get(x_6, 4);
x_16 = lean_ctor_get(x_6, 5);
x_17 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_21 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_21, 0, x_10);
lean_ctor_set(x_21, 1, x_1);
lean_ctor_set(x_21, 2, x_11);
lean_ctor_set(x_21, 3, x_14);
lean_ctor_set(x_21, 4, x_15);
lean_ctor_set(x_21, 5, x_16);
lean_ctor_set(x_21, 6, x_18);
lean_ctor_set(x_21, 7, x_19);
lean_ctor_set(x_21, 8, x_20);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 8, x_12);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 9, x_13);
lean_ctor_set_float(x_21, sizeof(void*)*9, x_17);
x_22 = lean_apply_1(x_4, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setChildren(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 2);
lean_dec(x_8);
lean_ctor_set(x_6, 2, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; double x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_14 = lean_ctor_get(x_6, 3);
x_15 = lean_ctor_get(x_6, 4);
x_16 = lean_ctor_get(x_6, 5);
x_17 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_21 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_21, 0, x_10);
lean_ctor_set(x_21, 1, x_11);
lean_ctor_set(x_21, 2, x_1);
lean_ctor_set(x_21, 3, x_14);
lean_ctor_set(x_21, 4, x_15);
lean_ctor_set(x_21, 5, x_16);
lean_ctor_set(x_21, 6, x_18);
lean_ctor_set(x_21, 7, x_19);
lean_ctor_set(x_21, 8, x_20);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 8, x_12);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 9, x_13);
lean_ctor_set_float(x_21, sizeof(void*)*9, x_17);
x_22 = lean_apply_1(x_4, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setState(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_uint8(x_6, sizeof(void*)*9 + 8, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; double x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get(x_6, 5);
x_16 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_17 = lean_ctor_get(x_6, 6);
x_18 = lean_ctor_get(x_6, 7);
x_19 = lean_ctor_get(x_6, 8);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_20 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_20, 0, x_9);
lean_ctor_set(x_20, 1, x_10);
lean_ctor_set(x_20, 2, x_11);
lean_ctor_set(x_20, 3, x_13);
lean_ctor_set(x_20, 4, x_14);
lean_ctor_set(x_20, 5, x_15);
lean_ctor_set(x_20, 6, x_17);
lean_ctor_set(x_20, 7, x_18);
lean_ctor_set(x_20, 8, x_19);
lean_ctor_set_uint8(x_20, sizeof(void*)*9 + 8, x_1);
lean_ctor_set_uint8(x_20, sizeof(void*)*9 + 9, x_12);
lean_ctor_set_float(x_20, sizeof(void*)*9, x_16);
x_21 = lean_apply_1(x_4, x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setState___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_aesop_Aesop_Rapp_setState(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIsIrrelevant(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_uint8(x_6, sizeof(void*)*9 + 9, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; double x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_13 = lean_ctor_get(x_6, 3);
x_14 = lean_ctor_get(x_6, 4);
x_15 = lean_ctor_get(x_6, 5);
x_16 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_17 = lean_ctor_get(x_6, 6);
x_18 = lean_ctor_get(x_6, 7);
x_19 = lean_ctor_get(x_6, 8);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_20 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_20, 0, x_9);
lean_ctor_set(x_20, 1, x_10);
lean_ctor_set(x_20, 2, x_11);
lean_ctor_set(x_20, 3, x_13);
lean_ctor_set(x_20, 4, x_14);
lean_ctor_set(x_20, 5, x_15);
lean_ctor_set(x_20, 6, x_17);
lean_ctor_set(x_20, 7, x_18);
lean_ctor_set(x_20, 8, x_19);
lean_ctor_set_uint8(x_20, sizeof(void*)*9 + 8, x_12);
lean_ctor_set_uint8(x_20, sizeof(void*)*9 + 9, x_1);
lean_ctor_set_float(x_20, sizeof(void*)*9, x_16);
x_21 = lean_apply_1(x_4, x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIsIrrelevant___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_aesop_Aesop_Rapp_setIsIrrelevant(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setAppliedRule(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 3);
lean_dec(x_8);
lean_ctor_set(x_6, 3, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; double x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_15 = lean_ctor_get(x_6, 4);
x_16 = lean_ctor_get(x_6, 5);
x_17 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_21 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_21, 0, x_10);
lean_ctor_set(x_21, 1, x_11);
lean_ctor_set(x_21, 2, x_12);
lean_ctor_set(x_21, 3, x_1);
lean_ctor_set(x_21, 4, x_15);
lean_ctor_set(x_21, 5, x_16);
lean_ctor_set(x_21, 6, x_18);
lean_ctor_set(x_21, 7, x_19);
lean_ctor_set(x_21, 8, x_20);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 8, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 9, x_14);
lean_ctor_set_float(x_21, sizeof(void*)*9, x_17);
x_22 = lean_apply_1(x_4, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setScriptSteps_x3f(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 4);
lean_dec(x_8);
lean_ctor_set(x_6, 4, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; double x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_15 = lean_ctor_get(x_6, 3);
x_16 = lean_ctor_get(x_6, 5);
x_17 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_21 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_21, 0, x_10);
lean_ctor_set(x_21, 1, x_11);
lean_ctor_set(x_21, 2, x_12);
lean_ctor_set(x_21, 3, x_15);
lean_ctor_set(x_21, 4, x_1);
lean_ctor_set(x_21, 5, x_16);
lean_ctor_set(x_21, 6, x_18);
lean_ctor_set(x_21, 7, x_19);
lean_ctor_set(x_21, 8, x_20);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 8, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 9, x_14);
lean_ctor_set_float(x_21, sizeof(void*)*9, x_17);
x_22 = lean_apply_1(x_4, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setOriginalSubgoals(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 5);
lean_dec(x_8);
lean_ctor_set(x_6, 5, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; double x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_15 = lean_ctor_get(x_6, 3);
x_16 = lean_ctor_get(x_6, 4);
x_17 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_18 = lean_ctor_get(x_6, 6);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_21 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_21, 0, x_10);
lean_ctor_set(x_21, 1, x_11);
lean_ctor_set(x_21, 2, x_12);
lean_ctor_set(x_21, 3, x_15);
lean_ctor_set(x_21, 4, x_16);
lean_ctor_set(x_21, 5, x_1);
lean_ctor_set(x_21, 6, x_18);
lean_ctor_set(x_21, 7, x_19);
lean_ctor_set(x_21, 8, x_20);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 8, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 9, x_14);
lean_ctor_set_float(x_21, sizeof(void*)*9, x_17);
x_22 = lean_apply_1(x_4, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setSuccessProbability(double x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_ctor_set_float(x_6, sizeof(void*)*9, x_1);
x_8 = lean_apply_1(x_4, x_6);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_ctor_get(x_6, 2);
x_12 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_14 = lean_ctor_get(x_6, 3);
x_15 = lean_ctor_get(x_6, 4);
x_16 = lean_ctor_get(x_6, 5);
x_17 = lean_ctor_get(x_6, 6);
x_18 = lean_ctor_get(x_6, 7);
x_19 = lean_ctor_get(x_6, 8);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_6);
x_20 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_20, 0, x_9);
lean_ctor_set(x_20, 1, x_10);
lean_ctor_set(x_20, 2, x_11);
lean_ctor_set(x_20, 3, x_14);
lean_ctor_set(x_20, 4, x_15);
lean_ctor_set(x_20, 5, x_16);
lean_ctor_set(x_20, 6, x_17);
lean_ctor_set(x_20, 7, x_18);
lean_ctor_set(x_20, 8, x_19);
lean_ctor_set_uint8(x_20, sizeof(void*)*9 + 8, x_12);
lean_ctor_set_uint8(x_20, sizeof(void*)*9 + 9, x_13);
lean_ctor_set_float(x_20, sizeof(void*)*9, x_1);
x_21 = lean_apply_1(x_4, x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setSuccessProbability___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
double x_3; lean_object* x_4; 
x_3 = lean_unbox_float(x_1);
lean_dec_ref(x_1);
x_4 = lp_aesop_Aesop_Rapp_setSuccessProbability(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setMetaState(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 6);
lean_dec(x_8);
lean_ctor_set(x_6, 6, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; double x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_15 = lean_ctor_get(x_6, 3);
x_16 = lean_ctor_get(x_6, 4);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_19 = lean_ctor_get(x_6, 7);
x_20 = lean_ctor_get(x_6, 8);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_21 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_21, 0, x_10);
lean_ctor_set(x_21, 1, x_11);
lean_ctor_set(x_21, 2, x_12);
lean_ctor_set(x_21, 3, x_15);
lean_ctor_set(x_21, 4, x_16);
lean_ctor_set(x_21, 5, x_17);
lean_ctor_set(x_21, 6, x_1);
lean_ctor_set(x_21, 7, x_19);
lean_ctor_set(x_21, 8, x_20);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 8, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 9, x_14);
lean_ctor_set_float(x_21, sizeof(void*)*9, x_18);
x_22 = lean_apply_1(x_4, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIntroducedMVars(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 7);
lean_dec(x_8);
lean_ctor_set(x_6, 7, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; double x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_15 = lean_ctor_get(x_6, 3);
x_16 = lean_ctor_get(x_6, 4);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_19 = lean_ctor_get(x_6, 6);
x_20 = lean_ctor_get(x_6, 8);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_21 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_21, 0, x_10);
lean_ctor_set(x_21, 1, x_11);
lean_ctor_set(x_21, 2, x_12);
lean_ctor_set(x_21, 3, x_15);
lean_ctor_set(x_21, 4, x_16);
lean_ctor_set(x_21, 5, x_17);
lean_ctor_set(x_21, 6, x_19);
lean_ctor_set(x_21, 7, x_1);
lean_ctor_set(x_21, 8, x_20);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 8, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 9, x_14);
lean_ctor_set_float(x_21, sizeof(void*)*9, x_18);
x_22 = lean_apply_1(x_4, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setAssignedMVars(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_2);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_6, 8);
lean_dec(x_8);
lean_ctor_set(x_6, 8, x_1);
x_9 = lean_apply_1(x_4, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; double x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
x_12 = lean_ctor_get(x_6, 2);
x_13 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 8);
x_14 = lean_ctor_get_uint8(x_6, sizeof(void*)*9 + 9);
x_15 = lean_ctor_get(x_6, 3);
x_16 = lean_ctor_get(x_6, 4);
x_17 = lean_ctor_get(x_6, 5);
x_18 = lean_ctor_get_float(x_6, sizeof(void*)*9);
x_19 = lean_ctor_get(x_6, 6);
x_20 = lean_ctor_get(x_6, 7);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_6);
x_21 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_21, 0, x_10);
lean_ctor_set(x_21, 1, x_11);
lean_ctor_set(x_21, 2, x_12);
lean_ctor_set(x_21, 3, x_15);
lean_ctor_set(x_21, 4, x_16);
lean_ctor_set(x_21, 5, x_17);
lean_ctor_set(x_21, 6, x_19);
lean_ctor_set(x_21, 7, x_20);
lean_ctor_set(x_21, 8, x_1);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 8, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*9 + 9, x_14);
lean_ctor_set_float(x_21, sizeof(void*)*9, x_18);
x_22 = lean_apply_1(x_4, x_21);
return x_22;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_instBEq___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_4);
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_4, x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_1(x_4, x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_nat_dec_eq(x_6, x_8);
lean_dec(x_8);
lean_dec(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instBEq___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Rapp_instBEq___lam__0(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_instBEq() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_instBEq___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Rapp_instHashable___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint64_t x_6; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_uint64_of_nat(x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instHashable___lam__0___boxed(lean_object* x_1) {
_start:
{
uint64_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Rapp_instHashable___lam__0(x_1);
x_3 = lean_box_uint64(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_Rapp_instHashable() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_instHashable___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = l_Array_isEmpty___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isSafe(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 3);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_4, 8);
lean_inc_ref(x_6);
lean_dec_ref(x_4);
x_7 = lp_aesop_Aesop_RegularRule_isSafe(x_5);
lean_dec_ref(x_5);
if (x_7 == 0)
{
lean_dec_ref(x_6);
return x_7;
}
else
{
uint8_t x_8; 
x_8 = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(x_6);
lean_dec_ref(x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(x_1);
lean_dec_ref(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isSafe___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Rapp_isSafe(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_postNormGoalAndMetaState_x3f(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 6);
lean_inc(x_5);
lean_dec_ref(x_4);
if (lean_obj_tag(x_5) == 1)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
x_9 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
else
{
lean_object* x_10; 
lean_dec(x_5);
x_10 = lean_box(0);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_postNormGoal_x3f(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 6);
lean_inc(x_5);
lean_dec_ref(x_4);
if (lean_obj_tag(x_5) == 1)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
else
{
lean_object* x_8; 
lean_dec(x_5);
x_8 = lean_box(0);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoal(lean_object* x_1) {
_start:
{
lean_object* x_2; 
lean_inc(x_1);
x_2 = lp_aesop_Aesop_Goal_postNormGoal_x3f(x_1);
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_4, x_1);
x_6 = lean_ctor_get(x_5, 5);
lean_inc(x_6);
lean_dec_ref(x_5);
return x_6;
}
else
{
lean_object* x_7; 
lean_dec(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec_ref(x_2);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_3, 5);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_4, x_1);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_st_ref_get(x_7);
lean_dec(x_7);
x_9 = lean_apply_1(x_5, x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_Goal_parentRapp_x3f(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentMetaState(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_Goal_parentRapp_x3f(x_1);
if (lean_obj_tag(x_4) == 0)
{
lean_inc_ref(x_2);
return x_2;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_st_ref_get(x_5);
lean_dec(x_5);
x_7 = lp_aesop_Aesop_treeImpl;
x_8 = lean_ctor_get(x_7, 3);
lean_inc_ref(x_8);
x_9 = lean_apply_1(x_8, x_6);
x_10 = lean_ctor_get(x_9, 6);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentMetaState___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_Goal_parentMetaState(x_1, x_2);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_aesop_Aesop_treeImpl;
x_5 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_5);
lean_inc(x_1);
x_6 = lean_apply_1(x_5, x_1);
x_7 = lean_ctor_get(x_6, 6);
lean_inc(x_7);
if (lean_obj_tag(x_7) == 1)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_dec_ref(x_6);
lean_dec(x_1);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_9);
lean_dec_ref(x_7);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_9);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_dec(x_7);
x_12 = lean_ctor_get(x_6, 5);
lean_inc(x_12);
lean_dec_ref(x_6);
x_13 = lp_aesop_Aesop_Goal_parentMetaState(x_1, x_2);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(x_1, x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_Goal_currentGoalAndMetaState(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(x_1, x_2);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4) {
_start:
{
uint8_t x_6; 
x_6 = lean_usize_dec_eq(x_2, x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_14; 
x_7 = lean_array_uget(x_1, x_2);
x_8 = lean_st_ref_get(x_7);
x_14 = lp_aesop_Aesop_Rapp_isSafe(x_8);
if (x_14 == 0)
{
lean_dec(x_7);
x_9 = x_4;
goto block_13;
}
else
{
lean_object* x_15; 
x_15 = lean_array_push(x_4, x_7);
x_9 = x_15;
goto block_13;
}
block_13:
{
size_t x_10; size_t x_11; 
x_10 = 1;
x_11 = lean_usize_add(x_2, x_10);
x_2 = x_11;
x_4 = x_9;
goto _start;
}
}
else
{
return x_4;
}
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_safeRapps___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_safeRapps(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_4, x_1);
x_6 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_unsigned_to_nat(0u);
x_8 = lean_array_get_size(x_6);
x_9 = lp_aesop_Aesop_Goal_safeRapps___closed__0;
x_10 = lean_nat_dec_lt(x_7, x_8);
if (x_10 == 0)
{
lean_dec_ref(x_6);
return x_9;
}
else
{
uint8_t x_11; 
x_11 = lean_nat_dec_le(x_8, x_8);
if (x_11 == 0)
{
lean_dec_ref(x_6);
return x_9;
}
else
{
size_t x_12; size_t x_13; lean_object* x_14; 
x_12 = 0;
x_13 = lean_usize_of_nat(x_8);
x_14 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0(x_6, x_12, x_13, x_9);
lean_dec_ref(x_6);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_7 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_8 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0(x_1, x_6, x_7, x_4);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_safeRapps___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_Goal_safeRapps(x_1);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0(lean_object* x_1, size_t x_2, size_t x_3) {
_start:
{
uint8_t x_5; 
x_5 = lean_usize_dec_eq(x_2, x_3);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_6 = lean_array_uget(x_1, x_2);
x_7 = lean_st_ref_get(x_6);
lean_dec(x_6);
x_8 = lp_aesop_Aesop_Rapp_isSafe(x_7);
if (x_8 == 0)
{
size_t x_9; size_t x_10; 
x_9 = 1;
x_10 = lean_usize_add(x_2, x_9);
x_2 = x_10;
goto _start;
}
else
{
return x_8;
}
}
else
{
uint8_t x_12; 
x_12 = 0;
return x_12;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasSafeRapp(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_4, x_1);
x_6 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_unsigned_to_nat(0u);
x_8 = lean_array_get_size(x_6);
x_9 = lean_nat_dec_lt(x_7, x_8);
if (x_9 == 0)
{
lean_dec_ref(x_6);
return x_9;
}
else
{
if (x_9 == 0)
{
lean_dec_ref(x_6);
return x_9;
}
else
{
size_t x_10; size_t x_11; uint8_t x_12; 
x_10 = 0;
x_11 = lean_usize_of_nat(x_8);
x_12 = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0(x_6, x_10, x_11);
lean_dec_ref(x_6);
return x_12;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; size_t x_6; uint8_t x_7; lean_object* x_8; 
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0(x_1, x_5, x_6);
lean_dec_ref(x_1);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasSafeRapp___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Goal_hasSafeRapp(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isUnsafeExhausted(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*14 + 11);
if (x_5 == 0)
{
lean_dec_ref(x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lean_ctor_get(x_4, 12);
lean_inc_ref(x_6);
lean_dec_ref(x_4);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 2);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_nat_dec_eq(x_7, x_8);
lean_dec(x_8);
lean_dec(x_7);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isUnsafeExhausted___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_isUnsafeExhausted(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isExhausted(lean_object* x_1) {
_start:
{
uint8_t x_3; 
lean_inc(x_1);
x_3 = lp_aesop_Aesop_Goal_isUnsafeExhausted(x_1);
if (x_3 == 0)
{
uint8_t x_4; 
x_4 = lp_aesop_Aesop_Goal_hasSafeRapp(x_1);
return x_4;
}
else
{
lean_dec(x_1);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isExhausted___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Goal_isExhausted(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isActive(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lp_aesop_Aesop_treeImpl;
x_7 = lean_ctor_get(x_6, 1);
lean_inc_ref(x_7);
lean_inc(x_1);
x_8 = lean_apply_1(x_7, x_1);
x_9 = lean_ctor_get_uint8(x_8, sizeof(void*)*14 + 9);
lean_dec_ref(x_8);
if (x_9 == 0)
{
uint8_t x_10; 
x_10 = lp_aesop_Aesop_Goal_isExhausted(x_1);
if (x_10 == 0)
{
uint8_t x_11; 
x_11 = 1;
return x_11;
}
else
{
x_3 = lean_box(0);
goto block_5;
}
}
else
{
lean_dec(x_1);
x_3 = lean_box(0);
goto block_5;
}
block_5:
{
uint8_t x_4; 
x_4 = 0;
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isActive___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Goal_isActive(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0(lean_object* x_1, size_t x_2, size_t x_3) {
_start:
{
uint8_t x_5; 
x_5 = lean_usize_dec_eq(x_2, x_3);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; uint8_t x_12; uint8_t x_13; 
x_6 = lean_array_uget(x_1, x_2);
x_7 = lean_st_ref_get(x_6);
lean_dec(x_6);
x_8 = lp_aesop_Aesop_treeImpl;
x_9 = lean_ctor_get(x_8, 3);
lean_inc_ref(x_9);
x_10 = lean_apply_1(x_9, x_7);
x_11 = lean_ctor_get_uint8(x_10, sizeof(void*)*9 + 8);
lean_dec_ref(x_10);
x_12 = 1;
x_13 = lp_aesop_Aesop_NodeState_isUnprovable(x_11);
if (x_13 == 0)
{
return x_12;
}
else
{
if (x_5 == 0)
{
size_t x_14; size_t x_15; 
x_14 = 1;
x_15 = lean_usize_add(x_2, x_14);
x_2 = x_15;
goto _start;
}
else
{
return x_12;
}
}
}
else
{
uint8_t x_17; 
x_17 = 0;
return x_17;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasProvableRapp(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_4, x_1);
x_6 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_unsigned_to_nat(0u);
x_8 = lean_array_get_size(x_6);
x_9 = lean_nat_dec_lt(x_7, x_8);
if (x_9 == 0)
{
lean_dec_ref(x_6);
return x_9;
}
else
{
if (x_9 == 0)
{
lean_dec_ref(x_6);
return x_9;
}
else
{
size_t x_10; size_t x_11; uint8_t x_12; 
x_10 = 0;
x_11 = lean_usize_of_nat(x_8);
x_12 = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0(x_6, x_10, x_11);
lean_dec_ref(x_6);
return x_12;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasProvableRapp___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Goal_hasProvableRapp(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; size_t x_6; uint8_t x_7; lean_object* x_8; 
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0(x_1, x_5, x_6);
lean_dec_ref(x_1);
x_8 = lean_box(x_7);
return x_8;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_firstProvenRapp_x3f___closed__0() {
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
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, size_t x_4, size_t x_5, lean_object* x_6) {
_start:
{
uint8_t x_8; 
x_8 = lean_usize_dec_lt(x_5, x_4);
if (x_8 == 0)
{
lean_inc_ref(x_6);
return x_6;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; 
x_9 = lean_array_uget(x_3, x_5);
x_10 = lean_st_ref_get(x_9);
x_11 = lp_aesop_Aesop_treeImpl;
x_12 = lean_ctor_get(x_11, 3);
lean_inc_ref(x_12);
x_13 = lean_apply_1(x_12, x_10);
x_14 = lean_ctor_get_uint8(x_13, sizeof(void*)*9 + 8);
lean_dec_ref(x_13);
x_15 = lp_aesop_Aesop_NodeState_isProven(x_14);
if (x_15 == 0)
{
size_t x_16; size_t x_17; 
lean_dec(x_9);
x_16 = 1;
x_17 = lean_usize_add(x_5, x_16);
{
size_t _tmp_4 = x_17;
lean_object* _tmp_5 = x_1;
x_5 = _tmp_4;
x_6 = _tmp_5;
}
goto _start;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_9);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_2);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; size_t x_10; size_t x_11; lean_object* x_12; lean_object* x_13; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_4, x_1);
x_6 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_box(0);
x_8 = lean_box(0);
x_9 = lp_aesop_Aesop_Goal_firstProvenRapp_x3f___closed__0;
x_10 = lean_array_size(x_6);
x_11 = 0;
x_12 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0(x_9, x_8, x_6, x_10, x_11, x_9);
lean_dec_ref(x_6);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
if (lean_obj_tag(x_13) == 0)
{
return x_7;
}
else
{
lean_object* x_14; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_Goal_firstProvenRapp_x3f(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
size_t x_8; size_t x_9; lean_object* x_10; 
x_8 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_9 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_10 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0(x_1, x_2, x_3, x_8, x_9, x_6);
lean_dec_ref(x_6);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasMVar(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 7);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = l_Array_isEmpty___redArg(x_5);
lean_dec_ref(x_5);
if (x_6 == 0)
{
uint8_t x_7; 
x_7 = 1;
return x_7;
}
else
{
uint8_t x_8; 
x_8 = 0;
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasMVar___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_hasMVar(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_array_get_size(x_1);
return x_2;
}
}
static double _init_lp_aesop_Aesop_Goal_priority___closed__0() {
_start:
{
double x_1; 
x_1 = lp_aesop_Aesop_unificationGoalPenalty;
return x_1;
}
}
LEAN_EXPORT double lp_aesop_Aesop_Goal_priority(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; double x_6; double x_7; lean_object* x_8; double x_9; double x_10; double x_11; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 7);
lean_inc_ref(x_5);
x_6 = lean_ctor_get_float(x_4, sizeof(void*)*14);
lean_dec_ref(x_4);
x_7 = lp_aesop_Aesop_Goal_priority___closed__0;
x_8 = lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0(x_5);
lean_dec_ref(x_5);
x_9 = lean_float_of_nat(x_8);
x_10 = pow(x_7, x_9);
x_11 = lean_float_mul(x_6, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_priority___boxed(lean_object* x_1) {
_start:
{
double x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_priority(x_1);
x_3 = lean_box_float(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isNormal(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 6);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lp_aesop_Aesop_NormalizationState_isNormal(x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isNormal___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Goal_isNormal(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_originalGoalId(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 3);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f(x_6);
lean_dec(x_6);
if (lean_obj_tag(x_7) == 0)
{
return x_5;
}
else
{
lean_object* x_8; 
lean_dec(x_5);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
return x_8;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isRoot(lean_object* x_1) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_Goal_parentRapp_x3f(x_1);
if (lean_obj_tag(x_3) == 0)
{
uint8_t x_4; 
x_4 = 1;
return x_4;
}
else
{
uint8_t x_5; 
lean_dec_ref(x_3);
x_5 = 0;
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isRoot___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_Goal_isRoot(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_introducesMVar(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lp_aesop_Aesop_treeImpl;
x_3 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_3);
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_ctor_get(x_4, 7);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = l_Array_isEmpty___redArg(x_5);
lean_dec_ref(x_5);
if (x_6 == 0)
{
uint8_t x_7; 
x_7 = 1;
return x_7;
}
else
{
uint8_t x_8; 
x_8 = 0;
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_introducesMVar___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_Rapp_introducesMVar(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parentPostNormMetaState(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lp_aesop_Aesop_treeImpl;
x_5 = lean_ctor_get(x_4, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_1);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_st_ref_get(x_7);
lean_dec(x_7);
x_9 = lp_aesop_Aesop_Goal_parentMetaState(x_8, x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parentPostNormMetaState___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_Rapp_parentPostNormMetaState(x_1, x_2);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_6 = lp_aesop_Aesop_treeImpl;
x_7 = lean_ctor_get(x_6, 5);
lean_inc_ref(x_7);
x_8 = lean_apply_1(x_7, x_5);
x_9 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_unsigned_to_nat(0u);
x_11 = lean_array_get_size(x_9);
x_12 = lean_nat_dec_lt(x_10, x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; 
lean_dec_ref(x_9);
lean_dec(x_4);
lean_dec_ref(x_3);
x_13 = lean_ctor_get(x_1, 1);
lean_inc(x_13);
lean_dec_ref(x_1);
x_14 = lean_apply_2(x_13, lean_box(0), x_2);
return x_14;
}
else
{
uint8_t x_15; 
x_15 = lean_nat_dec_le(x_11, x_11);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; 
lean_dec_ref(x_9);
lean_dec(x_4);
lean_dec_ref(x_3);
x_16 = lean_ctor_get(x_1, 1);
lean_inc(x_16);
lean_dec_ref(x_1);
x_17 = lean_apply_2(x_16, lean_box(0), x_2);
return x_17;
}
else
{
size_t x_18; size_t x_19; lean_object* x_20; 
lean_dec_ref(x_1);
x_18 = 0;
x_19 = lean_usize_of_nat(x_11);
x_20 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_3, x_4, x_9, x_18, x_19, x_2);
return x_20;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__0), 5, 4);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_6);
lean_closure_set(x_8, 2, x_2);
lean_closure_set(x_8, 3, x_3);
x_9 = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, x_7);
x_10 = lean_apply_2(x_4, lean_box(0), x_9);
x_11 = lean_apply_4(x_5, lean_box(0), lean_box(0), x_10, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 1);
x_8 = lp_aesop_Aesop_treeImpl;
x_9 = lean_ctor_get(x_8, 3);
lean_inc_ref(x_9);
x_10 = lean_apply_1(x_9, x_5);
x_11 = lean_ctor_get(x_10, 2);
lean_inc_ref(x_11);
lean_dec_ref(x_10);
x_12 = lean_unsigned_to_nat(0u);
x_13 = lean_array_get_size(x_11);
x_14 = lean_nat_dec_lt(x_12, x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; 
lean_dec_ref(x_11);
lean_inc_ref(x_6);
lean_dec(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_15 = lean_ctor_get(x_6, 1);
lean_inc(x_15);
lean_dec_ref(x_6);
x_16 = lean_apply_2(x_15, lean_box(0), x_3);
return x_16;
}
else
{
uint8_t x_17; 
x_17 = lean_nat_dec_le(x_13, x_13);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; 
lean_dec_ref(x_11);
lean_inc_ref(x_6);
lean_dec(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_18 = lean_ctor_get(x_6, 1);
lean_inc(x_18);
lean_dec_ref(x_6);
x_19 = lean_apply_2(x_18, lean_box(0), x_3);
return x_19;
}
else
{
lean_object* x_20; size_t x_21; size_t x_22; lean_object* x_23; 
lean_inc(x_7);
lean_inc_ref(x_1);
lean_inc_ref(x_6);
x_20 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__1), 7, 5);
lean_closure_set(x_20, 0, x_6);
lean_closure_set(x_20, 1, x_1);
lean_closure_set(x_20, 2, x_4);
lean_closure_set(x_20, 3, x_2);
lean_closure_set(x_20, 4, x_7);
x_21 = 0;
x_22 = lean_usize_of_nat(x_13);
x_23 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_1, x_20, x_11, x_21, x_22, x_3);
return x_23;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg(x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_5 = lp_aesop_Aesop_treeImpl;
x_6 = lean_ctor_get(x_5, 5);
lean_inc_ref(x_6);
x_7 = lean_apply_1(x_6, x_4);
x_8 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lean_unsigned_to_nat(0u);
x_10 = lean_array_get_size(x_8);
x_11 = lean_box(0);
x_12 = lean_nat_dec_lt(x_9, x_10);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; 
lean_dec_ref(x_8);
lean_dec(x_3);
lean_dec_ref(x_2);
x_13 = lean_ctor_get(x_1, 1);
lean_inc(x_13);
lean_dec_ref(x_1);
x_14 = lean_apply_2(x_13, lean_box(0), x_11);
return x_14;
}
else
{
uint8_t x_15; 
x_15 = lean_nat_dec_le(x_10, x_10);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; 
lean_dec_ref(x_8);
lean_dec(x_3);
lean_dec_ref(x_2);
x_16 = lean_ctor_get(x_1, 1);
lean_inc(x_16);
lean_dec_ref(x_1);
x_17 = lean_apply_2(x_16, lean_box(0), x_11);
return x_17;
}
else
{
size_t x_18; size_t x_19; lean_object* x_20; 
lean_dec_ref(x_1);
x_18 = 0;
x_19 = lean_usize_of_nat(x_10);
x_20 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_2, x_3, x_8, x_18, x_19, x_11);
return x_20;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_5);
x_7 = lean_apply_2(x_1, lean_box(0), x_6);
x_8 = lean_apply_4(x_2, lean_box(0), lean_box(0), x_7, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lp_aesop_Aesop_treeImpl;
x_8 = lean_ctor_get(x_7, 3);
lean_inc_ref(x_8);
x_9 = lean_apply_1(x_8, x_4);
x_10 = lean_ctor_get(x_9, 2);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lean_unsigned_to_nat(0u);
x_12 = lean_array_get_size(x_10);
x_13 = lean_box(0);
x_14 = lean_nat_dec_lt(x_11, x_12);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; 
lean_dec_ref(x_10);
lean_inc_ref(x_5);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_15 = lean_ctor_get(x_5, 1);
lean_inc(x_15);
lean_dec_ref(x_5);
x_16 = lean_apply_2(x_15, lean_box(0), x_13);
return x_16;
}
else
{
uint8_t x_17; 
x_17 = lean_nat_dec_le(x_12, x_12);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; 
lean_dec_ref(x_10);
lean_inc_ref(x_5);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_18 = lean_ctor_get(x_5, 1);
lean_inc(x_18);
lean_dec_ref(x_5);
x_19 = lean_apply_2(x_18, lean_box(0), x_13);
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; size_t x_23; size_t x_24; lean_object* x_25; 
x_20 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__0), 3, 1);
lean_closure_set(x_20, 0, x_3);
lean_inc_ref(x_1);
lean_inc_ref(x_5);
x_21 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__1), 4, 3);
lean_closure_set(x_21, 0, x_5);
lean_closure_set(x_21, 1, x_1);
lean_closure_set(x_21, 2, x_20);
lean_inc(x_6);
x_22 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__2), 5, 3);
lean_closure_set(x_22, 0, x_2);
lean_closure_set(x_22, 1, x_6);
lean_closure_set(x_22, 2, x_21);
x_23 = 0;
x_24 = lean_usize_of_nat(x_12);
x_25 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), x_1, x_22, x_10, x_23, x_24, x_13);
return x_25;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_array_push(x_2, x_3);
x_5 = lean_apply_2(x_1, lean_box(0), x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_4, 1);
x_6 = lp_aesop_Aesop_Goal_safeRapps___closed__0;
lean_inc(x_5);
x_7 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_subgoals___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
x_8 = lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg(x_1, x_2, x_6, x_7, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_Rapp_subgoals___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_depth(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_5);
x_6 = lean_apply_1(x_5, x_1);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_st_ref_get(x_7);
lean_dec(x_7);
x_9 = lean_apply_1(x_4, x_8);
x_10 = lean_ctor_get(x_9, 4);
lean_inc(x_10);
lean_dec_ref(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_depth___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_Rapp_depth(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, size_t x_4, size_t x_5, lean_object* x_6) {
_start:
{
uint8_t x_8; 
x_8 = lean_usize_dec_lt(x_5, x_4);
if (x_8 == 0)
{
lean_inc_ref(x_6);
return x_6;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; 
x_9 = lean_array_uget(x_3, x_5);
x_10 = lean_st_ref_get(x_9);
x_11 = lp_aesop_Aesop_treeImpl;
x_12 = lean_ctor_get(x_11, 1);
lean_inc_ref(x_12);
x_13 = lean_apply_1(x_12, x_10);
x_14 = lean_ctor_get_uint8(x_13, sizeof(void*)*14 + 8);
lean_dec_ref(x_13);
x_15 = lp_aesop_Aesop_GoalState_isProven(x_14);
if (x_15 == 0)
{
size_t x_16; size_t x_17; 
lean_dec(x_9);
x_16 = 1;
x_17 = lean_usize_add(x_5, x_16);
{
size_t _tmp_4 = x_17;
lean_object* _tmp_5 = x_1;
x_5 = _tmp_4;
x_6 = _tmp_5;
}
goto _start;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_9);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_2);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_provenGoal_x3f(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; size_t x_10; size_t x_11; lean_object* x_12; lean_object* x_13; 
x_3 = lp_aesop_Aesop_treeImpl;
x_4 = lean_ctor_get(x_3, 5);
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_4, x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_box(0);
x_8 = lean_box(0);
x_9 = lp_aesop_Aesop_Goal_firstProvenRapp_x3f___closed__0;
x_10 = lean_array_size(x_6);
x_11 = 0;
x_12 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0(x_9, x_8, x_6, x_10, x_11, x_9);
lean_dec_ref(x_6);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
if (lean_obj_tag(x_13) == 0)
{
return x_7;
}
else
{
lean_object* x_14; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_provenGoal_x3f___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_MVarCluster_provenGoal_x3f(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
size_t x_8; size_t x_9; lean_object* x_10; 
x_8 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_9 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_10 = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0(x_1, x_2, x_3, x_8, x_9, x_6);
lean_dec_ref(x_6);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_3 = lean_st_ref_take(x_1);
x_4 = lp_aesop_Aesop_treeImpl;
x_5 = lean_ctor_get(x_4, 2);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 3);
lean_inc_ref(x_6);
x_7 = lean_apply_1(x_6, x_3);
x_8 = lean_ctor_get(x_7, 6);
lean_inc_ref(x_8);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
x_11 = !lean_is_exclusive(x_7);
if (x_11 == 0)
{
lean_object* x_12; uint8_t x_13; 
x_12 = lean_ctor_get(x_7, 6);
lean_dec(x_12);
x_13 = !lean_is_exclusive(x_8);
if (x_13 == 0)
{
lean_object* x_14; uint8_t x_15; 
x_14 = lean_ctor_get(x_8, 0);
lean_dec(x_14);
x_15 = !lean_is_exclusive(x_9);
if (x_15 == 0)
{
lean_object* x_16; uint8_t x_17; 
x_16 = lean_ctor_get(x_9, 0);
lean_dec(x_16);
x_17 = !lean_is_exclusive(x_10);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_18 = lean_ctor_get(x_10, 3);
x_19 = l_Lean_DeclNameGenerator_mkChild(x_18);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
x_21 = lean_ctor_get(x_19, 1);
lean_inc(x_21);
lean_dec_ref(x_19);
lean_ctor_set(x_10, 3, x_21);
x_22 = lean_apply_1(x_5, x_7);
x_23 = lean_st_ref_set(x_1, x_22);
return x_20;
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_24 = lean_ctor_get(x_10, 0);
x_25 = lean_ctor_get(x_10, 1);
x_26 = lean_ctor_get(x_10, 2);
x_27 = lean_ctor_get(x_10, 3);
x_28 = lean_ctor_get(x_10, 4);
x_29 = lean_ctor_get(x_10, 5);
x_30 = lean_ctor_get(x_10, 6);
x_31 = lean_ctor_get(x_10, 7);
x_32 = lean_ctor_get(x_10, 8);
lean_inc(x_32);
lean_inc(x_31);
lean_inc(x_30);
lean_inc(x_29);
lean_inc(x_28);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_24);
lean_dec(x_10);
x_33 = l_Lean_DeclNameGenerator_mkChild(x_27);
x_34 = lean_ctor_get(x_33, 0);
lean_inc(x_34);
x_35 = lean_ctor_get(x_33, 1);
lean_inc(x_35);
lean_dec_ref(x_33);
x_36 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_36, 0, x_24);
lean_ctor_set(x_36, 1, x_25);
lean_ctor_set(x_36, 2, x_26);
lean_ctor_set(x_36, 3, x_35);
lean_ctor_set(x_36, 4, x_28);
lean_ctor_set(x_36, 5, x_29);
lean_ctor_set(x_36, 6, x_30);
lean_ctor_set(x_36, 7, x_31);
lean_ctor_set(x_36, 8, x_32);
lean_ctor_set(x_9, 0, x_36);
x_37 = lean_apply_1(x_5, x_7);
x_38 = lean_st_ref_set(x_1, x_37);
return x_34;
}
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_39 = lean_ctor_get(x_9, 1);
lean_inc(x_39);
lean_dec(x_9);
x_40 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_40);
x_41 = lean_ctor_get(x_10, 1);
lean_inc(x_41);
x_42 = lean_ctor_get(x_10, 2);
lean_inc_ref(x_42);
x_43 = lean_ctor_get(x_10, 3);
lean_inc_ref(x_43);
x_44 = lean_ctor_get(x_10, 4);
lean_inc_ref(x_44);
x_45 = lean_ctor_get(x_10, 5);
lean_inc_ref(x_45);
x_46 = lean_ctor_get(x_10, 6);
lean_inc_ref(x_46);
x_47 = lean_ctor_get(x_10, 7);
lean_inc_ref(x_47);
x_48 = lean_ctor_get(x_10, 8);
lean_inc_ref(x_48);
if (lean_is_exclusive(x_10)) {
 lean_ctor_release(x_10, 0);
 lean_ctor_release(x_10, 1);
 lean_ctor_release(x_10, 2);
 lean_ctor_release(x_10, 3);
 lean_ctor_release(x_10, 4);
 lean_ctor_release(x_10, 5);
 lean_ctor_release(x_10, 6);
 lean_ctor_release(x_10, 7);
 lean_ctor_release(x_10, 8);
 x_49 = x_10;
} else {
 lean_dec_ref(x_10);
 x_49 = lean_box(0);
}
x_50 = l_Lean_DeclNameGenerator_mkChild(x_43);
x_51 = lean_ctor_get(x_50, 0);
lean_inc(x_51);
x_52 = lean_ctor_get(x_50, 1);
lean_inc(x_52);
lean_dec_ref(x_50);
if (lean_is_scalar(x_49)) {
 x_53 = lean_alloc_ctor(0, 9, 0);
} else {
 x_53 = x_49;
}
lean_ctor_set(x_53, 0, x_40);
lean_ctor_set(x_53, 1, x_41);
lean_ctor_set(x_53, 2, x_42);
lean_ctor_set(x_53, 3, x_52);
lean_ctor_set(x_53, 4, x_44);
lean_ctor_set(x_53, 5, x_45);
lean_ctor_set(x_53, 6, x_46);
lean_ctor_set(x_53, 7, x_47);
lean_ctor_set(x_53, 8, x_48);
x_54 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_54, 0, x_53);
lean_ctor_set(x_54, 1, x_39);
lean_ctor_set(x_8, 0, x_54);
x_55 = lean_apply_1(x_5, x_7);
x_56 = lean_st_ref_set(x_1, x_55);
return x_51;
}
}
else
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; 
x_57 = lean_ctor_get(x_8, 1);
lean_inc(x_57);
lean_dec(x_8);
x_58 = lean_ctor_get(x_9, 1);
lean_inc(x_58);
if (lean_is_exclusive(x_9)) {
 lean_ctor_release(x_9, 0);
 lean_ctor_release(x_9, 1);
 x_59 = x_9;
} else {
 lean_dec_ref(x_9);
 x_59 = lean_box(0);
}
x_60 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_60);
x_61 = lean_ctor_get(x_10, 1);
lean_inc(x_61);
x_62 = lean_ctor_get(x_10, 2);
lean_inc_ref(x_62);
x_63 = lean_ctor_get(x_10, 3);
lean_inc_ref(x_63);
x_64 = lean_ctor_get(x_10, 4);
lean_inc_ref(x_64);
x_65 = lean_ctor_get(x_10, 5);
lean_inc_ref(x_65);
x_66 = lean_ctor_get(x_10, 6);
lean_inc_ref(x_66);
x_67 = lean_ctor_get(x_10, 7);
lean_inc_ref(x_67);
x_68 = lean_ctor_get(x_10, 8);
lean_inc_ref(x_68);
if (lean_is_exclusive(x_10)) {
 lean_ctor_release(x_10, 0);
 lean_ctor_release(x_10, 1);
 lean_ctor_release(x_10, 2);
 lean_ctor_release(x_10, 3);
 lean_ctor_release(x_10, 4);
 lean_ctor_release(x_10, 5);
 lean_ctor_release(x_10, 6);
 lean_ctor_release(x_10, 7);
 lean_ctor_release(x_10, 8);
 x_69 = x_10;
} else {
 lean_dec_ref(x_10);
 x_69 = lean_box(0);
}
x_70 = l_Lean_DeclNameGenerator_mkChild(x_63);
x_71 = lean_ctor_get(x_70, 0);
lean_inc(x_71);
x_72 = lean_ctor_get(x_70, 1);
lean_inc(x_72);
lean_dec_ref(x_70);
if (lean_is_scalar(x_69)) {
 x_73 = lean_alloc_ctor(0, 9, 0);
} else {
 x_73 = x_69;
}
lean_ctor_set(x_73, 0, x_60);
lean_ctor_set(x_73, 1, x_61);
lean_ctor_set(x_73, 2, x_62);
lean_ctor_set(x_73, 3, x_72);
lean_ctor_set(x_73, 4, x_64);
lean_ctor_set(x_73, 5, x_65);
lean_ctor_set(x_73, 6, x_66);
lean_ctor_set(x_73, 7, x_67);
lean_ctor_set(x_73, 8, x_68);
if (lean_is_scalar(x_59)) {
 x_74 = lean_alloc_ctor(0, 2, 0);
} else {
 x_74 = x_59;
}
lean_ctor_set(x_74, 0, x_73);
lean_ctor_set(x_74, 1, x_58);
x_75 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_75, 0, x_74);
lean_ctor_set(x_75, 1, x_57);
lean_ctor_set(x_7, 6, x_75);
x_76 = lean_apply_1(x_5, x_7);
x_77 = lean_st_ref_set(x_1, x_76);
return x_71;
}
}
else
{
lean_object* x_78; lean_object* x_79; lean_object* x_80; uint8_t x_81; uint8_t x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; double x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; 
x_78 = lean_ctor_get(x_7, 0);
x_79 = lean_ctor_get(x_7, 1);
x_80 = lean_ctor_get(x_7, 2);
x_81 = lean_ctor_get_uint8(x_7, sizeof(void*)*9 + 8);
x_82 = lean_ctor_get_uint8(x_7, sizeof(void*)*9 + 9);
x_83 = lean_ctor_get(x_7, 3);
x_84 = lean_ctor_get(x_7, 4);
x_85 = lean_ctor_get(x_7, 5);
x_86 = lean_ctor_get_float(x_7, sizeof(void*)*9);
x_87 = lean_ctor_get(x_7, 7);
x_88 = lean_ctor_get(x_7, 8);
lean_inc(x_88);
lean_inc(x_87);
lean_inc(x_85);
lean_inc(x_84);
lean_inc(x_83);
lean_inc(x_80);
lean_inc(x_79);
lean_inc(x_78);
lean_dec(x_7);
x_89 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_89);
if (lean_is_exclusive(x_8)) {
 lean_ctor_release(x_8, 0);
 lean_ctor_release(x_8, 1);
 x_90 = x_8;
} else {
 lean_dec_ref(x_8);
 x_90 = lean_box(0);
}
x_91 = lean_ctor_get(x_9, 1);
lean_inc(x_91);
if (lean_is_exclusive(x_9)) {
 lean_ctor_release(x_9, 0);
 lean_ctor_release(x_9, 1);
 x_92 = x_9;
} else {
 lean_dec_ref(x_9);
 x_92 = lean_box(0);
}
x_93 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_93);
x_94 = lean_ctor_get(x_10, 1);
lean_inc(x_94);
x_95 = lean_ctor_get(x_10, 2);
lean_inc_ref(x_95);
x_96 = lean_ctor_get(x_10, 3);
lean_inc_ref(x_96);
x_97 = lean_ctor_get(x_10, 4);
lean_inc_ref(x_97);
x_98 = lean_ctor_get(x_10, 5);
lean_inc_ref(x_98);
x_99 = lean_ctor_get(x_10, 6);
lean_inc_ref(x_99);
x_100 = lean_ctor_get(x_10, 7);
lean_inc_ref(x_100);
x_101 = lean_ctor_get(x_10, 8);
lean_inc_ref(x_101);
if (lean_is_exclusive(x_10)) {
 lean_ctor_release(x_10, 0);
 lean_ctor_release(x_10, 1);
 lean_ctor_release(x_10, 2);
 lean_ctor_release(x_10, 3);
 lean_ctor_release(x_10, 4);
 lean_ctor_release(x_10, 5);
 lean_ctor_release(x_10, 6);
 lean_ctor_release(x_10, 7);
 lean_ctor_release(x_10, 8);
 x_102 = x_10;
} else {
 lean_dec_ref(x_10);
 x_102 = lean_box(0);
}
x_103 = l_Lean_DeclNameGenerator_mkChild(x_96);
x_104 = lean_ctor_get(x_103, 0);
lean_inc(x_104);
x_105 = lean_ctor_get(x_103, 1);
lean_inc(x_105);
lean_dec_ref(x_103);
if (lean_is_scalar(x_102)) {
 x_106 = lean_alloc_ctor(0, 9, 0);
} else {
 x_106 = x_102;
}
lean_ctor_set(x_106, 0, x_93);
lean_ctor_set(x_106, 1, x_94);
lean_ctor_set(x_106, 2, x_95);
lean_ctor_set(x_106, 3, x_105);
lean_ctor_set(x_106, 4, x_97);
lean_ctor_set(x_106, 5, x_98);
lean_ctor_set(x_106, 6, x_99);
lean_ctor_set(x_106, 7, x_100);
lean_ctor_set(x_106, 8, x_101);
if (lean_is_scalar(x_92)) {
 x_107 = lean_alloc_ctor(0, 2, 0);
} else {
 x_107 = x_92;
}
lean_ctor_set(x_107, 0, x_106);
lean_ctor_set(x_107, 1, x_91);
if (lean_is_scalar(x_90)) {
 x_108 = lean_alloc_ctor(0, 2, 0);
} else {
 x_108 = x_90;
}
lean_ctor_set(x_108, 0, x_107);
lean_ctor_set(x_108, 1, x_89);
x_109 = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(x_109, 0, x_78);
lean_ctor_set(x_109, 1, x_79);
lean_ctor_set(x_109, 2, x_80);
lean_ctor_set(x_109, 3, x_83);
lean_ctor_set(x_109, 4, x_84);
lean_ctor_set(x_109, 5, x_85);
lean_ctor_set(x_109, 6, x_108);
lean_ctor_set(x_109, 7, x_87);
lean_ctor_set(x_109, 8, x_88);
lean_ctor_set_uint8(x_109, sizeof(void*)*9 + 8, x_81);
lean_ctor_set_uint8(x_109, sizeof(void*)*9 + 9, x_82);
lean_ctor_set_float(x_109, sizeof(void*)*9, x_86);
x_110 = lean_apply_1(x_5, x_109);
x_111 = lean_st_ref_set(x_1, x_110);
return x_104;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator(x_1);
lean_dec(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Constants(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_Step(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tracing(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Data_ForwardRuleMatches(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_UnsafeQueue(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_State(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_Data(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Constants(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Step(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Data_ForwardRuleMatches(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_UnsafeQueue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0 = _init_lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0);
lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1 = _init_lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1);
lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2 = _init_lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2);
lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3 = _init_lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3();
lean_mark_persistent(lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3);
lp_aesop_Aesop_instInhabitedGoalId_default = _init_lp_aesop_Aesop_instInhabitedGoalId_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalId_default);
lp_aesop_Aesop_instInhabitedGoalId = _init_lp_aesop_Aesop_instInhabitedGoalId();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalId);
lp_aesop_Aesop_GoalId_zero = _init_lp_aesop_Aesop_GoalId_zero();
lean_mark_persistent(lp_aesop_Aesop_GoalId_zero);
lp_aesop_Aesop_GoalId_one = _init_lp_aesop_Aesop_GoalId_one();
lean_mark_persistent(lp_aesop_Aesop_GoalId_one);
lp_aesop_Aesop_GoalId_dummy = _init_lp_aesop_Aesop_GoalId_dummy();
lean_mark_persistent(lp_aesop_Aesop_GoalId_dummy);
lp_aesop_Aesop_GoalId_instLT = _init_lp_aesop_Aesop_GoalId_instLT();
lean_mark_persistent(lp_aesop_Aesop_GoalId_instLT);
lp_aesop_Aesop_GoalId_instToString___closed__0 = _init_lp_aesop_Aesop_GoalId_instToString___closed__0();
lean_mark_persistent(lp_aesop_Aesop_GoalId_instToString___closed__0);
lp_aesop_Aesop_GoalId_instToString = _init_lp_aesop_Aesop_GoalId_instToString();
lean_mark_persistent(lp_aesop_Aesop_GoalId_instToString);
lp_aesop_Aesop_GoalId_instHashable___closed__0 = _init_lp_aesop_Aesop_GoalId_instHashable___closed__0();
lean_mark_persistent(lp_aesop_Aesop_GoalId_instHashable___closed__0);
lp_aesop_Aesop_GoalId_instHashable = _init_lp_aesop_Aesop_GoalId_instHashable();
lean_mark_persistent(lp_aesop_Aesop_GoalId_instHashable);
lp_aesop_Aesop_instInhabitedRappId_default = _init_lp_aesop_Aesop_instInhabitedRappId_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRappId_default);
lp_aesop_Aesop_instInhabitedRappId = _init_lp_aesop_Aesop_instInhabitedRappId();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRappId);
lp_aesop_Aesop_RappId_zero = _init_lp_aesop_Aesop_RappId_zero();
lean_mark_persistent(lp_aesop_Aesop_RappId_zero);
lp_aesop_Aesop_RappId_one = _init_lp_aesop_Aesop_RappId_one();
lean_mark_persistent(lp_aesop_Aesop_RappId_one);
lp_aesop_Aesop_RappId_dummy = _init_lp_aesop_Aesop_RappId_dummy();
lean_mark_persistent(lp_aesop_Aesop_RappId_dummy);
lp_aesop_Aesop_RappId_instLT = _init_lp_aesop_Aesop_RappId_instLT();
lean_mark_persistent(lp_aesop_Aesop_RappId_instLT);
lp_aesop_Aesop_RappId_instToString = _init_lp_aesop_Aesop_RappId_instToString();
lean_mark_persistent(lp_aesop_Aesop_RappId_instToString);
lp_aesop_Aesop_RappId_instHashable = _init_lp_aesop_Aesop_RappId_instHashable();
lean_mark_persistent(lp_aesop_Aesop_RappId_instHashable);
lp_aesop_Aesop_instInhabitedIteration = _init_lp_aesop_Aesop_instInhabitedIteration();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedIteration);
lp_aesop_Aesop_Iteration_one = _init_lp_aesop_Aesop_Iteration_one();
lean_mark_persistent(lp_aesop_Aesop_Iteration_one);
lp_aesop_Aesop_Iteration_none = _init_lp_aesop_Aesop_Iteration_none();
lean_mark_persistent(lp_aesop_Aesop_Iteration_none);
lp_aesop_Aesop_Iteration_instToString = _init_lp_aesop_Aesop_Iteration_instToString();
lean_mark_persistent(lp_aesop_Aesop_Iteration_instToString);
lp_aesop_Aesop_Iteration_instLT = _init_lp_aesop_Aesop_Iteration_instLT();
lean_mark_persistent(lp_aesop_Aesop_Iteration_instLT);
lp_aesop_Aesop_Iteration_instLE = _init_lp_aesop_Aesop_Iteration_instLE();
lean_mark_persistent(lp_aesop_Aesop_Iteration_instLE);
lp_aesop_Aesop_instInhabitedNodeState_default = _init_lp_aesop_Aesop_instInhabitedNodeState_default();
lp_aesop_Aesop_instInhabitedNodeState = _init_lp_aesop_Aesop_instInhabitedNodeState();
lp_aesop_Aesop_instBEqNodeState___closed__0 = _init_lp_aesop_Aesop_instBEqNodeState___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instBEqNodeState___closed__0);
lp_aesop_Aesop_instBEqNodeState = _init_lp_aesop_Aesop_instBEqNodeState();
lean_mark_persistent(lp_aesop_Aesop_instBEqNodeState);
lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0 = _init_lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0();
lean_mark_persistent(lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0);
lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1 = _init_lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1();
lean_mark_persistent(lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1);
lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2 = _init_lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2();
lean_mark_persistent(lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2);
lp_aesop_Aesop_NodeState_instToString = _init_lp_aesop_Aesop_NodeState_instToString();
lean_mark_persistent(lp_aesop_Aesop_NodeState_instToString);
lp_aesop_Aesop_NodeState_toEmoji___closed__0 = _init_lp_aesop_Aesop_NodeState_toEmoji___closed__0();
lean_mark_persistent(lp_aesop_Aesop_NodeState_toEmoji___closed__0);
lp_aesop_Aesop_NodeState_toEmoji___closed__1 = _init_lp_aesop_Aesop_NodeState_toEmoji___closed__1();
lean_mark_persistent(lp_aesop_Aesop_NodeState_toEmoji___closed__1);
lp_aesop_Aesop_NodeState_toEmoji___closed__2 = _init_lp_aesop_Aesop_NodeState_toEmoji___closed__2();
lean_mark_persistent(lp_aesop_Aesop_NodeState_toEmoji___closed__2);
lp_aesop_Aesop_instInhabitedGoalState_default = _init_lp_aesop_Aesop_instInhabitedGoalState_default();
lp_aesop_Aesop_instInhabitedGoalState = _init_lp_aesop_Aesop_instInhabitedGoalState();
lp_aesop_Aesop_instBEqGoalState___closed__0 = _init_lp_aesop_Aesop_instBEqGoalState___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instBEqGoalState___closed__0);
lp_aesop_Aesop_instBEqGoalState = _init_lp_aesop_Aesop_instBEqGoalState();
lean_mark_persistent(lp_aesop_Aesop_instBEqGoalState);
lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0 = _init_lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0();
lean_mark_persistent(lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0);
lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1 = _init_lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1();
lean_mark_persistent(lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1);
lp_aesop_Aesop_GoalState_instToString = _init_lp_aesop_Aesop_GoalState_instToString();
lean_mark_persistent(lp_aesop_Aesop_GoalState_instToString);
lp_aesop_Aesop_instInhabitedNormalizationState_default = _init_lp_aesop_Aesop_instInhabitedNormalizationState_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormalizationState_default);
lp_aesop_Aesop_instInhabitedNormalizationState = _init_lp_aesop_Aesop_instInhabitedNormalizationState();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormalizationState);
lp_aesop_Aesop_instInhabitedGoalOrigin_default = _init_lp_aesop_Aesop_instInhabitedGoalOrigin_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalOrigin_default);
lp_aesop_Aesop_instInhabitedGoalOrigin = _init_lp_aesop_Aesop_instInhabitedGoalOrigin();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalOrigin);
lp_aesop_Aesop_GoalOrigin_toString___closed__0 = _init_lp_aesop_Aesop_GoalOrigin_toString___closed__0();
lean_mark_persistent(lp_aesop_Aesop_GoalOrigin_toString___closed__0);
lp_aesop_Aesop_GoalOrigin_toString___closed__1 = _init_lp_aesop_Aesop_GoalOrigin_toString___closed__1();
lean_mark_persistent(lp_aesop_Aesop_GoalOrigin_toString___closed__1);
lp_aesop_Aesop_GoalOrigin_toString___closed__2 = _init_lp_aesop_Aesop_GoalOrigin_toString___closed__2();
lean_mark_persistent(lp_aesop_Aesop_GoalOrigin_toString___closed__2);
lp_aesop_Aesop_GoalOrigin_toString___closed__3 = _init_lp_aesop_Aesop_GoalOrigin_toString___closed__3();
lean_mark_persistent(lp_aesop_Aesop_GoalOrigin_toString___closed__3);
lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0 = _init_lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0);
lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1 = _init_lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1);
lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0 = _init_lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0);
lp_aesop_Aesop_treeImpl = _init_lp_aesop_Aesop_treeImpl();
lean_mark_persistent(lp_aesop_Aesop_treeImpl);
lp_aesop_Aesop_Goal_instBEq = _init_lp_aesop_Aesop_Goal_instBEq();
lean_mark_persistent(lp_aesop_Aesop_Goal_instBEq);
lp_aesop_Aesop_Goal_instHashable = _init_lp_aesop_Aesop_Goal_instHashable();
lean_mark_persistent(lp_aesop_Aesop_Goal_instHashable);
lp_aesop_Aesop_Rapp_instBEq = _init_lp_aesop_Aesop_Rapp_instBEq();
lean_mark_persistent(lp_aesop_Aesop_Rapp_instBEq);
lp_aesop_Aesop_Rapp_instHashable = _init_lp_aesop_Aesop_Rapp_instHashable();
lean_mark_persistent(lp_aesop_Aesop_Rapp_instHashable);
lp_aesop_Aesop_Goal_safeRapps___closed__0 = _init_lp_aesop_Aesop_Goal_safeRapps___closed__0();
lean_mark_persistent(lp_aesop_Aesop_Goal_safeRapps___closed__0);
lp_aesop_Aesop_Goal_firstProvenRapp_x3f___closed__0 = _init_lp_aesop_Aesop_Goal_firstProvenRapp_x3f___closed__0();
lean_mark_persistent(lp_aesop_Aesop_Goal_firstProvenRapp_x3f___closed__0);
lp_aesop_Aesop_Goal_priority___closed__0 = _init_lp_aesop_Aesop_Goal_priority___closed__0();
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
