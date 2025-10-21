%% Full-fidelity Sovereign-Grade Decomposition Workflow (TB) - GitHub Markdown
graph TB
    %% ================= USER INPUT =================
    Start([🎯 USER SUBMITS<br/>COMPLEX PROBLEM])

    %% ================= STAGE 0: CONTENT ANALYSIS =================
    subgraph Stage0["Stage 0: Content Analysis"]
        direction TB
        S0_Init(("Initialize Stage 0"))
        S0_Team{{🔵 Content Analyzer<br/>Blue Team}}
        S0_Process[Analyze Problem:<br/>• Domain & Keywords<br/>• Complexity 1-10<br/>• Challenges<br/>• Required Expertise]
        S0_Output[(📊 AnalyzedContext)]
        S0_Init --> S0_Team --> S0_Process --> S0_Output
    end
    Start --> S0_Init

    %% ================= STAGE 1: AI DECOMPOSITION =================
    subgraph Stage1["Stage 1: AI Decomposition"]
        direction TB
        S1_Init(("Initialize Stage 1"))
        S1_Team{{🔵 Planner<br/>Blue Team}}
        S1_Process[Generate Plan:<br/>• Define SubProblems<br/>• Map Dependencies<br/>• Suggest Strategies<br/>• Draft Criteria]
        S1_Validate{Valid<br/>Structure?}
        S1_Output[(📋 DecompositionPlan<br/>N SubProblems)]
        S1_Init --> S1_Team --> S1_Process --> S1_Validate
        S1_Validate -->|No| S1_Team
        S1_Validate -->|Yes| S1_Output
    end
    S0_Output --> S1_Init

    %% ================= STAGE 2: USER REVIEW & OVERRIDE =================
    subgraph Stage2["Stage 2: User Review & Override"]
        direction TB
        S2_Init(("Initialize Stage 2"))
        S2_UI[👤 Interactive UI]
        S2_Review[Review & Edit:<br/>• SubProblems<br/>• Dependencies<br/>• Teams & Gauntlets<br/>• Acceptance Thresholds]
        S2_Decision{Approve?}
        S2_Output[(✅ ApprovedPlan)]
        End_Reject([❌ TERMINATED<br/>User Rejected Plan])

        S2_Init --> S2_UI --> S2_Review --> S2_Decision
        S2_Decision -->|✅ Approve| S2_Output
        S2_Decision -->|❌ Reject| End_Reject
    end
    S1_Output --> S2_Init

    %% ================= STAGE 3: SUB-PROBLEM SOLVING LOOP =================
    subgraph Stage3["Stage 3: SubProblem Loop (Iterative Solve/Critique/Verify)"]
        direction TB
        %% Queue and loop init
        S3_Init[Initialize Queue:<br/>Sort by Dependencies]
        S3_Check{All SubProblems<br/>Verified?}
        S3_Select[Select Next<br/>Unverified SubProblem]
        S3_Attempt[Attempt Counter = 0]
        S3_Init --> S3_Check

        %% Blue Team generation (A)
        S3A_Init[🔵 BLUE TEAM: Generate Solution]
        S3A_Team{{🔵 Solver Team}}
        S3A_Gen[Generate Solution<br/>for SubProblem]
        S3A_Output[(💡 Solution Candidate)]

        %% Red Team critique (B)
        S3B_Init[🔴 RED TEAM: Critique & Attack]
        S3B_Team{{🔴 Assailant Team}}
        S3B_Attack[Multi-Round Attack:<br/>• Security Vulnerabilities<br/>• Edge Cases<br/>• Logic Flaws<br/>• Assumption Holes<br/>• Stress Testing]
        S3B_Gauntlet[Red Team Gauntlet:<br/>• Quorum M of N<br/>• Multiple Rounds<br/>• Attack Modes<br/>• Severity Scores]
        S3B_Decision{Red Team<br/>Approves?}
        S3B_Report[(🔴 Critique Report:<br/>Detailed Flaws)]
        S3B_Check{Max Attempts<br/>Reached?}
        S3B_Patch{{🔵 Patcher Team}}
        S3B_Fix[Analyze Critique<br/>Generate Targeted Fix]
        S3B_Inc[Increment Attempts]
        S3B_Fail[❌ SubProblem Failed<br/>Red Team Validation]

        %% Gold Team verification (C)
        S3C_Init[🟡 GOLD TEAM: Verification]
        S3C_Team{{🟡 Judge Team}}
        S3C_Eval[Multi-Round Evaluation:<br/>• Correctness Check<br/>• Quality Assessment<br/>• Requirement Match<br/>• Completeness Score<br/>• Acceptance Criteria]
        S3C_Gauntlet[Gold Team Gauntlet:<br/>• Quorum M of N<br/>• Min Confidence Score<br/>• Max Score Variance<br/>• Per-Judge Requirements<br/>• Collaboration Rounds]
        S3C_Decision{Meets<br/>Acceptance<br/>Threshold?}
        S3C_Report[(🟡 Verification Report:<br/>Quality Issues)]
        S3C_Check{Max Attempts<br/>Reached?}
        S3C_Patch{{🔵 Patcher Team}}
        S3C_Fix[Analyze Report<br/>Generate Quality Fix]
        S3C_Inc[Increment Attempts]
        S3C_Fail[❌ SubProblem Failed<br/>Gold Team Validation]

        %% Storage and status
        S3_Store[(✅ Verified SubProblem<br/>Stored)]
        S3_MarkFailed[Mark as Failed]

        %% User intervention nodes (shared)
        S3_UserInt1{User<br/>Intervention?}
        S3_UserInt2{User<br/>Intervention?}
        S3_Manual1[User Provides Solution]
        S3_Manual2[User Provides Solution]
        S3_Reconfig1[Adjust Teams/Thresholds]
        S3_Reconfig2[Adjust Teams/Thresholds]

        %% Completion & validation
        S3_Complete[SubProblem Solving<br/>Complete]
        S3_ValidateAll{All SubProblems<br/>Successfully<br/>Verified?}
        S3_FailedAlert[⚠️ Some SubProblems Failed]
        S3_UserDecision{User<br/>Decision?}

        %% flows: main selection to attempt
        S3_Check -->|No - Select Next| S3_Select --> S3_Attempt --> S3A_Init
        S3A_Init --> S3A_Team --> S3A_Gen --> S3A_Output

        %% to red critique
        S3A_Output --> S3B_Init --> S3B_Team --> S3B_Attack --> S3B_Gauntlet --> S3B_Decision

        %% red decision paths
        S3B_Decision -->|❌ Flaws Found| S3B_Report --> S3B_Check
        S3B_Check -->|Yes| S3B_Fail
        S3B_Check -->|No| S3B_Patch --> S3B_Fix --> S3B_Inc --> S3A_Init

        S3B_Decision -->|✅ Robust| S3C_Init

        %% red fail handling
        S3B_Fail --> S3_UserInt1
        S3_UserInt1 -->|Skip| S3_MarkFailed
        S3_UserInt1 -->|Manual Fix| S3_Manual1 --> S3C_Init
        S3_UserInt1 -->|Reconfigure| S3_Reconfig1 --> S3A_Init

        %% gold verification flow
        S3C_Init --> S3C_Team --> S3C_Eval --> S3C_Gauntlet --> S3C_Decision
        S3C_Decision -->|✅ Approved| S3_Store --> S3_Check

        S3C_Decision -->|❌ Below Threshold| S3C_Report --> S3C_Check
        S3C_Check -->|Yes| S3C_Fail
        S3C_Check -->|No| S3C_Patch --> S3C_Fix --> S3C_Inc --> S3B_Init

        %% gold fail handling
        S3C_Fail --> S3_UserInt2
        S3_UserInt2 -->|Skip| S3_MarkFailed
        S3_UserInt2 -->|Manual Fix| S3_Manual2 --> S3C_Init
        S3_UserInt2 -->|Reconfigure| S3_Reconfig2 --> S3A_Init

        %% completion & validation
        S3_MarkFailed --> S3_Check
        S3_Check -->|Yes - All Done| S3_Complete
        S3_Complete --> S3_ValidateAll
        S3_ValidateAll -->|No - Some Failed| S3_FailedAlert --> S3_UserDecision
        S3_UserDecision -->|Abort| End_Incomplete([❌ INCOMPLETE<br/>Some SubProblems Failed])
        S3_UserDecision -->|Continue Partial| S4_Init
        S3_ValidateAll -->|Yes - All Verified| S4_Init
    end
    S2_Output --> S3_Init

    %% ================= STAGE 4: REASSEMBLY =================
    subgraph Stage4["Stage 4: Reassembly"]
        direction TB
        S4_Init[Stage 4: Reassembly]
        S4_Team{{🔵 Assembler Team}}
        S4_Gather[Gather All Verified<br/>SubProblem Solutions]
        S4_Integrate[Integrate & Synthesize:<br/>• Respect Dependencies<br/>• Maintain Coherence<br/>• Resolve Interfaces<br/>• Polish & Format]
        S4_Output[(🧬 Assembled Solution<br/>Candidate)]
        S4_Init --> S4_Team --> S4_Gather --> S4_Integrate --> S4_Output
    end

    %% ================= STAGE 5: FINAL VERIFICATION & SELF-HEALING =================
    subgraph Stage5["Stage 5: Final Verification & Self-Healing"]
        direction TB
        S5_Init[refinement_loop = 0]

        %% Final red
        S5_Red_Init[🔴 RED TEAM: Final Attack]
        S5_Red_Team{{🔴 Final Assailant Team}}
        S5_Red_Attack[Integration Testing:<br/>• Component Integration<br/>• Interface Consistency<br/>• New Vulnerabilities<br/>• System-Level Flaws<br/>• Emergent Issues]
        S5_Red_Gauntlet[Red Team Gauntlet:<br/>Multi-Round Attack]
        S5_Red_Decision{Red Team<br/>Approves?}
        S5_Red_Report[(🔴 Final Critique:<br/>Integration Issues)]

        %% Final gold
        S5_Gold_Init[🟡 GOLD TEAM: Final Judgment]
        S5_Gold_Team{{🟡 Final Judge Team}}
        S5_Gold_Eval[Holistic Evaluation:<br/>• Original Problem Solved?<br/>• Overall Quality Score<br/>• Completeness Check<br/>• Expert-Level Review<br/>• Acceptance Criteria]
        S5_Gold_Gauntlet[Gold Team Gauntlet:<br/>• Strict Quorum<br/>• High Confidence Threshold<br/>• Low Score Variance<br/>• Consensus Required]
        S5_Gold_Decision{Meets Final<br/>Acceptance<br/>Threshold?}
        S5_Gold_Report[(🟡 Final Verification:<br/>Quality/Completeness Issues)]

        %% Decompose/back-to-stage3 logic
        S5_Decompose_Init[🔄 DECOMPOSE BACK]
        S5_Parse[Parse Targeted Feedback:<br/>Identify Root Cause]
        S5_Identify[Identify Problematic<br/>SubProblem IDs]
        S5_Valid{Valid SubProblem<br/>IDs Found?}
        S5_General[Cannot Isolate<br/>Specific SubProblem]
        S5_UserReview{User<br/>Review?}
        S5_ManualFix[User Adjusts<br/>Final Solution]
        S5_Flag[Flag Identified<br/>SubProblems for Rework]
        S5_LoopCheck{refinement_loop<br/>< max_loops?}
        S5_MaxLoops[⚠️ MAX REFINEMENT<br/>LOOPS REACHED]
        S5_FinalUser{User<br/>Final Decision?}
        S5_ManualFix2[User Manually<br/>Fixes Issues]
        S5_Increment[Increment refinement_loop]
        S5_Clear[Clear Flagged<br/>SubProblem Solutions]
        S5_Return[🔄 RETURN TO STAGE 3:<br/>Re-solve Flagged SubProblems]
        S5_Success[✅ Final Solution<br/>Meets All Criteria]

        %% flows
        S4_Output --> S5_Init --> S5_Red_Init
        S5_Red_Init --> S5_Red_Team --> S5_Red_Attack --> S5_Red_Gauntlet --> S5_Red_Decision
        S5_Red_Decision -->|❌ Integration Flaws| S5_Red_Report --> S5_Decompose_Init
        S5_Red_Decision -->|✅ Passes| S5_Gold_Init
        S5_Gold_Init --> S5_Gold_Team --> S5_Gold_Eval --> S5_Gold_Gauntlet --> S5_Gold_Decision
        S5_Gold_Decision -->|❌ Below Threshold| S5_Gold_Report --> S5_Decompose_Init
        S5_Gold_Decision -->|✅ APPROVED| S5_Success

        %% decompose/init parsing flow
        S5_Decompose_Init --> S5_Parse --> S5_Identify --> S5_Valid
        S5_Valid -->|No - General Issue| S5_General --> S5_UserReview
        S5_UserReview -->|Manual Fix| S5_ManualFix --> S5_Red_Init
        S5_UserReview -->|Abort| End_Fail([❌ FAILED<br/>Manual Intervention Needed])

        S5_Valid -->|Yes| S5_Flag --> S5_LoopCheck
        S5_LoopCheck -->|No| S5_MaxLoops --> S5_FinalUser
        S5_FinalUser -->|Accept Partial| End_Partial([⚠️ PARTIAL SUCCESS<br/>Accepted with Issues])
        S5_FinalUser -->|Manual Fix| S5_ManualFix2 --> S5_Red_Init
        S5_FinalUser -->|Abort| End_Fail

        S5_LoopCheck -->|Yes| S5_Increment --> S5_Clear --> S5_Return --> S3_Check
    end

    %% ================= STAGE 6: FINALIZATION =================
    subgraph Stage6["Stage 6: Finalization & Delivery"]
        direction TB
        S6_Init[Stage 6: Finalization]
        S6_Package[Package Complete Solution:<br/>• All Verification Reports<br/>• Critique Reports<br/>• Quality Scores<br/>• SubProblem Solutions<br/>• Metadata & Trace<br/>• Timestamps]
        S6_Document[Generate Documentation:<br/>• Technical Specifications<br/>• Solution Architecture<br/>• Quality Certificates<br/>• Verification Attestations<br/>• Usage Guidelines]
        S6_Certify[Issue Certificates:<br/>• Sovereign-Grade Badge<br/>• Quality Certification<br/>• Compliance Attestation<br/>• Security Clearance<br/>• Verification Proof]
        S6_Archive[Archive Everything:<br/>• Full WorkflowState<br/>• All Reports & Scores<br/>• Team Configurations<br/>• Version Control<br/>• Backup Archives]
        S6_Report[Generate Final Report:<br/>• Executive Summary<br/>• Detailed Analysis<br/>• All Iterations Logged<br/>• Performance Metrics<br/>• Quality Dashboard]
        S6_Notify[Notify Stakeholders:<br/>• Success Notification<br/>• Delivery Confirmation<br/>• Access Instructions<br/>• Support Contacts<br/>• Deployment Info]
        S6_Handoff[Complete Handoff:<br/>• Client Presentation<br/>• Knowledge Transfer<br/>• Training Materials<br/>• Maintenance Plan<br/>• Support Setup]
        S6_Monitor[Setup Monitoring:<br/>• Performance Tracking<br/>• Health Checks<br/>• Alert Systems<br/>• Analytics Dashboard<br/>• Feedback Loop]
        S6_QA[Final QA & Validation:<br/>• End-to-End Testing<br/>• Compliance Check<br/>• Security Audit<br/>• Performance Validation<br/>• Reproducibility Test]
        S6_Release[Prepare Release:<br/>• Staging Setup<br/>• Release Notes<br/>• Deployment Package<br/>• Rollback Plan<br/>• Launch Checklist]
        S6_Stats[Update Statistics:<br/>• Success Metrics<br/>• Quality Benchmarks<br/>• Performance Baseline<br/>• Historical Records<br/>• Team Analytics]
        S6_Celebrate[🎉 Mark Achievement:<br/>Sovereign-Grade<br/>Solution Delivered]
        S6_Final[(🏆 VerifiedFinalSolution<br/>Ready for Deployment)]
        S6_Init --> S6_Package --> S6_Document --> S6_Certify --> S6_Archive --> S6_Report --> S6_Notify --> S6_Handoff --> S6_Monitor --> S6_QA --> S6_Release --> S6_Stats --> S6_Celebrate --> S6_Final
    end
    S5_Success --> S6_Init

    %% ================= TERMINAL NODES =================
    End_Reject([❌ TERMINATED<br/>User Rejected Plan])
    End_Incomplete([❌ INCOMPLETE<br/>Some SubProblems Failed])
    End_Fail([❌ FAILED<br/>Manual Intervention Needed])
    End_Partial([⚠️ PARTIAL SUCCESS<br/>Accepted with Issues])
    End_Success([🎉 COMPLETE SUCCESS 🎉<br/>Sovereign-Grade Solution<br/>Successfully Delivered<br/>✅ All Verifications Passed<br/>✅ All Thresholds Met<br/>✅ Iterative Refinement Complete<br/>🏆 Ready for Deployment 🏆])
    S6_Final --> End_Success

    %% ================= AUXILIARY LINKS/CITATIONS =================
    %% (linking key artifacts across stages)
    S0_Output --> S1_Init
    S1_Output --> S2_Init
    S2_Output --> S3_Init
    S3_Store --> S4_Init
    S4_Output --> S5_Init
    S5_Success --> S6_Init

    %% ================= CLASS STYLING =================
    classDef blueTeam fill:#2196f3,stroke:#0d47a1,stroke-width:3px,color:#fff,font-weight:bold
    classDef redTeam fill:#f44336,stroke:#b71c1c,stroke-width:3px,color:#fff,font-weight:bold
    classDef goldTeam fill:#ffc107,stroke:#f57f17,stroke-width:3px,color:#000,font-weight:bold
    classDef userControl fill:#ff9800,stroke:#e65100,stroke-width:3px,color:#fff,font-weight:bold
    classDef dataStore fill:#00acc1,stroke:#006064,stroke-width:3px,color:#fff,font-weight:bold
    classDef success fill:#4caf50,stroke:#1b5e20,stroke-width:4px,color:#fff,font-weight:bold
    classDef failure fill:#f44336,stroke:#b71c1c,stroke-width:4px,color:#fff,font-weight:bold
    classDef warning fill:#ff9800,stroke:#e65100,stroke-width:3px,color:#000,font-weight:bold
    classDef healing fill:#9c27b0,stroke:#4a148c,stroke-width:3px,color:#fff,font-weight:bold
    classDef stage fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef loop fill:#e1bee7,stroke:#6a1b9a,stroke-width:3px,color:#000,font-weight:bold

    class S0_Team,S1_Team,S3A_Team,S3B_Patch,S3C_Patch,S4_Team blueTeam
    class S3B_Team,S5_Red_Team redTeam
    class S3C_Team,S5_Gold_Team goldTeam
    class S2_UI,S2_Review userControl
    class S0_Output,S1_Output,S2_Output,S3A_Output,S3_Store,S4_Output,S6_Final dataStore
    class Start,End_Success success
    class End_Reject,End_Incomplete,End_Fail failure
    class End_Partial,S3B_Fail,S3C_Fail,S5_MaxLoops warning
    class S3B_Fix,S3C_Fix,S5_Parse,S5_Identify,S5_Flag,S5_Clear,S5_Return healing
    class S0_Init,S1_Init,S2_Init,S4_Init,S6_Init stage
    class S3_Init,S3_Check,S5_Init,S5_Decompose_Init loop
