graph TB
    Start([🎯 USER SUBMITS<br/>COMPLEX PROBLEM])
    
    Start --> S0_Init[Stage 0: Content Analysis]
    
    S0_Init --> S0_Team{{🔵 Content Analyzer<br/>Blue Team}}
    S0_Team --> S0_Process[Analyze Problem:<br/>• Domain & Keywords<br/>• Complexity 1-10<br/>• Challenges<br/>• Required Expertise]
    S0_Process --> S0_Output[(📊 AnalyzedContext)]
    
    S0_Output --> S1_Init[Stage 1: AI Decomposition]
    
    S1_Init --> S1_Team{{🔵 Planner<br/>Blue Team}}
    S1_Team --> S1_Process[Generate Plan:<br/>• Define SubProblems<br/>• Map Dependencies<br/>• Suggest Strategies<br/>• Draft Criteria]
    S1_Process --> S1_Validate{Valid<br/>Structure?}
    S1_Validate -->|No| S1_Team
    S1_Validate -->|Yes| S1_Output[(📋 DecompositionPlan)]
    
    S1_Output --> S2_Init[Stage 2: User Review]
    
    S2_Init --> S2_UI[👤 Interactive UI]
    S2_UI --> S2_Review[Review & Edit:<br/>• SubProblems<br/>• Dependencies<br/>• Teams & Gauntlets<br/>• Parameters]
    S2_Review --> S2_Decision{Approve?}
    S2_Decision -->|❌ Reject| End_Reject([❌ TERMINATED])
    S2_Decision -->|✅ Approve| S2_Output[(✅ ApprovedPlan)]
    
    S2_Output --> S3_Init[Stage 3: Solve SubProblems]
    
    S3_Init --> S3_Sort[Sort by Dependencies]
    S3_Sort --> S3_Loop{Next<br/>SubProblem?}
    S3_Loop -->|None Left| S3_Done[✅ All Verified]
    S3_Loop -->|Select Next| S3_Gen_Init
    
    S3_Gen_Init[A: Generate Solution] --> S3_Gen_Team{{🔵 Solver<br/>Blue Team}}
    S3_Gen_Team --> S3_Gen_Mode{Mode?}
    S3_Gen_Mode -->|Single| S3_Gen_Single[Direct Generation]
    S3_Gen_Mode -->|Multi| S3_Gen_Multi[Multi-Candidate<br/>+ Peer Review]
    S3_Gen_Single --> S3_Gen_Output
    S3_Gen_Multi --> S3_Gen_Output[(💡 SolutionAttempt)]
    
    S3_Gen_Output --> S3_Red_Init[B: Red Team Critique]
    
    S3_Red_Init --> S3_Red_Team{{🔴 Assailants<br/>Red Team}}
    S3_Red_Team --> S3_Red_Attack[Attack Modes:<br/>• Security Scan<br/>• Edge Cases<br/>• Assumptions<br/>• Stress Test]
    S3_Red_Attack --> S3_Red_Rounds[Multi-Round<br/>Gauntlet]
    S3_Red_Rounds --> S3_Red_Rules[Check Rules:<br/>• Quorum M/N<br/>• Confidence<br/>• Variance]
    S3_Red_Rules --> S3_Red_Pass{Pass?}
    
    S3_Red_Pass -->|❌ Flaws| S3_Red_Report[(🔴 CritiqueReport)]
    S3_Red_Report --> S3_Red_Heal{Attempts<br/>< Max?}
    S3_Red_Heal -->|No| S3_Red_Alert[⚠️ Max Attempts]
    S3_Red_Alert --> S3_User1{User?}
    S3_User1 -->|Skip| S3_Loop
    S3_User1 -->|Fix| S3_Manual1[Manual Solution]
    S3_Manual1 --> S3_Red_Init
    S3_Red_Heal -->|Yes| S3_Red_Patch{{🔵 Patcher<br/>Blue Team}}
    S3_Red_Patch --> S3_Red_Fix[Generate Patch<br/>from Report]
    S3_Red_Fix --> S3_Red_Init
    
    S3_Red_Pass -->|✅ Robust| S3_Gold_Init[C: Gold Team Verification]
    
    S3_Gold_Init --> S3_Gold_Team{{🟡 Judges<br/>Gold Team}}
    S3_Gold_Team --> S3_Gold_Eval[Evaluate:<br/>• Correctness<br/>• Quality<br/>• Requirements<br/>• Custom Criteria]
    S3_Gold_Eval --> S3_Gold_Rounds[Multi-Round<br/>Gauntlet]
    S3_Gold_Rounds --> S3_Gold_Rules[Check Rules:<br/>• Quorum M/N<br/>• Confidence<br/>• Variance<br/>• Collaboration]
    S3_Gold_Rules --> S3_Gold_Pass{Pass?}
    
    S3_Gold_Pass -->|❌ Issues| S3_Gold_Report[(🟡 VerificationReport)]
    S3_Gold_Report --> S3_Gold_Heal{Attempts<br/>< Max?}
    S3_Gold_Heal -->|No| S3_Gold_Alert[⚠️ Max Attempts]
    S3_Gold_Alert --> S3_User2{User?}
    S3_User2 -->|Skip| S3_Loop
    S3_User2 -->|Fix| S3_Manual2[Manual Solution]
    S3_Manual2 --> S3_Gold_Init
    S3_Gold_Heal -->|Yes| S3_Gold_Patch{{🔵 Patcher<br/>Blue Team}}
    S3_Gold_Patch --> S3_Gold_Fix[Generate Patch<br/>from Report]
    S3_Gold_Fix --> S3_Red_Init
    
    S3_Gold_Pass -->|✅ Verified| S3_Store[(✅ VerifiedSolution)]
    S3_Store --> S3_Loop
    
    S3_Done --> S4_Init[Stage 4: Reassembly]
    
    S4_Init --> S4_Team{{🔵 Assembler<br/>Blue Team}}
    S4_Team --> S4_Gather[Gather All<br/>Verified Solutions]
    S4_Gather --> S4_Integrate[Integrate:<br/>• Respect Dependencies<br/>• Synthesize Coherently<br/>• Polish Output]
    S4_Integrate --> S4_Internal{Internal<br/>Review?}
    S4_Internal -->|Yes| S4_Check[Self-Review]
    S4_Check --> S4_Output
    S4_Internal -->|No| S4_Output[(🧬 FinalCandidate)]
    
    S4_Output --> S5_Init[Stage 5: Final Verification]
    
    S5_Init --> S5_Counter[refinement_loop = 0]
    S5_Counter --> S5_Red_Init
    
    S5_Red_Init[Final Red Gauntlet] --> S5_Red_Team{{🔴 Final Attack<br/>Red Team}}
    S5_Red_Team --> S5_Red_Test[Test Integration:<br/>• Integration Errors<br/>• New Vulnerabilities<br/>• Consistency<br/>• Conflicts]
    S5_Red_Test --> S5_Red_Rounds[Multi-Round<br/>Gauntlet]
    S5_Red_Rounds --> S5_Red_Rules[Apply Rules]
    S5_Red_Rules --> S5_Red_Pass{Pass?}
    
    S5_Red_Pass -->|❌ Fail| S5_Red_Report[(🔴 Final Critique)]
    S5_Red_Report --> S5_Heal_Init
    
    S5_Red_Pass -->|✅ Pass| S5_Gold_Init
    
    S5_Gold_Init[Final Gold Gauntlet] --> S5_Gold_Team{{🟡 Final Judges<br/>Gold Team}}
    S5_Gold_Team --> S5_Gold_Eval[Holistic Evaluation:<br/>• Problem Solved?<br/>• Overall Quality<br/>• Completeness<br/>• Expert Review]
    S5_Gold_Eval --> S5_Gold_Rounds[Multi-Round<br/>Gauntlet]
    S5_Gold_Rounds --> S5_Gold_Rules[Strict Rules:<br/>• High Confidence<br/>• Low Variance<br/>• Consensus]
    S5_Gold_Rules --> S5_Gold_Pass{Pass?}
    
    S5_Gold_Pass -->|❌ Fail| S5_Gold_Report[(🟡 Final Verification)]
    S5_Gold_Report --> S5_Heal_Init
    
    S5_Heal_Init[🔄 Global Self-Healing] --> S5_Heal_Parse[Parse Targeted<br/>Feedback]
    S5_Heal_Parse --> S5_Heal_ID[Identify Problematic<br/>SubProblem IDs]
    S5_Heal_ID --> S5_Heal_Valid{Valid IDs?}
    
    S5_Heal_Valid -->|No| S5_Heal_Manual[⚠️ General Failure]
    S5_Heal_Manual --> S5_User3{User?}
    S5_User3 -->|Fix| S5_Manual_Fix[Manual Adjustment]
    S5_Manual_Fix --> S5_Red_Init
    S5_User3 -->|Abort| End_Fail
    
    S5_Heal_Valid -->|Yes| S5_Heal_Flag[Flag SubProblems<br/>for Rework]
    S5_Heal_Flag --> S5_Heal_Check{Loop Count<br/>< Max?}
    
    S5_Heal_Check -->|No| S5_Heal_Max[⚠️ MAX LOOPS REACHED]
    S5_Heal_Max --> S5_Heal_Alert[Alert User with<br/>Full Details]
    S5_Heal_Alert --> S5_User4{User?}
    S5_User4 -->|Accept| End_Partial
    S5_User4 -->|Fix| S5_Manual_Fix2[Manual Fix]
    S5_Manual_Fix2 --> S5_Red_Init
    S5_User4 -->|Abort| End_Fail
    
    S5_Heal_Check -->|Yes| S5_Heal_Inc[Increment Loop]
    S5_Heal_Inc --> S5_Heal_Clear[Clear Flagged<br/>Solutions]
    S5_Heal_Clear --> S5_Heal_Return[🔄 Return to Stage 3]
    S5_Heal_Return --> S3_Loop
    
    S5_Gold_Pass -->|✅ VERIFIED| S5_Finalize[Finalize Workflow]
    
    S5_User4 -->|Abort| End_Fail
    S5_User3 -->|Abort| End_Fail
    S5_User4 -->|Accept| End_Partial
    
    End_Fail([❌ FAILED<br/>Manual Intervention Needed])
    End_Partial([⚠️ PARTIAL SUCCESS<br/>Accepted with Issues])
    
    S5_Finalize --> S5_Package[Package Final Solution:<br/>• All Verification Reports<br/>• Quality Scores<br/>• Metadata & Trace<br/>• Timestamp]
    S5_Package --> S5_Final[(🏆 VerifiedFinalSolution)]
    S5_Final --> S5_Present[Present to User]
    S5_Present --> S5_Archive[Archive Complete<br/>Workflow State]
    S5_Archive --> S5_UserNotify[Notify User of Success]
    S5_UserNotify --> S5_GenerateReport[Generate Final Report]
    S5_GenerateReport --> S5_SaveArtifacts[Save All Artifacts]
    S5_SaveArtifacts --> S5_Cleanup[Cleanup Temporary Data]
    S5_Cleanup --> S5_LogSuccess[Log Success Metrics]
    S5_LogSuccess --> End_Success
    
    End_Success([✅ ✅ ✅ SUCCESS ✅ ✅ ✅<br/>SOVEREIGN-GRADE SOLUTION DELIVERED<br/>ALL VERIFICATIONS PASSED])
    
    classDef blueTeam fill:#2196f3,stroke:#0d47a1,stroke-width:3px,color:#fff,font-weight:bold
    classDef redTeam fill:#f44336,stroke:#b71c1c,stroke-width:3px,color:#fff,font-weight:bold
    classDef goldTeam fill:#ffc107,stroke:#f57f17,stroke-width:3px,color:#000,font-weight:bold
    classDef userControl fill:#ff9800,stroke:#e65100,stroke-width:3px,color:#fff,font-weight:bold
    classDef dataStore fill:#90a4ae,stroke:#37474f,stroke-width:2px,color:#fff,font-weight:bold
    classDef success fill:#4caf50,stroke:#1b5e20,stroke-width:4px,color:#fff,font-weight:bold
    classDef failure fill:#f44336,stroke:#b71c1c,stroke-width:4px,color:#fff,font-weight:bold
    classDef warning fill:#ff9800,stroke:#e65100,stroke-width:3px,color:#000,font-weight:bold
    classDef healing fill:#9c27b0,stroke:#4a148c,stroke-width:3px,color:#fff,font-weight:bold
    classDef stage fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    
    class S0_Team,S1_Team,S3_Gen_Team,S3_Red_Patch,S3_Gold_Patch,S4_Team blueTeam
    class S3_Red_Team,S5_Red_Team redTeam
    class S3_Gold_Team,S5_Gold_Team goldTeam
    class S2_UI,S2_Review userControl
    class S0_Output,S1_Output,S2_Output,S3_Gen_Output,S3_Store,S4_Output,S5_Final dataStore
    class Start,End_Success success
    class End_Reject,End_Fail failure
    class End_Partial,S3_Red_Alert,S3_Gold_Alert,S5_Heal_Manual,S5_Heal_Max,S5_Heal_Alert warning
    class S3_Red_Fix,S3_Gold_Fix,S5_Heal_Init,S5_Heal_Inc,S5_Heal_Clear,S5_Heal_Return healing
    class S0_Init,S1_Init,S2_Init,S3_Init,S4_Init,S5_Init stage
