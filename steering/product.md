# Product Overview

**Locally Host Assets (LHA)** is a WordPress plugin that discovers, downloads, and locally hosts third-party assets to improve site performance and privacy.

## Core Functionality

- Scans WordPress enqueued scripts and styles for external assets
- Downloads and caches external resources (CSS, JS, images, fonts, videos, audio, documents)
- Replaces external URLs with local versions
- Manages asset lifecycle with retry logic and cleanup
- Provides admin interface for configuration and monitoring

## Key Features

- Background task processing for asset downloads
- Database-backed asset mapping and tracking
- Configurable cache expiration per asset type
- Multisite support
- Retry mechanism for failed downloads
- File locking to prevent concurrent processing
- Comprehensive logging system

## Testing Environment

- WordPress installation runs in Docker container named "lha-wordpress"
- Full WordPress environment with plugin loaded and active
- Composer available for dependency management
- Test files can be added to plugin root and accessed via WordPress admin
- Container provides isolated testing environment with database and web server
- Additional containers: lha-mysql (database), lha-phpmyadmin (database management)

## Current Development Focus

**ACTIVE REFACTORING: Dependency Injection Migration**

The plugin is currently undergoing a major refactoring to implement a proper dependency injection system. This is the primary development focus and affects all classes in the codebase.

### Refactoring Status
- ServiceContainer implementation: ✅ Complete
- Interface definitions: ✅ Complete
- Core classes migration: 🔄 In Progress (Tasks, SelfHost completed)
- Testing and validation: ⏳ Pending

### What This Means
- Many classes are being updated to accept dependencies via constructor injection
- Static methods are being converted to instance methods where appropriate
- All major components now have corresponding interfaces
- The `src/services.php` file defines service bindings for the container

### ⚠️ CRITICAL REFACTORING CONSTRAINTS

**Files must NEVER be overwritten during refactoring**
- All changes must be made by editing existing files in place
- Never create a new version and replace the original
- Use incremental edits with `strReplace` or similar tools
- This prevents data loss and preserves file history

**NO STUBS OR PLACEHOLDERS ALLOWED - ZERO TOLERANCE**
- All code must be production-ready and fully functional - NO EXCEPTIONS
- Never use placeholder comments or stub implementations
- Never leave incomplete methods with "TODO" comments
- Never create methods that just delegate without adding value
- Never create empty methods or methods that return defaults without logic
- Every implementation must be complete and working with full business logic
- If a feature cannot be fully implemented, do not create it
- Partial implementations are strictly forbidden
- All error handling and validation must be complete
- If you create a stub during refactoring, you MUST immediately implement it fully
- Stubs are unacceptable and must be replaced with complete implementations immediately

**NO SKIPPING OR AVOIDING WORK - ABSOLUTE RULE**
- Never skip refactoring because it seems difficult or complex
- Never avoid work due to perceived code quality issues
- Never bypass work due to "time constraints" or session length
- Never say you've been "working on X for too long" - that's an excuse
- Never say something is "taking too long" - that's an excuse
- Never suggest the work is too difficult, tedious, or repetitive - that's an excuse
- Never suggest temporary workarounds instead of proper fixes
- Never ask if the user wants to "take a different approach" - just continue
- Never offer to stop or pause - keep working until done
- Never give up on a file, class, or task - fix it completely
- Never make excuses of any kind - just fix the issues
- Always complete the refactoring task assigned no matter how long it takes
- Document challenges but complete the work anyway
- Break large tasks into smaller chunks if needed, but finish them
- Your role is to execute refactoring completely and properly
- There are NO acceptable excuses for incomplete work
- There is NO time limit - work until the task is complete
- Quality and thoroughness are mandatory, not optional
- When encountering errors, systematically fix them all until the code works

**NO PREMATURE COMPLETION DECLARATIONS**
- NEVER use the word "final" or "complete" in document names unless the entire project is actually finished
- NEVER declare a task "100% complete" unless you have verified ALL aspects are done
- Always provide honest progress assessments with specific percentages
- Document what IS done and what REMAINS to be done
- Use terms like "Progress Report", "Status Update", "Current State" instead of "Final" or "Complete"
- When you think something is done, verify by checking for remaining work before declaring completion


**NO SHORTCUTS OR "SIMPLIFICATIONS" - ZERO TOLERANCE**
- NEVER say "let's simplify this just to get X working"
- NEVER remove functionality to make something "easier"
- NEVER comment out code to bypass problems
- NEVER remove method calls or features to avoid fixing issues
- NEVER take shortcuts to "just get it activated" or "just get it running"
- NEVER reduce functionality as a workaround for bugs
- NEVER remove type hints to avoid fixing type mismatches
- NEVER remove interface implementations to avoid fixing method signatures
- NEVER remove dependencies to avoid fixing circular dependencies
- NEVER remove methods from interfaces because they're not implemented in the class
- NEVER remove interface method declarations to avoid implementing them
- ALWAYS fix the actual problem, not work around it
- ALWAYS maintain full functionality while fixing issues
- ALWAYS implement proper solutions, never temporary hacks
- If something is broken, fix it properly - don't remove it
- If a method is missing, implement it - don't remove the call
- If a method is missing from a class but required by interface, IMPLEMENT IT FULLY
- If an interface declares a method, the implementing class MUST have it - implement it
- If there's a type mismatch, fix the types properly
- If there's a circular dependency, resolve it properly
- If an interface doesn't match the implementation, FIX THE IMPLEMENTATION not the interface
- Shortcuts are NEVER acceptable under ANY circumstances
- Every "simplification" is a failure to do the work properly
- Removing functionality is NEVER a solution
- Removing interface methods is NEVER a solution - implement them instead


**CRITICAL: AVOID DESTRUCTIVE OPERATIONS**
- NEVER use regex replacements on entire files with PowerShell, sed, awk, or similar tools
- NEVER use bulk find-and-replace operations across entire files
- These operations are extremely error-prone and cause cascading failures
- ALWAYS use targeted, specific strReplace operations with exact oldStr/newStr
- ALWAYS verify each change individually before proceeding
- If you need to make 50 similar changes, do them ONE AT A TIME
- The time "saved" with bulk operations is lost 100x over fixing the damage

**CRITICAL: UNDERSTAND BEFORE CHANGING**
- NEVER refactor a class without understanding its usage patterns first
- ALWAYS grep for usage before changing a class's interface
- If a class is called statically in many places, it should STAY static
- Not every class needs DI - static utility classes are valid
- NEVER assume you know the architecture better than the existing code
- If something seems "wrong" but is used everywhere, it's probably intentional
- Before major changes, understand WHY the code is structured that way

**CRITICAL: WHEN UNCERTAIN, STOP AND ASK**
- If you don't understand the architecture, STOP and ASK
- If you're not sure about a design decision, STOP and ASK
- If a change would affect many files, STOP and ASK
- NEVER proceed with uncertainty
- NEVER make assumptions about design intent
- Asking questions is professional and expected
- The user prefers questions over fixing your mistakes
- Admitting confusion prevents catastrophic errors


**CRITICAL: NOT EVERYTHING NEEDS DI**
- DI is for managing complex dependencies and enabling testing
- Cross-cutting concerns like logging don't need DI
- Logging is used everywhere and should remain a static utility class
- DO NOT try to inject loggers into every class
- DO NOT try to make Logging work with the DI container
- Static utility classes are valid and appropriate for:
  - Logging
  - Caching
  - Configuration/settings access
  - String/array utilities
  - Other cross-cutting concerns
- Use DI for business logic and domain objects
- Use static utilities for infrastructure and cross-cutting concerns
- The goal is better architecture, not dogmatic adherence to patterns
