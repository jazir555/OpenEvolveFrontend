# Project Structure

## Root Directory

- `locally-host-assets.php`: Main plugin file with initialization
- `ServiceContainer.php`: Dependency injection container
- `*.php`: Individual class files (one class per file)

## Key Directories

### `/interfaces`
Contains all interface definitions following the pattern `{ClassName}Interface.php`
- All interfaces are in the `LHA\Interfaces` namespace
- Interfaces define contracts for dependency injection

### `/Admin`
Admin-related functionality
- `AdminPages.php`: Admin UI and settings pages

### `/src`
Service configuration
- `services.php`: Service definitions for dependency injection container

### `/.kiro`
Kiro IDE configuration (not part of plugin)

## ⚠️ CRITICAL: Docker Bind Mount Warning

**This workspace has a Docker bind mount to the lha-wordpress container:**
- Local path: `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes`
- Container path: `/var/www/html/wp-content/plugins/locally-host-assets`

**THESE ARE THE SAME FILES - NOT COPIES!**

**NEVER run delete commands inside the container** - they will permanently delete your local work files without going to Recycle Bin.

Always check mounts before container operations: `docker inspect lha-wordpress | Select-String -Pattern "Mounts"`

## File Naming Conventions

- **Classes**: PascalCase (e.g., `AssetData.php`, `ServiceContainer.php`)
- **Interfaces**: PascalCase with `Interface` suffix (e.g., `DatabaseInterface.php`)
- **One class per file**: Each PHP file contains exactly one class
- **Namespace matches structure**: `LHA\` namespace for root classes, `LHA\Interfaces\` for interfaces

## Core Classes

### Initialization & Container
- `Initialize.php`: Plugin activation, deactivation, setup
- `ServiceContainer.php`: DI container implementation

### Asset Management
- `Scanner.php`: Discovers external assets
- `AssetData.php`: Asset data management
- `AssetActionHandler.php`: Handles asset processing actions
- `AssetOrder.php`: Asset ordering logic
- `AssetValidator.php`: Asset validation
- `AssetUtils.php`: Asset utility functions

### Processing
- `SelfHost.php`: Main download and hosting logic
- `Process.php`: Asset processing orchestration
- `UrlProcessor.php`: URL manipulation and validation
- `Normalize.php`: URL normalization
- `Extract.php`: Extract assets from content

### Data & Storage
- `Database.php`: Database operations
- `Cache.php`: Caching layer
- `FileLock.php`: File locking mechanism

### Background Tasks
- `Tasks.php`: Task queue management
- `Retry.php`: Retry failed operations
- `Cleanup.php`: Cleanup old assets

### Utilities
- `Logging.php`: Logging functionality
- `Sanitize.php`: Input sanitization
- `GetOption.php`: Options management
- `GetData.php`: Data retrieval
- `Settings.php`: Settings management
- `Render.php`: Template rendering
- `Replace.php`: URL replacement
- `Generate.php`: Asset generation
- `Enqueue.php`: Asset enqueueing
- `Speculation.php`: Speculative loading
- `SVG.php`: SVG handling
- `Ajax.php`: AJAX handlers

## Documentation Files

Multiple markdown files document the dependency injection refactoring process:
- `DI-*.md`: Various DI implementation documentation
- `CLASSES-TO-UPDATE.md`: Classes requiring updates
- `todo*.txt`: Task tracking files
- `refactoring-progress.txt`: Refactoring status

## Refactoring Status by Component

### ✅ Completed
- `ServiceContainer.php`: DI container implementation
- `Initialize.php`: Uses constructor injection for Logger and Database
- `Database.php`: Accepts wpdb and LoggerInterface
- `Tasks.php`: Refactored with LoggerInterface, DatabaseInterface, wpdb
- `SelfHost.php`: Updated with DatabaseInterface
- All interfaces defined in `/interfaces`

### 🔄 In Progress / Needs Update
Most classes still need to be migrated to use dependency injection. When working on any class:

1. Check if it has a corresponding interface
2. Update constructor to accept dependencies
3. Remove static method calls to other LHA classes
4. Add service definition to `src/services.php`
5. Update all callers to use container resolution

### ⚠️ CRITICAL REFACTORING RULES

**NEVER OVERWRITE EXISTING FILES**
- Always edit files in place using `strReplace` or similar editing tools
- Never create a temporary/refactored version and move it over the original
- This preserves file history and prevents accidental data loss
- Make incremental changes, not wholesale replacements

**NO STUBS OR PLACEHOLDERS ALLOWED**
- All implementations must be production-ready and fully functional
- Never use placeholder comments like "// TODO: Implement this"
- Never use stub methods that throw exceptions or return null
- Never leave incomplete implementations
- Every method must have complete, working logic
- If a feature cannot be fully implemented, do not create the method
- Partial implementations are strictly forbidden

**NO PREMATURE COMPLETION CLAIMS**
- NEVER declare a refactoring "complete" or "final" without verifying ALL work is done
- NEVER use "100%" or "finished" unless you have checked the entire scope
- Always provide honest, accurate progress assessments
- Check for remaining static methods, unupdated callers, missing tests, etc.
- Use document names like "Progress-Report" or "Status-Update", not "Final" or "Complete"
- If you discover more work exists, immediately update your assessment

**NEVER SKIP WORK OR TAKE SHORTCUTS**
- NEVER bypass work due to "time constraints" or similar excuses
- NEVER suggest workarounds or temporary solutions instead of proper fixes
- NEVER comment out code to avoid fixing issues
- NEVER suggest "we can fix this later" - fix it now
- ALL work must be completed properly and thoroughly
- If something is broken, fix it completely before moving on
- There are no acceptable reasons to skip proper implementation
- Quality and completeness are non-negotiable

### Key Files for DI System
- `ServiceContainer.php`: The DI container
- `src/services.php`: Service definitions and bindings
- `locally-host-assets.php`: Container initialization
- `/interfaces/*.php`: All interface definitions
