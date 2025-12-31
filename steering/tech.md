# Technical Stack

## Platform

- **WordPress**: 5.5+ required
- **PHP**: 7.4+ required (strict types enabled)
- **Database**: WordPress wpdb (MySQL/MariaDB)

## Architecture

### OpenEvolve Frontend

**Current Focus**: Migrating entire codebase to use dependency injection pattern.

- **ServiceContainer**: Custom DI container in `ServiceContainer.php`
  - Singleton pattern for resolved services
  - Supports closures for lazy instantiation
  - Interface-to-implementation mapping
  - Service definitions in `src/services.php`
  
- **Interfaces**: All major components have interfaces in `/interfaces` directory
  - Pattern: `{ClassName}Interface.php`
  - Namespace: `LHA\Interfaces\`
  - Used for type hints in constructors

- **Constructor Injection**: Classes receive dependencies via `__construct()`
  - Type hint against interfaces, not concrete classes
  - Example: `public function __construct(LoggerInterface $logger, DatabaseInterface $database)`
  
- **Service Resolution**: Container automatically resolves dependencies
  - `$container->get(ClassName::class)` returns instance
  - Interfaces automatically map to implementations
  - Services are singletons by default

### ⚠️ CRITICAL: Static Proxy Pattern - DO NOT REMOVE

**ABSOLUTE RULE**: Classes that need backward compatibility MUST have static proxy methods using `__callStatic`

- **Instance Methods**: Public methods that do the actual work (for DI)
- **Static Proxy**: `__callStatic` magic method that delegates to instance via container
- **NEVER remove static proxy methods** - they provide backward compatibility
- **NEVER remove instance methods** - they provide DI functionality
- **Use `__callStatic`** - PHP doesn't allow both static and instance methods with the same name

Example pattern:
```php
class MyClass {
    // Instance method (for DI)
    public function doSomething() { /* implementation */ }
    
    // Static proxy (for backward compatibility)
    private static $staticInstance = null;
    
    private static function getStaticInstance(): MyClassInterface {
        if (self::$staticInstance === null) {
            global $lha_container;
            self::$staticInstance = $lha_container->get(MyClassInterface::class);
        }
        return self::$staticInstance;
    }
    
    public static function __callStatic(string $method, array $arguments) {
        return self::getStaticInstance()->$method(...$arguments);
    }
}
```

### General Architecture

- **Namespace**: `LHA\` for all classes
- **Autoloading**: PSR-4 compatible (Composer or fallback autoloader)

## Key Libraries & Dependencies

- WordPress Core APIs (WP_Filesystem, WP_Scripts, WP_Styles, wpdb)
- Action Scheduler (optional, for background tasks)
- PHP DOM extension (for HTML parsing)
- PHP libxml (for XML/HTML processing)

## Constants

- `LHA_VERSION`: Plugin version
- `LHA_PLUGIN_FILE`: Main plugin file path
- `LHA_PLUGIN_PATH`: Plugin directory path
- `LHA_PLUGIN_URL`: Plugin URL
- `LHA_CRON_HOOK_PREFIX`: Prefix for cron hooks
- `CRON_HOOK_RETRY_DATABASE`: Retry cron hook name

## Common Commands

### Development

```bash
# No build process - direct PHP execution
# Plugin uses WordPress autoloader or Composer autoloader
# Composer is available for dependency management
composer install
composer update
```

### Testing

```bash
# Testing is done via WordPress installation in Docker container "lha-wordpress"
# Container provides full WordPress environment with plugin loaded

# Access container shell
docker exec -it lha-wordpress bash

# Run Composer commands in container
docker exec lha-wordpress composer install
docker exec lha-wordpress composer test  # if test scripts configured

# Add test file to plugin root and access via WordPress admin
# Example: test-di.php, test-dependency-injection.php
# Access via browser: http://localhost/wp-admin/ or site URL

# Run PHP scripts directly in container
docker exec lha-wordpress php /path/to/test-script.php

# Activate plugin for testing
docker exec lha-wordpress php /var/www/html/wp-content/plugins/locally-host-assets/activate-plugin.php
```

### ⚠️ CRITICAL DOCKER VOLUME MOUNT WARNING ⚠️

**NEVER RUN `rm -rf` OR DELETE COMMANDS IN THE CONTAINER**

The Docker container has a **bind mount** that directly links the local workspace to the container:
- Local: `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes`
- Container: `/var/www/html/wp-content/plugins/locally-host-assets`

**These are the SAME files, not copies!**

**FORBIDDEN COMMANDS:**
```bash
# NEVER run these - they will permanently delete your local files:
docker exec lha-wordpress sh -c "rm -rf /var/www/html/wp-content/plugins/locally-host-assets/*"
docker exec lha-wordpress sh -c "rm /var/www/html/wp-content/plugins/locally-host-assets/*.php"
# ANY delete command in the container = deleting your local work files
```

**SAFE PRACTICES:**
- Always check volume mounts before running any delete commands: `docker inspect lha-wordpress | Select-String -Pattern "Mounts" -Context 0,20`
- To clear files, delete them from your LOCAL workspace, not from inside the container
- Files deleted via bind mounts do NOT go to Recycle Bin and are NOT recoverable
- Always commit changes to git before running container commands
- Use `docker cp` to copy files TO the container, never delete first

### Diagnostics

```bash
# Access with query parameter: ?lha_diagnose=1
# Requires WP_DEBUG to be enabled
# Shows DI container status and service registrations

# Check container logs
docker logs lha-wordpress

# Access WordPress debug.log
docker exec lha-wordpress tail -f /var/www/html/wp-content/debug.log
```

## Dependency Injection Refactoring Guidelines

### CRITICAL: File Modification Rules

**NEVER OVERWRITE EXISTING FILES**
- You MUST edit existing files in place using `strReplace` or similar tools
- NEVER create a new version of a file and overwrite the original
- NEVER use file move/copy operations to replace existing code files
- Always make incremental edits to preserve file history and avoid data loss

**NEVER DELETE FILES IN DOCKER CONTAINERS WITH BIND MOUNTS**
- ALWAYS check for bind mounts before running ANY delete commands in containers
- Run `docker inspect <container> | Select-String -Pattern "Mounts"` first
- If bind mounts exist, deleting files in the container PERMANENTLY deletes local files
- Files deleted via bind mounts bypass the Recycle Bin and are NOT recoverable
- NEVER run `rm -rf`, `rm`, `del`, or any delete command inside containers with bind mounts

### CRITICAL: Implementation Quality Rules

**NO STUBS OR PLACEHOLDERS ALLOWED - ZERO TOLERANCE**
- ALL code must be production-ready and fully functional - NO EXCEPTIONS
- NEVER use placeholder comments like "// TODO: Implement this"
- NEVER use stub methods that throw exceptions or return null
- NEVER leave incomplete implementations with comments to "add later"
- NEVER create methods that just call other methods without adding logic
- NEVER create empty methods or methods that just return default values
- Every method must have complete, working logic that actually does what it's supposed to do
- If a feature cannot be fully implemented, do not create the method at all
- Partial implementations are strictly forbidden
- All error handling must be complete and functional
- All validation logic must be fully implemented
- If you create a stub, you MUST immediately go back and fully implement it
- Stubs are considered incomplete work and are NOT acceptable under any circumstances

### CRITICAL: Work Completion Rules

**NO SKIPPING OR AVOIDING WORK - ABSOLUTE RULE**
- NEVER skip refactoring work because it seems "difficult" or "complex"
- NEVER avoid work because interfaces seem "over-engineered"
- NEVER make excuses about code quality or design decisions
- NEVER make excuses about time, session length, or complexity
- NEVER say you've been "working on X for too long" - that's an excuse
- NEVER say something is "taking too long" - that's an excuse
- NEVER suggest the work is too difficult or tedious - that's an excuse
- NEVER ask the user if they want to "take a different approach" - just fix it
- NEVER suggest stopping or pausing - continue until the work is done
- NEVER give up on a file or task - fix it completely
- ALWAYS complete the refactoring task at hand no matter how long it takes
- If an interface has methods the class doesn't implement, implement them or delegate to existing static methods
- If a class is large, break the work into smaller chunks but complete it
- Document challenges but DO NOT use them as reasons to skip work
- Your job is to refactor, not to judge whether refactoring is worthwhile
- When you encounter errors, fix them systematically until resolved
- There is NO time limit - work until the task is complete

**NO PREMATURE COMPLETION DECLARATIONS**
- NEVER use words like "final", "complete", "done", or "100%" in document titles unless verified
- NEVER declare completion without checking ALL remaining work
- Always verify the full scope before declaring anything finished
- Use accurate progress percentages based on actual work completed vs remaining
- Document names should reflect current state: "Progress", "Status", "Current", "Session-N"
- Before declaring completion, explicitly list and verify all remaining tasks
- If you discover more work after declaring completion, immediately correct the assessment

### When Updating a Class

1. **Add Interface**: Create corresponding interface in `/interfaces` if not exists
2. **Update Constructor**: Accept dependencies as typed parameters
3. **Keep Static Proxy Methods**: NEVER remove static proxy methods - add `__callStatic` if needed
4. **Add Instance Methods**: Create public instance methods for DI (these coexist with static proxies)
5. **Update Service Definitions**: Add to `src/services.php` with dependencies
6. **Register in Container**: Ensure `ServiceContainer::registerServices()` includes the class
7. **Edit In Place**: Use `strReplace` to modify existing files, never create new versions

### ⚠️ CRITICAL RULE: NEVER REMOVE STATIC PROXY METHODS

**THIS IS AN ABSOLUTE, NON-NEGOTIABLE RULE**

- Static proxy methods provide backward compatibility for existing code
- Instance methods provide DI functionality for new code
- BOTH must coexist in the same class
- Use `__callStatic` magic method to avoid PHP's "cannot redeclare" error
- If you even THINK about removing static proxies, STOP and re-read this section
- Removing static proxies breaks backward compatibility and is FORBIDDEN

### Pattern to Follow

```php
// Before (static, no DI)
class MyClass {
    public static function doSomething() {
        $logger = new Logging();
        $logger->log_info('message');
    }
}

// After (instance, with DI)
class MyClass implements MyClassInterface {
    private LoggerInterface $logger;
    
    public function __construct(LoggerInterface $logger) {
        $this->logger = $logger;
    }
    
    public function doSomething(): void {
        $this->logger->log_info('message');
    }
}
```

### Common Dependencies

- `LoggerInterface` → `Logging` class
- `DatabaseInterface` → `Database` class
- `DownloaderInterface` → `SelfHost` class
- `SanitizerInterface` → `Sanitize` class
- `TaskQueueInterface` → `Tasks` class
- `OptionsInterface` → `GetOption` class
- `LockInterface` → `FileLock` class

## Database Tables

- `{prefix}lha_asset_mapping`: Asset URL mappings and status
- `{prefix}lha_tasks`: Background task queue


**STATIC PROXY PATTERN - CRITICAL UNDERSTANDING**
- The DI refactoring uses a STATIC PROXY PATTERN for backward compatibility
- Static methods MUST remain static - they are NOT converted to instance methods
- Static methods call through to the container to get the instance, then call the instance method
- Pattern: `public static function method() { $instance = Container::get(Interface::class); return $instance->method(); }`
- NEVER change existing static method calls throughout the codebase to instance calls
- NEVER remove static methods - they are essential for backward compatibility
- The class has BOTH instance methods (for DI) AND static methods (for backward compatibility)
- Static methods are proxies that delegate to instance methods via the container
- This allows gradual migration without breaking existing code
- Existing code continues to call `ClassName::staticMethod()`
- New code can use DI and call `$instance->method()`
- Both approaches work simultaneously during the transition


**CRITICAL: AVOID DESTRUCTIVE REGEX REPLACEMENTS**
- NEVER use PowerShell or bash regex replacements on entire files
- NEVER use commands like `(Get-Content file -Raw) -replace 'pattern', 'replacement' | Set-Content file`
- NEVER use sed, awk, or similar tools for bulk replacements across entire files
- These operations are EXTREMELY error-prone and cause cascading syntax errors
- Regex replacements cannot understand code context and will break things
- ALWAYS use strReplace tool with specific, targeted oldStr/newStr pairs
- ALWAYS verify each change is correct before moving to the next
- If you need to make many similar changes, do them ONE AT A TIME with strReplace
- Each strReplace should be a small, verifiable change
- NEVER try to "save time" with bulk regex operations
- The time "saved" is lost 10x over fixing the resulting breakage

**CRITICAL: UNDERSTAND THE ARCHITECTURE BEFORE CHANGING IT**
- NEVER start refactoring a class without understanding how it's used
- ALWAYS check if a class is meant to be static-only or instantiable
- ALWAYS check how many places call a class before changing its interface
- If a class has static methods called throughout the codebase, DO NOT convert them to instance methods
- If a class is called statically in 100+ places, it's meant to be static
- NEVER assume you can just "convert everything to DI" without understanding the design
- Some classes (like Logging utilities) are intentionally static for simplicity
- Not every class needs to be refactored for DI
- Static utility classes are VALID and should remain static
- Before changing a class, grep for its usage patterns across the codebase
- If you see `ClassName::method()` everywhere, that class should stay static

**CRITICAL: WHEN YOU'RE CONFUSED, STOP AND ASK**
- If you don't understand why something is designed a certain way, STOP
- If you're not sure whether to make something static or instance, STOP
- If you're about to make a change that affects 100+ files, STOP
- NEVER proceed with major refactoring when you're uncertain
- NEVER make assumptions about architecture
- ALWAYS ask the user for clarification when uncertain
- It's better to ask a "stupid question" than to break the entire codebase
- Admitting confusion is professional, not weak
- The user would rather answer questions than fix your mistakes


**CRITICAL: NOT EVERY CLASS NEEDS DI**
- Some classes are intentionally static utility classes (like Logging)
- If a class is called statically in 100+ places, it should STAY static
- Do NOT try to force DI onto static utility classes
- Static utility classes are a valid design pattern
- Logging, in particular, should remain a static utility class
- Classes that need to log can call `\LHA\Logging::log_error()` statically
- This is simpler and more appropriate than trying to inject a logger everywhere
- Accept that not every class fits the DI pattern
- The goal is better architecture, not dogmatic adherence to DI everywhere


**CRITICAL: LOGGING DOES NOT NEED DI**
- Logging is a cross-cutting concern used everywhere in the application
- Logging is a perfect candidate for a static utility class
- DO NOT try to integrate Logging with the DI container
- DO NOT try to inject loggers into every class
- Logging doesn't need to be mocked in tests - it can just log
- There's no reason to swap logging implementations
- Static utility classes are VALID and APPROPRIATE for cross-cutting concerns
- Classes should call `\LHA\Logging::log_error()` directly
- This is simpler, cleaner, and more maintainable than DI for logging
- Not every class needs DI - use the right tool for the job
- Cross-cutting concerns like logging, caching, and configuration are often better as static utilities
