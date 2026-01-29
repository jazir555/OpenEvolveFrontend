# TaskHelpers WordPress/ActionScheduler Compatibility Report

**Date:** 2025-12-30
**Files Reviewed:** 13 TaskHelpers files
**Files Needing Fixes:** 9 out of 13

## Executive Summary

After comprehensive review of all 13 TaskHelpers files, **9 files require WordPress/ActionScheduler function guards** to ensure compatibility when running outside WordPress context. The helper files are calling WordPress functions directly without checking if they exist first.

## Files by Status

### ✅ FILES REQUIRING NO FIXES (4 files)

These files contain no direct WordPress/ActionScheduler function calls:

1. **TaskUtilityHelper.php** - CLEAN
   - No WordPress functions found
   - Only uses PHP native functions and class methods

2. **TaskSchedulerHelper.php** - CLEAN
   - No WordPress functions found
   - Only uses dependency injection and service container

3. **TasksStaticHelper.php** - CLEAN
   - Placeholder class with no implementation

4. **TasksHelper.php** - CLEAN
   - Only uses container and dependency injection

---

### ❌ FILES REQUIRING FIXES (9 files)

These files contain direct WordPress function calls that need guards:

#### 1. TaskCacheHelper.php (CRITICAL)

**Issues Found:**
- Line 22: `get_transient()` - needs guard
- Line 34: `set_transient()` - needs guard
- Line 45: `delete_transient()` - needs guard
- Line 148: `wp_cache_delete()` - needs guard

**Functions to guard:**
```php
get_transient(), set_transient(), delete_transient(), wp_cache_delete()
```

**Risk Level:** HIGH - Cache operations will fail outside WordPress

---

#### 2. TaskValidationHelper.php (HIGH)

**Issues Found:**
- Line 39: `esc_html()` - needs guard
- Line 57: `wp_cache_get()` - needs guard
- Line 73: `esc_sql()` - needs guard
- Line 97: `wp_cache_set()` - needs guard

**Functions to guard:**
```php
esc_html(), wp_cache_get(), esc_sql(), wp_cache_set()
```

**Risk Level:** HIGH - Security functions need fallbacks

---

#### 3. TaskStatusHelper.php (CRITICAL)

**Issues Found:**
- Lines 36, 58, 79-84, 97, 108, 192, 214-216, 257, 290-296: Multiple WordPress cache and time functions
- Lines 336, 354, 382-386: Transient, filter, and translation functions

**Functions to guard:**
```php
esc_sql(), current_time(), wp_cache_delete(), get_transient(),
apply_filters(), __()
```

**Risk Level:** CRITICAL - Core task status management depends on these

---

#### 4. TaskQueryHelper.php (CRITICAL)

**Issues Found:**
- Lines 51, 98, 157, 222: `wp_cache_get()`
- Lines 59, 76, 110, 179, 235, 254: `esc_sql()`
- Lines 126, 129, 194, 204, 254: `wp_cache_set()`

**Functions to guard:**
```php
wp_cache_get(), esc_sql(), wp_cache_set()
```

**Risk Level:** CRITICAL - All database queries depend on these

---

#### 5. TaskCronHelper.php (CRITICAL - 50+ instances)

**Issues Found:**
This file has the MOST WordPress dependencies:

**Cron Functions:**
- `wp_next_scheduled()` - lines 26, 36, 80, 136, 190, 427
- `wp_schedule_event()` - lines 51, 195, 464, 556
- `wp_schedule_single_event()` - line 427
- `wp_clear_scheduled_hook()` - lines 27, 29, 137, 515, 541, 625, 644, 744
- `_get_cron_array()` - line 20

**Cache Functions:**
- `wp_cache_add()` - lines 232, 240, 33
- `wp_cache_get()` - line 236
- `wp_cache_delete()` - lines 239, 263

**Filter/Action Functions:**
- `add_filter()` - lines 75, 218
- `add_action()` - lines 79, 89, 93, 98
- `remove_filter()` - line 415
- `apply_filters()` - line 224
- `has_action()` - line 462

**Other WordPress Functions:**
- `wp_get_schedules()` - lines 41, 174, 494
- `is_multisite()` - line 604
- `get_sites()` - line 608
- `get_current_blog_id()` - line 612
- `switch_to_blog()` - line 619
- `restore_current_blog()` - line 636
- `get_date_from_gmt()` - line 572
- `wp_date()` - lines 468, 478

**Constants:**
- `HOUR_IN_SECONDS` - line 331
- `MINUTE_IN_SECONDS` - line 553, 708

**Functions to guard:**
```php
wp_next_scheduled(), wp_schedule_event(), wp_schedule_single_event(),
wp_clear_scheduled_hook(), _get_cron_array(), wp_cache_add(),
wp_cache_get(), wp_cache_delete(), add_filter(), add_action(),
remove_filter(), apply_filters(), has_action(), wp_get_schedules(),
is_multisite(), get_sites(), get_current_blog_id(), switch_to_blog(),
restore_current_blog(), get_date_from_gmt(), wp_date(), gmdate(),
esc_sql(), current_time(), __()
```

**Risk Level:** CRITICAL - All cron functionality will break

---

#### 6. TaskEnqueueHelper.php (CRITICAL - 60+ instances)

**Issues Found:**
This file has extensive WordPress dependencies for task enqueueing:

**Cache Functions:**
- `wp_cache_delete()` - lines 175-176, 503, 1169, 1171, 1446-1447
- `wp_cache_get()` - (implied in various operations)

**Cron Functions:**
- `wp_next_scheduled()` - lines 284, 432, 907, 918
- `wp_schedule_event()` - line 305
- `wp_schedule_single_event()` - lines 443, 907, 918, 427
- `wp_unschedule_event()` - line 437
- `wp_clear_scheduled_hook()` - line 469
- `has_action()` - line 462
- `add_action()` - line 463
- `add_filter()` - line 293

**Other WordPress Functions:**
- `esc_html()` - lines 37, 224, 333, 366-370, 467, 484, 509, 528, 558, 594, 626, 681, 909, 920
- `esc_sql()` - lines 777, 786, 828, 861, 961, 1088, 1142, 1339, 1415
- `current_time()` - lines 131, 777, 836, 1108
- `esc_url()` - lines 648, 664, 685, 874
- `delete_option()` - lines 472, 473
- `wp_get_schedules()` - lines 289, 295

**Constants:**
- `MINUTE_IN_SECONDS` - line 609

**Functions to guard:**
```php
wp_cache_delete(), wp_next_scheduled(), wp_schedule_event(),
wp_schedule_single_event(), wp_unschedule_event(), wp_clear_scheduled_hook(),
has_action(), add_action(), add_filter(), esc_html(), esc_sql(),
current_time(), esc_url(), delete_option(), wp_get_schedules(), __()
```

**Risk Level:** CRITICAL - Task enqueueing is core functionality

---

#### 7. TaskMaintenanceHelper.php (HIGH)

**Issues Found:**
- Line 20: `_get_cron_array()` - needs guard
- Line 34: `wp_clear_scheduled_hook()` - needs guard
- Lines 70, 161: `esc_sql()` - needs guard
- Lines 97-98, 246-247: `wp_cache_delete()` - needs guard
- Lines 206, 207: `gmdate()`, `current_time()` - needs guard
- Line 210: `esc_sql()` - needs guard

**Functions to guard:**
```php
_get_cron_array(), wp_clear_scheduled_hook(), esc_sql(),
wp_cache_delete(), gmdate(), current_time()
```

**Risk Level:** HIGH - Maintenance tasks will fail

---

#### 8. TaskProcessingHelper.php (HIGH)

**Issues Found:**
- Lines 133, 196: `remove_filter()` - needs guard
- Lines 139, 420: `apply_filters()` - needs guard
- Line 211: `wp_cache_get()` - needs guard
- Line 241: `wp_cache_set()` - needs guard
- Lines 220, 666, 704: `esc_sql()` - needs guard
- Lines 332, 706: `gmdate()` - needs guard
- Line 623: `wp_clear_scheduled_hook()` - needs guard

**Functions to guard:**
```php
remove_filter(), apply_filters(), wp_cache_get(), wp_cache_set(),
esc_sql(), gmdate(), wp_clear_scheduled_hook()
```

**Risk Level:** HIGH - Task processing will fail

---

#### 9. TaskScheduleHelper.php (HIGH)

**Issues Found:**
- Line 22: `HOUR_IN_SECONDS` - needs constant guard
- Line 33: `wp_cache_add()` - needs guard
- Lines 50, 56, 62: `get_option()`, `update_option()` - needs guard
- Line 100: `do_action()` - needs guard
- Line 109: `wp_cache_delete()` - needs guard
- Lines 183, 206, 211, 230: `update_option()`, `delete_option()` - needs guard
- Line 259: `gmdate()` - needs guard

**Functions to guard:**
```php
wp_cache_add(), get_option(), update_option(), do_action(),
wp_cache_delete(), delete_option(), gmdate()
```

**Constants to guard:**
```php
HOUR_IN_SECONDS
```

**Risk Level:** HIGH - Task scheduling will fail

---

## Recommended Solution

### Option 1: Load WordPressCompatibility.php (RECOMMENDED)

Since `WordPressCompatibility.php` already exists with fallback implementations, the best solution is to ensure it's loaded before any helper classes.

**Add to top of each helper file:**
```php
// Load WordPress compatibility shims if running outside WordPress
if (!defined('ABSPATH') && file_exists(__DIR__ . '/WordPressCompatibility.php')) {
    require_once __DIR__ . '/WordPressCompatibility.php';
}
```

### Option 2: Add function_exists() Guards (ALTERNATIVE)

For each WordPress function call, add a guard:

```php
// Example for wp_cache_get
if (function_exists('wp_cache_get')) {
    $cached = wp_cache_get($cache_key, self::CACHE_GROUP);
} else {
    $cached = false;
}

// Example for esc_sql
if (function_exists('esc_sql')) {
    $escaped_table = esc_sql($tasks_table);
} else {
    $escaped_table = addslashes($tasks_table);
}

// Example for wp_schedule_event
if (function_exists('wp_schedule_event')) {
    $scheduled = wp_schedule_event($timestamp, $recurrence, $hook);
} else {
    if (class_exists('\LHA\Logging')) {
        Logging::log_warning('wp_schedule_event not available in non-WP environment');
    }
    $scheduled = false;
}
```

---

## Constants Requiring Guards

These WordPress constants are used and need fallbacks:

```php
HOUR_IN_SECONDS    // 3600
MINUTE_IN_SECONDS  // 60
DAY_IN_SECONDS     // 86400
```

**Add to WordPressCompatibility.php:**
```php
if (!defined('HOUR_IN_SECONDS')) {
    define('HOUR_IN_SECONDS', 3600);
}
if (!defined('MINUTE_IN_SECONDS')) {
    define('MINUTE_IN_SECONDS', 60);
}
if (!defined('DAY_IN_SECONDS')) {
    define('DAY_IN_SECONDS', 86400);
}
```

---

## Implementation Priority

### Phase 1: CRITICAL (Do First)
1. TaskCronHelper.php - 50+ WordPress function calls
2. TaskEnqueueHelper.php - 60+ WordPress function calls
3. TaskStatusHelper.php - Core status management
4. TaskQueryHelper.php - All database queries

### Phase 2: HIGH (Do Second)
5. TaskProcessingHelper.php - Task processing logic
6. TaskCacheHelper.php - Cache operations
7. TaskScheduleHelper.php - Task scheduling
8. TaskValidationHelper.php - Input validation

### Phase 3: MEDIUM (Do Third)
9. TaskMaintenanceHelper.php - Maintenance tasks

---

## Validation Commands

After fixing each file, validate syntax:

```bash
php -l C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\TaskHelpers\TaskCronHelper.php
php -l C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\TaskHelpers\TaskEnqueueHelper.php
# ... etc for all files
```

---

## Testing Strategy

1. **Standalone Test:** Run helper methods in standalone PHP (no WordPress)
2. **WordPress Test:** Run helper methods within WordPress context
3. **Docker Test:** Run in Docker container with/without WordPress

---

## Summary Statistics

- **Total Files:** 13
- **Files Requiring Fixes:** 9 (69%)
- **Clean Files:** 4 (31%)
- **Estimated Function Calls to Guard:** 200+
- **Critical Files:** 5
- **High Priority Files:** 4

---

## Conclusion

The TaskHelpers directory has significant WordPress dependencies that need to be addressed for standalone compatibility. The **recommended approach** is to ensure `WordPressCompatibility.php` is loaded in all helper files, as it already provides fallback implementations for most required functions.

The alternative approach of adding `function_exists()` guards to every WordPress function call would require modifying **200+ function calls** across **9 files**, which is time-consuming and increases code complexity.

**Best Path Forward:**
1. Add WordPressCompatibility.php require statement to all 9 helper files
2. Add missing constants to WordPressCompatibility.php
3. Add any missing function fallbacks to WordPressCompatibility.php
4. Test each helper in standalone mode

This approach centralizes compatibility logic in one file and reduces maintenance burden.
