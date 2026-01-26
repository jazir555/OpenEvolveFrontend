<?php
namespace LHA;

use LHA\Interfaces\ActionSchedulerHelperInterface;

/**
 * Helper class for Action Scheduler integration.
 * Provides utility methods for checking status, migrating, and admin UI.
 */
class ActionSchedulerHelper implements ActionSchedulerHelperInterface {
    
    /** @var string Path to Action Scheduler in vendor */
    private const AS_VENDOR_PATH = '/vendor/woocommerce/action-scheduler/action-scheduler.php';
    
    /**
     * Check if Action Scheduler is available.
     *
     * @return bool True if available.
     */
    public static function is_available(): bool {
        // Check if AS functions exist (loaded by another plugin like WooCommerce)
        if ( function_exists( 'as_schedule_single_action' ) ) {
            return true;
        }
        
        // Check if we can load from vendor
        // Use LHA_PLUGIN_PATH constant if available, otherwise fallback to dirname(__FILE__)
        $plugin_path = defined( 'LHA_PLUGIN_PATH' ) ? LHA_PLUGIN_PATH : dirname( __FILE__ ) . '/';
        $vendor_path = rtrim( $plugin_path, '/\\' ) . self::AS_VENDOR_PATH;
        return file_exists( $vendor_path );
    }
    
    /**
     * Load Action Scheduler from vendor if not already loaded.
     *
     * Note: If another plugin (like WooCommerce) has already loaded Action Scheduler,
     * this method will return true without loading our vendor version. Action Scheduler
     * is designed to handle version conflicts by using the latest version found, but
     * the actual version in use may differ from what's in our vendor directory.
     *
     * @return bool True if loaded successfully or already available.
     */
    public static function load(): bool {
        // If Action Scheduler is already loaded by another plugin, return true
        // Action Scheduler handles version conflicts internally by using the latest version
        if ( function_exists( 'as_schedule_single_action' ) ) {
            return true;
        }
        
        // Use LHA_PLUGIN_PATH constant if available, otherwise fallback to dirname(__FILE__)
        $plugin_path = defined( 'LHA_PLUGIN_PATH' ) ? LHA_PLUGIN_PATH : dirname( __FILE__ ) . '/';
        $vendor_path = rtrim( $plugin_path, '/\\' ) . self::AS_VENDOR_PATH;
        if ( file_exists( $vendor_path ) ) {
            require_once $vendor_path;
            return function_exists( 'as_schedule_single_action' );
        }
        
        return false;
    }
    
    /**
     * Get Action Scheduler version.
     *
     * @return string|null Version string or null if not available.
     */
    public static function get_version(): string {
        if ( ! self::load() ) {
            return null;
        }
        
        // Try to get version from ActionScheduler_Versions class constant
        if ( class_exists( 'ActionScheduler_Versions' ) && defined( 'ActionScheduler_Versions::VERSION' ) ) {
            return \ActionScheduler_Versions::VERSION;
        }
        
        return 'unknown';
    }
    
    /**
     * Get Action Scheduler status for admin display.
     *
     * @return array Status information.
     */
    public static function get_status(): array {
        $status = [
            'available'     => self::is_available(),
            'loaded'        => function_exists( 'as_schedule_single_action' ),
            'version'       => self::get_version(),
            'source'        => self::get_source(),
            'store_class'   => null,
            'runner_class'  => null,
        ];
        
        if ( $status['loaded'] && class_exists( 'ActionScheduler' ) ) {
            try {
                $status['store_class'] = get_class( \ActionScheduler::store() );
                $status['runner_class'] = get_class( \ActionScheduler::runner() );
            } catch ( \Exception $e ) {
                // Ignore errors
            }
        }
        
        return $status;
    }
    
    /**
     * Get the source of Action Scheduler.
     *
     * @return string Source description.
     */
    public static function get_source(): string {
        if ( ! function_exists( 'as_schedule_single_action' ) ) {
            // Use LHA_PLUGIN_PATH constant if available, otherwise fallback to dirname(__FILE__)
            $plugin_path = defined( 'LHA_PLUGIN_PATH' ) ? LHA_PLUGIN_PATH : dirname( __FILE__ ) . '/';
            $vendor_path = rtrim( $plugin_path, '/\\' ) . self::AS_VENDOR_PATH;
            if ( file_exists( $vendor_path ) ) {
                return 'LHA Vendor (not loaded)';
            }
            return 'Not available';
        }
        
        // Use LHA_PLUGIN_PATH constant if available, otherwise fallback to dirname(__FILE__)
        $plugin_path = defined( 'LHA_PLUGIN_PATH' ) ? LHA_PLUGIN_PATH : dirname( __FILE__ ) . '/';
        $vendor_path = rtrim( $plugin_path, '/\\' ) . self::AS_VENDOR_PATH;
        
        if ( file_exists( $vendor_path ) ) {
            // Check if the loaded AS is from our vendor
            $as_file = null;
            if ( class_exists( 'ActionScheduler' ) ) {
                $reflection = new \ReflectionClass( 'ActionScheduler' );
                $as_file = $reflection->getFileName();
            }
            
            // Normalize paths for comparison (handle both Windows and Unix paths)
            if ( $as_file ) {
                // Normalize paths - use wp_normalize_path if available, otherwise use str_replace
                // wp_normalize_path handles trailing slashes and double slashes automatically
                if ( function_exists( 'wp_normalize_path' ) ) {
                    $normalized_as_file = wp_normalize_path( $as_file );
                    $normalized_plugin_path = wp_normalize_path( $plugin_path );
                } else {
                    $normalized_as_file = str_replace( '\\', '/', $as_file );
                    $normalized_plugin_path = rtrim( str_replace( '\\', '/', $plugin_path ), '/' );
                }
                
                // Check if AS file is within our plugin directory or vendor directory
                // Note: If multiple plugins bundle Action Scheduler, only the "freshest" version loads.
                // This check correctly identifies which plugin's version is actually in use.
                if ( strpos( $normalized_as_file, $normalized_plugin_path ) !== false ) {
                    return 'LHA Vendor';
                }
            }
        }
        
        // Check common sources
        if ( class_exists( 'WooCommerce' ) ) {
            return 'WooCommerce';
        }
        
        if ( defined( 'ACTION_SCHEDULER_PLUGIN_FILE' ) ) {
            return 'Standalone Plugin';
        }
        
        return 'External (unknown)';
    }
    
    /**
     * Get LHA task statistics from Action Scheduler.
     *
     * @return array Task statistics.
     */
    public static function get_lha_stats(): array {
        if ( ! self::load() || ! class_exists( 'ActionScheduler' ) ) {
            return [
                'available' => false,
            ];
        }
        
        try {
            $store = \ActionScheduler::store();
            $group = 'lha_tasks';
            
            return [
                'available' => true,
                'pending'   => $store->query_actions( [
                    'status'   => \ActionScheduler_Store::STATUS_PENDING,
                    'group'    => $group,
                    'per_page' => 1,
                ], 'count' ),
                'running'   => $store->query_actions( [
                    'status'   => \ActionScheduler_Store::STATUS_RUNNING,
                    'group'    => $group,
                    'per_page' => 1,
                ], 'count' ),
                'complete'  => $store->query_actions( [
                    'status'   => \ActionScheduler_Store::STATUS_COMPLETE,
                    'group'    => $group,
                    'per_page' => 1,
                ], 'count' ),
                'failed'    => $store->query_actions( [
                    'status'   => \ActionScheduler_Store::STATUS_FAILED,
                    'group'    => $group,
                    'per_page' => 1,
                ], 'count' ),
                'canceled'  => $store->query_actions( [
                    'status'   => \ActionScheduler_Store::STATUS_CANCELED,
                    'group'    => $group,
                    'per_page' => 1,
                ], 'count' ),
            ];
            
        } catch ( \Exception $e ) {
            return [
                'available' => true,
                'error'     => $e->getMessage(),
            ];
        }
    }
    
    /**
     * Get recent LHA actions from Action Scheduler.
     *
     * @param int    $limit  Number of actions to retrieve.
     * @param string $status Status filter (or 'all').
     * @return array Recent actions.
     */
    public static function get_recent_actions( int $limit = 20, string $status = 'all' ): array {
        if ( ! self::load() || ! class_exists( 'ActionScheduler' ) ) {
            return [];
        }
        
        try {
            $store = \ActionScheduler::store();
            $group = 'lha_tasks';
            
            $args = [
                'group'    => $group,
                'per_page' => $limit,
                'orderby'  => 'scheduled_date_gmt',
                'order'    => 'DESC',
            ];
            
            if ( $status !== 'all' ) {
                $args['status'] = $status;
            }
            
            $action_ids = $store->query_actions( $args, 'ids' );
            $actions = [];
            
            foreach ( $action_ids as $action_id ) {
                $action = $store->fetch_action( $action_id );
                if ( $action ) {
                    $actions[] = [
                        'id'        => $action_id,
                        'hook'      => $action->get_hook(),
                        'args'      => $action->get_args(),
                        'group'     => $action->get_group(),
                        'status'    => $store->get_status( $action_id ),
                        'scheduled' => $action->get_schedule()->get_date() 
                            ? $action->get_schedule()->get_date()->format( 'Y-m-d H:i:s' ) 
                            : null,
                    ];
                }
            }
            
            return $actions;
            
        } catch ( \Exception $e ) {
            return [];
        }
    }
    
    /**
     * Cancel all pending LHA actions.
     *
     * @return int Number of actions canceled.
     */
    public static function cancel_all_pending(): int {
        if ( ! self::load() || ! class_exists( 'ActionScheduler' ) ) {
            return 0;
        }
        
        try {
            $store = \ActionScheduler::store();
            $group = 'lha_tasks';
            $total_canceled = 0;
            $max_batches = 5; // Safety cap: max 5 batches of 1000 = 5000 actions max
            $batch_size = 1000;
            
            // Process in batches with a maximum iteration limit to prevent infinite loops
            // This protects against cases where actions fail to cancel but remain pending
            for ( $batch = 0; $batch < $max_batches; $batch++ ) {
                $action_ids = $store->query_actions( [
                    'status'   => \ActionScheduler_Store::STATUS_PENDING,
                    'group'    => $group,
                    'per_page' => $batch_size,
                ], 'ids' );
                
                // If no actions found, we're done
                if ( empty( $action_ids ) ) {
                    break;
                }
                
                // Cancel each action in this batch
                foreach ( $action_ids as $action_id ) {
                    try {
                        $store->cancel_action( $action_id );
                        $total_canceled++;
                    } catch ( \Exception $e ) {
                        // If we can't cancel this specific one, skip it to avoid infinite loop
                        // The action may have been deleted by another process or have a database lock
                        continue;
                    }
                }
                
                // If we processed fewer than batch_size actions, we're done
                if ( count( $action_ids ) < $batch_size ) {
                    break;
                }
            }
            
            return $total_canceled;
            
        } catch ( \Exception $e ) {
            return 0;
        }
    }
    
    /**
     * Retry a failed action.
     *
     * @param int $action_id Action ID to retry.
     * @return bool True if rescheduled.
     */
    public static function retry_action( int $action_id ): bool {
        if ( ! self::load() || ! class_exists( 'ActionScheduler' ) ) {
            return false;
        }
        
        try {
            $store = \ActionScheduler::store();
            $action = $store->fetch_action( $action_id );
            
            if ( ! $action || ! $action->is_finished() ) {
                return false;
            }
            
            // Get the original schedule to preserve schedule type and date
            $schedule = $action->get_schedule();
            $scheduled_date = $schedule->get_date();
            
            // Get priority from the action (default to 10 if not available)
            $priority = 10;
            if ( method_exists( $action, 'get_priority' ) ) {
                $priority = $action->get_priority();
            }
            
            // Determine when to schedule the retry
            // If original schedule had a date, use it; otherwise schedule immediately
            $schedule_time = $scheduled_date ? $scheduled_date->getTimestamp() : time();
            
            // Reschedule the action with preserved priority and schedule time
            // Priority is passed as the 6th parameter to as_schedule_single_action
            $new_action_id = as_schedule_single_action(
                $schedule_time,
                $action->get_hook(),
                $action->get_args(),
                $action->get_group(),
                false, // Don't mark as unique
                $priority // Preserve original priority
            );
            
            return $new_action_id > 0;
            
        } catch ( \Exception $e ) {
            return false;
        }
    }
    
    /**
     * Get the Action Scheduler admin URL.
     *
     * @return string|null Admin URL or null if not available.
     */
    public static function get_admin_url(): string {
        if ( ! self::load() ) {
            return null;
        }
        
        // Action Scheduler adds its admin page under Tools
        return admin_url( 'tools.php?page=action-scheduler' );
    }
    
    /**
     * Check if Action Scheduler admin page exists.
     *
     * @return bool True if admin page exists.
     */
    public static function has_admin_page(): bool {
        if ( ! self::load() ) {
            return false;
        }
        
        global $submenu;
        
        if ( isset( $submenu['tools.php'] ) ) {
            foreach ( $submenu['tools.php'] as $item ) {
                if ( isset( $item[2] ) && $item[2] === 'action-scheduler' ) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    /**
     * Get tasks with details for display.
     * Required by ActionSchedulerHelperInterface.
     *
     * @param array $args Query arguments (per_page, paged, status, etc.)
     * @return array Array with 'tasks' and 'total' keys
     */
    public function get_tasks_with_details(array $args): array {
        if ( ! self::load() || ! class_exists( 'ActionScheduler' ) ) {
            return [
                'tasks' => [],
                'total' => 0,
            ];
        }
        
        try {
            $store = \ActionScheduler::store();
            $group = 'lha_tasks';
            
            // Build query args
            $query_args = [
                'group'    => $group,
                'per_page' => $args['per_page'] ?? 20,
                'offset'   => (($args['paged'] ?? 1) - 1) * ($args['per_page'] ?? 20),
                'orderby'  => 'scheduled_date_gmt',
                'order'    => 'DESC',
            ];
            
            // Add status filter if provided
            if ( ! empty( $args['status'] ) && $args['status'] !== 'all' ) {
                $status_map = [
                    'pending'  => \ActionScheduler_Store::STATUS_PENDING,
                    'running'  => \ActionScheduler_Store::STATUS_RUNNING,
                    'complete' => \ActionScheduler_Store::STATUS_COMPLETE,
                    'failed'   => \ActionScheduler_Store::STATUS_FAILED,
                    'canceled' => \ActionScheduler_Store::STATUS_CANCELED,
                ];
                if ( isset( $status_map[ $args['status'] ] ) ) {
                    $query_args['status'] = $status_map[ $args['status'] ];
                }
            }
            
            // Get total count
            // Use per_page => 1 for consistent count behavior across AS versions
            $count_args = $query_args;
            $count_args['per_page'] = 1;
            unset( $count_args['offset'] );
            $total = $store->query_actions( $count_args, 'count' );
            
            // Get action IDs
            $action_ids = $store->query_actions( $query_args, 'ids' );
            $tasks = [];
            
            foreach ( $action_ids as $action_id ) {
                $action = $store->fetch_action( $action_id );
                if ( $action ) {
                    $action_args = $action->get_args();
                    $scheduled_date = $action->get_schedule()->get_date();
                    
                    $tasks[] = [
                        'id'           => $action_id,
                        'hook'         => $action->get_hook(),
                        'args'         => $action_args,
                        'group'        => $action->get_group(),
                        'status'       => $store->get_status( $action_id ),
                        'scheduled'    => $scheduled_date ? $scheduled_date->format( 'Y-m-d H:i:s' ) : null,
                        'task_id'      => $action_args[0] ?? null,
                        'priority'     => $action_args[1] ?? 10,
                        'task_data'    => $action_args[2] ?? [],
                    ];
                }
            }
            
            return [
                'tasks' => $tasks,
                'total' => (int) $total,
            ];
            
        } catch ( \Exception $e ) {
            return [
                'tasks' => [],
                'total' => 0,
                'error' => $e->getMessage(),
            ];
        }
    }
    
    /**
     * Cancel a specific task by task ID.
     * Required by ActionSchedulerHelperInterface.
     *
     * @param int|string $task_id The task ID to cancel
     * @return bool True if cancelled successfully, false otherwise
     */
    public function cancel_task($task_id): bool {
        if ( ! self::load() || ! class_exists( 'ActionScheduler' ) ) {
            return false;
        }
        
        $task_id = (int) $task_id;
        if ( $task_id <= 0 ) {
            return false;
        }
        
        try {
            $store = \ActionScheduler::store();
            $group = 'lha_tasks';
            $cancelled = false;
            $max_batches = 10; // Safety cap: max 10 batches = 10,000 actions max
            $batch_size = 1000;
            
            // Process in batches to handle cases where a task has many pending actions
            for ( $batch = 0; $batch < $max_batches; $batch++ ) {
                $action_ids = $store->query_actions( [
                    'hook'     => 'lha_as_process_task',
                    'status'   => \ActionScheduler_Store::STATUS_PENDING,
                    'group'    => $group,
                    'per_page' => $batch_size,
                ], 'ids' );
                
                // If no actions found, we're done
                if ( empty( $action_ids ) ) {
                    break;
                }
                
                // Check each action in this batch
                foreach ( $action_ids as $action_id ) {
                    try {
                        $action = $store->fetch_action( $action_id );
                        if ( $action ) {
                            $args = $action->get_args();
                            // Check if this action is for the specified task_id
                            if ( isset( $args[0] ) && (int) $args[0] === $task_id ) {
                                // Cancel the action
                                // Note: as_unschedule_action() cancels ALL actions matching the hook, args, and group.
                                // If there are 100 identical actions, this single call cancels all 100.
                                // We break the inner loop after cancellation to avoid unnecessary iterations,
                                // but continue checking other batches in case there are different action configurations.
                                if ( function_exists( 'as_unschedule_action' ) ) {
                                    as_unschedule_action( 'lha_as_process_task', $args, $group );
                                    $cancelled = true;
                                    // Break inner loop since as_unschedule_action cancelled all matching actions
                                    // Continue outer loop to check other batches for different configurations
                                    break;
                                } else {
                                    // Fallback to store method if function doesn't exist
                                    // This only cancels the specific action_id, so we continue checking
                                    $store->cancel_action( $action_id );
                                    $cancelled = true;
                                }
                            }
                        }
                    } catch ( \Exception $e ) {
                        // Skip actions that can't be fetched or cancelled
                        continue;
                    }
                }
                
                // If we processed fewer than batch_size actions, we're done
                if ( count( $action_ids ) < $batch_size ) {
                    break;
                }
            }
            
            // Also try to cancel by action ID if task_id is actually an action ID
            if ( ! $cancelled && function_exists( 'as_cancel_action' ) ) {
                try {
                    $action = $store->fetch_action( $task_id );
                    if ( $action && $action->get_group() === $group ) {
                        $store->cancel_action( $task_id );
                        $cancelled = true;
                    }
                } catch ( \Exception $e ) {
                    // Action ID not found, that's okay
                }
            }
            
            return $cancelled;
            
        } catch ( \Exception $e ) {
            return false;
        }
    }
}
