<?php
namespace LHA;

class ServiceContainer
{
    private static $instance = null;
    private $services = [];
    private $resolved = [];

    private function __construct()
    {
        $this->registerServices();
    }

    public static function getInstance()
    {
        if (self::$instance === null) {
            self::$instance = new self();
        }
        return self::$instance;
    }

    private function registerServices()
    {
        // Use LHA_PLUGIN_PATH if available, otherwise use __DIR__
        $basePath = defined('LHA_PLUGIN_PATH') ? LHA_PLUGIN_PATH : __DIR__ . '/';
        $servicesFile = $basePath . 'src/services.php';

        // Try alternative path if first attempt fails
        if (!file_exists($servicesFile)) {
            $servicesFile = __DIR__ . '/src/services.php';
        }

        if (file_exists($servicesFile)) {
            $services = require $servicesFile;
            if (is_array($services)) {
                $this->services = $services;
            }
        } else {
            // If services file not found, fall back to default services
            $this->services = $this->getDefaultServices();

            // Log error if services file not found
            if (function_exists('error_log')) {
                error_log("LHA ServiceContainer: services.php not found at: {$servicesFile}, using default services");
                error_log("LHA ServiceContainer: LHA_PLUGIN_PATH = " . (defined('LHA_PLUGIN_PATH') ? LHA_PLUGIN_PATH : 'not defined'));
                error_log("LHA ServiceContainer: __DIR__ = " . __DIR__);
            }
        }
    }

    /**
     * Get default services as fallback when services.php is not available
     */
    private function getDefaultServices()
    {
        return [
            // Core interfaces with default implementations
            \LHA\Interfaces\LoggerInterface::class => function ($container) {
                return new \LHA\LoggingAdapter();
            },
            \LHA\Interfaces\DatabaseInterface::class => function ($container) {
                // Check if WordPress is loaded and $wpdb is available
                if (!function_exists('get_option') || !isset($GLOBALS['wpdb'])) {
                    // In test environments or before full WP initialization, return a mock or throw a more specific exception
                    if (defined('LHA_TEST_ENVIRONMENT') && LHA_TEST_ENVIRONMENT) {
                        // In test environment, we may need to handle this differently
                        // For now, let's return a placeholder or throw a specific exception
                        throw new \Exception("WordPress database object (\$wpdb) not available in test environment");
                    } else {
                        throw new \Exception("WordPress database object (\$wpdb) not available");
                    }
                }

                global $wpdb;
                if ($wpdb === null || !is_object($wpdb)) {
                    throw new \Exception("WordPress database object (\$wpdb) not available");
                }
                return new \LHA\Database(
                    $wpdb,
                    null, // Logger is optional to avoid circular dependency
                    $container->get(\LHA\Interfaces\LockInterface::class),
                    $container->get(\LHA\Interfaces\AssetValidatorInterface::class),
                    $container->get(\LHA\Interfaces\NormalizeInterface::class),
                    $container->get(\LHA\Interfaces\UrlProcessorInterface::class)
                );
            },
            \LHA\Interfaces\InitializeInterface::class => function ($container) {
                $database = null;
                try {
                    $database = $container->get(\LHA\Interfaces\DatabaseInterface::class);
                } catch (\Exception $e) {
                    // Database service not available, perhaps in test environment
                    if (defined('LHA_TEST_ENVIRONMENT') && LHA_TEST_ENVIRONMENT) {
                        // In test environment, we might need to create a mock or skip
                        // For now, we'll skip initialization in test mode
                        throw new \Exception("Database service not available for Initialize in test environment");
                    }
                    // In non-test environment, let the original exception bubble up
                    throw $e;
                }
                return new \LHA\Initialize($database);
            },
            \LHA\Interfaces\LockInterface::class => function ($container) {
                return new \LHA\FileLock();
            },
            \LHA\Interfaces\AssetValidatorInterface::class => function ($container) {
                $normalize = null;
                try {
                    $normalize = $container->get(\LHA\Interfaces\NormalizeInterface::class);
                } catch (\Exception $e) {
                    // Normalize service not available, continue without it
                }
                return new \LHA\AssetValidator($normalize);
            },
            \LHA\Interfaces\NormalizeInterface::class => function ($container) {
                $logger = null;
                try {
                    $logger = $container->get(\LHA\Interfaces\LoggerInterface::class);
                } catch (\Exception $e) {
                    // Logger service not available, continue without it
                }
                return new \LHA\Normalize($logger);
            },
            \LHA\Interfaces\UrlProcessorInterface::class => function ($container) {
                $assetValidator = null;
                $normalize = null;
                try {
                    $assetValidator = $container->get(\LHA\Interfaces\AssetValidatorInterface::class);
                } catch (\Exception $e) {
                    // AssetValidator service not available, continue without it
                }
                try {
                    $normalize = $container->get(\LHA\Interfaces\NormalizeInterface::class);
                } catch (\Exception $e) {
                    // Normalize service not available, continue without it
                }
                return new \LHA\UrlProcessor($assetValidator, $normalize);
            },
        ];
    }

    /**
     * Get a service from the container
     * @param string $className Service class name or interface
     * @return object The resolved service instance
     * @throws \Exception If service not found
     */
    public function get(string $className): object
    {
        // Normalize class name (remove leading backslash if present)
        $className = ltrim($className, '\\');
        
        // Return cached instance if already resolved
        if (isset($this->resolved[$className])) {
            return $this->resolved[$className];
        }

        // Check if service is registered
        if (!isset($this->services[$className])) {
            throw new \Exception("Service not found: {$className}");
        }

        $definition = $this->services[$className];

        // If it's a closure, call it with the container
        if ($definition instanceof \Closure) {
            $instance = $definition($this);
        } 
        // If it's a string (class name), instantiate it
        elseif (is_string($definition)) {
            $instance = new $definition();
        } 
        // If it's already an instance, use it
        else {
            $instance = $definition;
        }

        // Cache the resolved instance
        $this->resolved[$className] = $instance;

        return $instance;
    }

    /**
     * Check if a service is registered
     * @param string $className Service class name or interface
     * @return bool
     */
    public function has($className): bool
    {
        return isset($this->services[$className]);
    }
}