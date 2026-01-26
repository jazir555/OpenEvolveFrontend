<<<<<<< HEAD
<?php

namespace LHA;

use LHA\Interfaces\LoggerInterface;

/**
 * MemoryUtils class
 * Provides utility methods for memory management, monitoring, and optimization
 * with consistent error handling and logging.
 */
class MemoryUtils
{
    /**
     * Logger instance
     * @var LoggerInterface
     */
    private LoggerInterface $logger;

    /**
     * Constructor
     *
     * @param LoggerInterface $logger Logger instance for error reporting
     */
    public function __construct(LoggerInterface $logger)
    {
        $this->logger = $logger;
    }

    /**
     * Convert PHP memory size notation to bytes
     *
     * @param string $size Memory size string (e.g., '128M', '1G', '512K')
     * @return int Size in bytes
     */
    public function convertToBytes(string $size): int
    {
        $unit = strtolower(substr($size, -1));
        $value = (int) substr($size, 0, -1);

        switch ($unit) {
            case 'g':
                return $value * 1024 * 1024 * 1024;
            case 'm':
                return $value * 1024 * 1024;
            case 'k':
                return $value * 1024;
            default:
                return $value;
        }
    }

    /**
     * Get current memory usage percentage relative to PHP memory limit
     *
     * @return float|null Memory usage percentage, or null if unable to determine
     */
    public function getMemoryUsagePercentage(): ?float
    {
        if (!function_exists('memory_get_usage') || !function_exists('ini_get')) {
            return null;
        }

        $memory_limit = ini_get('memory_limit');
        if ($memory_limit === false || $memory_limit === '-1') {
            return null; // No memory limit set
        }

        $memory_limit_bytes = $this->convertToBytes($memory_limit);
        $current_usage = memory_get_usage(true);
        return ($current_usage / $memory_limit_bytes) * 100;
    }

    /**
     * Get current memory usage in bytes
     *
     * @param bool $real_usage Whether to get real usage (including unused allocated memory)
     * @return int|null Memory usage in bytes, or null if function not available
     */
    public function getMemoryUsage(bool $real_usage = true): ?int
    {
        if (!function_exists('memory_get_usage')) {
            return null;
        }
        return memory_get_usage($real_usage);
    }

    /**
     * Get PHP memory limit in bytes
     *
     * @return int|null Memory limit in bytes, or null if unable to determine
     */
    public function getMemoryLimit(): ?int
    {
        if (!function_exists('ini_get')) {
            return null;
        }

        $memory_limit = ini_get('memory_limit');
        if ($memory_limit === false || $memory_limit === '-1') {
            return null; // No memory limit set
        }

        return $this->convertToBytes($memory_limit);
    }

    /**
     * Perform garbage collection if available
     *
     * @return int|null Number of cycles collected, or null if GC not available
     */
    public function performGarbageCollection(): ?int
    {
        if (!function_exists('gc_collect_cycles')) {
            return null;
        }

        // Multiple GC passes for circular references
        $collected = gc_collect_cycles();
        if ($collected > 0) {
            gc_collect_cycles(); // Second pass
        }

        return $collected;
    }

    /**
     * Check if memory usage is above a critical threshold and log warnings
     *
     * @param float $critical_threshold Critical threshold percentage (default 80%)
     * @param float $warning_threshold Warning threshold percentage (default 65%)
     * @return bool True if action was taken due to high memory usage
     */
    public function checkMemoryUsage(float $critical_threshold = 80.0, float $warning_threshold = 65.0): bool
    {
        $usage_percentage = $this->getMemoryUsagePercentage();
        if ($usage_percentage === null) {
            return false; // Unable to check memory
        }

        $memory_limit = $this->getMemoryLimit();
        $current_usage = $this->getMemoryUsage(true);

        if ($usage_percentage > $critical_threshold) {
            $this->logger->log(
                sprintf(
                    'Critical memory usage detected: %.1f%% (%s of %s).',
                    $usage_percentage,
                    $current_usage ? size_format($current_usage) : 'unknown',
                    $memory_limit ? size_format($memory_limit) : 'unknown'
                ),
                'critical',
                [
                    'context' => 'MemoryUtils::checkMemoryUsage',
                    'usage_percentage' => $usage_percentage,
                    'critical_threshold' => $critical_threshold
                ]
            );
            return true;

        } elseif ($usage_percentage > $warning_threshold) {
            $this->logger->log(
                sprintf(
                    'High memory usage detected: %.1f%% (%s of %s).',
                    $usage_percentage,
                    $current_usage ? size_format($current_usage) : 'unknown',
                    $memory_limit ? size_format($memory_limit) : 'unknown'
                ),
                'warning',
                [
                    'context' => 'MemoryUtils::checkMemoryUsage',
                    'usage_percentage' => $usage_percentage,
                    'warning_threshold' => $warning_threshold
                ]
            );
            return true;
        }

        return false;
    }
}


=======
<?php

namespace LHA;

use LHA\Interfaces\LoggerInterface;

/**
 * MemoryUtils class
 * Provides utility methods for memory management, monitoring, and optimization
 * with consistent error handling and logging.
 */
class MemoryUtils
{
    /**
     * Logger instance
     * @var LoggerInterface
     */
    private LoggerInterface $logger;

    /**
     * Constructor
     *
     * @param LoggerInterface $logger Logger instance for error reporting
     */
    public function __construct(LoggerInterface $logger)
    {
        $this->logger = $logger;
    }

    /**
     * Convert PHP memory size notation to bytes
     *
     * @param string $size Memory size string (e.g., '128M', '1G', '512K')
     * @return int Size in bytes
     */
    public function convertToBytes(string $size): int
    {
        $unit = strtolower(substr($size, -1));
        $value = (int) substr($size, 0, -1);

        switch ($unit) {
            case 'g':
                return $value * 1024 * 1024 * 1024;
            case 'm':
                return $value * 1024 * 1024;
            case 'k':
                return $value * 1024;
            default:
                return $value;
        }
    }

    /**
     * Get current memory usage percentage relative to PHP memory limit
     *
     * @return float|null Memory usage percentage, or null if unable to determine
     */
    public function getMemoryUsagePercentage(): ?float
    {
        if (!function_exists('memory_get_usage') || !function_exists('ini_get')) {
            return null;
        }

        $memory_limit = ini_get('memory_limit');
        if ($memory_limit === false || $memory_limit === '-1') {
            return null; // No memory limit set
        }

        $memory_limit_bytes = $this->convertToBytes($memory_limit);
        $current_usage = memory_get_usage(true);
        return ($current_usage / $memory_limit_bytes) * 100;
    }

    /**
     * Get current memory usage in bytes
     *
     * @param bool $real_usage Whether to get real usage (including unused allocated memory)
     * @return int|null Memory usage in bytes, or null if function not available
     */
    public function getMemoryUsage(bool $real_usage = true): ?int
    {
        if (!function_exists('memory_get_usage')) {
            return null;
        }
        return memory_get_usage($real_usage);
    }

    /**
     * Get PHP memory limit in bytes
     *
     * @return int|null Memory limit in bytes, or null if unable to determine
     */
    public function getMemoryLimit(): ?int
    {
        if (!function_exists('ini_get')) {
            return null;
        }

        $memory_limit = ini_get('memory_limit');
        if ($memory_limit === false || $memory_limit === '-1') {
            return null; // No memory limit set
        }

        return $this->convertToBytes($memory_limit);
    }

    /**
     * Perform garbage collection if available
     *
     * @return int|null Number of cycles collected, or null if GC not available
     */
    public function performGarbageCollection(): ?int
    {
        if (!function_exists('gc_collect_cycles')) {
            return null;
        }

        // Multiple GC passes for circular references
        $collected = gc_collect_cycles();
        if ($collected > 0) {
            gc_collect_cycles(); // Second pass
        }

        return $collected;
    }

    /**
     * Check if memory usage is above a critical threshold and log warnings
     *
     * @param float $critical_threshold Critical threshold percentage (default 80%)
     * @param float $warning_threshold Warning threshold percentage (default 65%)
     * @return bool True if action was taken due to high memory usage
     */
    public function checkMemoryUsage(float $critical_threshold = 80.0, float $warning_threshold = 65.0): bool
    {
        $usage_percentage = $this->getMemoryUsagePercentage();
        if ($usage_percentage === null) {
            return false; // Unable to check memory
        }

        $memory_limit = $this->getMemoryLimit();
        $current_usage = $this->getMemoryUsage(true);

        if ($usage_percentage > $critical_threshold) {
            $this->logger->log(
                sprintf(
                    'Critical memory usage detected: %.1f%% (%s of %s).',
                    $usage_percentage,
                    $current_usage ? size_format($current_usage) : 'unknown',
                    $memory_limit ? size_format($memory_limit) : 'unknown'
                ),
                'critical',
                [
                    'context' => 'MemoryUtils::checkMemoryUsage',
                    'usage_percentage' => $usage_percentage,
                    'critical_threshold' => $critical_threshold
                ]
            );
            return true;

        } elseif ($usage_percentage > $warning_threshold) {
            $this->logger->log(
                sprintf(
                    'High memory usage detected: %.1f%% (%s of %s).',
                    $usage_percentage,
                    $current_usage ? size_format($current_usage) : 'unknown',
                    $memory_limit ? size_format($memory_limit) : 'unknown'
                ),
                'warning',
                [
                    'context' => 'MemoryUtils::checkMemoryUsage',
                    'usage_percentage' => $usage_percentage,
                    'warning_threshold' => $warning_threshold
                ]
            );
            return true;
        }

        return false;
    }
}


>>>>>>> 1cb9c5e35 (update)
