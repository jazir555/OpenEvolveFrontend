<<<<<<< HEAD
<?php

namespace LHA;

use LHA\Interfaces\LoggerInterface;

/**
 * ValidationUtils class
 * Provides utility methods for input validation and type conversion
 * with consistent error handling and constraints support.
 */
class ValidationUtils
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
     * Validate and convert input value to specified type with constraints
     *
     * @param mixed $value The value to validate and convert
     * @param string $type The expected type ('int', 'string', 'array', 'bool')
     * @param mixed $default Default value if validation fails
     * @param array $constraints Additional constraints (min, max, min_length, max_length, etc.)
     * @return mixed The validated and converted value, or default if validation fails
     */
    public function validateInputType($value, string $type, $default = null, array $constraints = [])
    {
        switch ($type) {
            case 'int':
                if (!is_numeric($value)) {
                    $this->logger->log(
                        sprintf('Invalid integer value: %s, using default', var_export($value, true)),
                        'debug',
                        ['context' => __METHOD__, 'expected_type' => $type]
                    );
                    return $default;
                }
                $int_val = (int) $value;
                if (isset($constraints['min']) && $int_val < $constraints['min']) {
                    $this->logger->log(
                        sprintf('Integer value %d below minimum %d, using default', $int_val, $constraints['min']),
                        'debug',
                        ['context' => __METHOD__]
                    );
                    return $default;
                }
                if (isset($constraints['max']) && $int_val > $constraints['max']) {
                    $this->logger->log(
                        sprintf('Integer value %d above maximum %d, using default', $int_val, $constraints['max']),
                        'debug',
                        ['context' => __METHOD__]
                    );
                    return $default;
                }
                return $int_val;

            case 'string':
                if (!is_string($value)) {
                    $this->logger->log(
                        sprintf('Invalid string value: %s, using default', var_export($value, true)),
                        'debug',
                        ['context' => __METHOD__, 'expected_type' => $type]
                    );
                    return $default;
                }
                $str_val = trim($value);
                if (isset($constraints['min_length']) && strlen($str_val) < $constraints['min_length']) {
                    $this->logger->log(
                        sprintf('String length %d below minimum %d, using default', strlen($str_val), $constraints['min_length']),
                        'debug',
                        ['context' => __METHOD__]
                    );
                    return $default;
                }
                if (isset($constraints['max_length']) && strlen($str_val) > $constraints['max_length']) {
                    $this->logger->log(
                        sprintf('String length %d above maximum %d, using default', strlen($str_val), $constraints['max_length']),
                        'debug',
                        ['context' => __METHOD__]
                    );
                    return $default;
                }
                return $str_val;

            case 'array':
                if (!is_array($value)) {
                    $this->logger->log(
                        sprintf('Invalid array value: %s, using default', var_export($value, true)),
                        'debug',
                        ['context' => __METHOD__, 'expected_type' => $type]
                    );
                    return $default;
                }
                return $value;

            case 'bool':
                return (bool) $value;

            default:
                $this->logger->log(
                    sprintf('Unknown validation type: %s, using default', $type),
                    'warning',
                    ['context' => __METHOD__]
                );
                return $default;
        }
    }

    /**
     * Validate asset ID string and convert to integer
     *
     * @param string $asset_id_raw Raw asset ID string
     * @return int Validated asset ID
     * @throws \InvalidArgumentException If asset ID is invalid
     */
    public function validateAssetId(string $asset_id_raw): int
    {
        $asset_id = absint($asset_id_raw);
        if ($asset_id <= 0) {
            throw new \InvalidArgumentException(__('Invalid asset ID provided.', 'lha'));
        }
        return $asset_id;
    }

    /**
     * Validate parsed URL structure
     *
     * @param array $parsed Parsed URL components from parse_url
     * @return array Validated URL structure with defaults
     */
    public function validateParsedUrlStructure(array $parsed): array
    {
        // Ensure all expected components are present with proper defaults
        return [
            'scheme' => $parsed['scheme'] ?? null,
            'host' => $parsed['host'] ?? null,
            'port' => $parsed['port'] ?? null,
            'user' => $parsed['user'] ?? null,
            'pass' => $parsed['pass'] ?? null,
            'path' => $parsed['path'] ?? null,
            'query' => $parsed['query'] ?? null,
            'fragment' => $parsed['fragment'] ?? null,
        ];
    }
}


=======
<?php

namespace LHA;

use LHA\Interfaces\LoggerInterface;

/**
 * ValidationUtils class
 * Provides utility methods for input validation and type conversion
 * with consistent error handling and constraints support.
 */
class ValidationUtils
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
     * Validate and convert input value to specified type with constraints
     *
     * @param mixed $value The value to validate and convert
     * @param string $type The expected type ('int', 'string', 'array', 'bool')
     * @param mixed $default Default value if validation fails
     * @param array $constraints Additional constraints (min, max, min_length, max_length, etc.)
     * @return mixed The validated and converted value, or default if validation fails
     */
    public function validateInputType($value, string $type, $default = null, array $constraints = [])
    {
        switch ($type) {
            case 'int':
                if (!is_numeric($value)) {
                    $this->logger->log(
                        sprintf('Invalid integer value: %s, using default', var_export($value, true)),
                        'debug',
                        ['context' => __METHOD__, 'expected_type' => $type]
                    );
                    return $default;
                }
                $int_val = (int) $value;
                if (isset($constraints['min']) && $int_val < $constraints['min']) {
                    $this->logger->log(
                        sprintf('Integer value %d below minimum %d, using default', $int_val, $constraints['min']),
                        'debug',
                        ['context' => __METHOD__]
                    );
                    return $default;
                }
                if (isset($constraints['max']) && $int_val > $constraints['max']) {
                    $this->logger->log(
                        sprintf('Integer value %d above maximum %d, using default', $int_val, $constraints['max']),
                        'debug',
                        ['context' => __METHOD__]
                    );
                    return $default;
                }
                return $int_val;

            case 'string':
                if (!is_string($value)) {
                    $this->logger->log(
                        sprintf('Invalid string value: %s, using default', var_export($value, true)),
                        'debug',
                        ['context' => __METHOD__, 'expected_type' => $type]
                    );
                    return $default;
                }
                $str_val = trim($value);
                if (isset($constraints['min_length']) && strlen($str_val) < $constraints['min_length']) {
                    $this->logger->log(
                        sprintf('String length %d below minimum %d, using default', strlen($str_val), $constraints['min_length']),
                        'debug',
                        ['context' => __METHOD__]
                    );
                    return $default;
                }
                if (isset($constraints['max_length']) && strlen($str_val) > $constraints['max_length']) {
                    $this->logger->log(
                        sprintf('String length %d above maximum %d, using default', strlen($str_val), $constraints['max_length']),
                        'debug',
                        ['context' => __METHOD__]
                    );
                    return $default;
                }
                return $str_val;

            case 'array':
                if (!is_array($value)) {
                    $this->logger->log(
                        sprintf('Invalid array value: %s, using default', var_export($value, true)),
                        'debug',
                        ['context' => __METHOD__, 'expected_type' => $type]
                    );
                    return $default;
                }
                return $value;

            case 'bool':
                return (bool) $value;

            default:
                $this->logger->log(
                    sprintf('Unknown validation type: %s, using default', $type),
                    'warning',
                    ['context' => __METHOD__]
                );
                return $default;
        }
    }

    /**
     * Validate asset ID string and convert to integer
     *
     * @param string $asset_id_raw Raw asset ID string
     * @return int Validated asset ID
     * @throws \InvalidArgumentException If asset ID is invalid
     */
    public function validateAssetId(string $asset_id_raw): int
    {
        $asset_id = absint($asset_id_raw);
        if ($asset_id <= 0) {
            throw new \InvalidArgumentException(__('Invalid asset ID provided.', 'lha'));
        }
        return $asset_id;
    }

    /**
     * Validate parsed URL structure
     *
     * @param array $parsed Parsed URL components from parse_url
     * @return array Validated URL structure with defaults
     */
    public function validateParsedUrlStructure(array $parsed): array
    {
        // Ensure all expected components are present with proper defaults
        return [
            'scheme' => $parsed['scheme'] ?? null,
            'host' => $parsed['host'] ?? null,
            'port' => $parsed['port'] ?? null,
            'user' => $parsed['user'] ?? null,
            'pass' => $parsed['pass'] ?? null,
            'path' => $parsed['path'] ?? null,
            'query' => $parsed['query'] ?? null,
            'fragment' => $parsed['fragment'] ?? null,
        ];
    }
}


>>>>>>> 1cb9c5e35 (update)
