"""
Fuzzing Integration for OpenEvolve Gauntlet System

Automated fuzzing of solutions to find edge cases, crashes, and
vulnerabilities. Integrates with Red Team for comprehensive testing.

Key Features:
- Automatic input generation based on solution type
- Crash and exception detection
- Vulnerability reporting with severity classification
- Corpus management for interesting inputs
- Coverage tracking
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import random
import string
import logging
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class VulnerabilitySeverity(Enum):
    """Severity levels for vulnerabilities"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Vulnerability:
    """Represents a discovered vulnerability"""
    vulnerability_id: str
    severity: VulnerabilitySeverity
    title: str
    description: str
    input_data: Any
    crash_type: str  # exception, timeout, hang, assertion_failure
    stack_trace: Optional[str] = None
    reproduction_steps: List[str] = field(default_factory=list)
    discovered_at: datetime = None
    reproducible: bool = True

    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()


@dataclass
class FuzzResult:
    """Result of a fuzzing session"""
    iterations: int
    crashes_found: int
    unique_crashes: int
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    corpus_size: int = 0
    coverage_percentage: float = 0.0
    execution_time: float = 0.0
    interesting_inputs: List[Any] = field(default_factory=list)


class FuzzInputGenerator:
    """
    Generates random and structured inputs for fuzzing.
    """

    def __init__(self, seed: int = None):
        self.random = random.Random(seed)
        self.generators = {
            'string': self._generate_string,
            'number': self._generate_number,
            'boolean': self._generate_boolean,
            'array': self._generate_array,
            'object': self._generate_object,
            'null': self._generate_null,
            'edge_case': self._generate_edge_case,
        }

    def generate_input(self, input_type: str = 'auto', constraints: Dict[str, Any] = None) -> Any:
        """
        Generate a random input for fuzzing.

        Args:
            input_type: Type of input to generate
            constraints: Optional constraints (min, max, length, etc.)

        Returns:
            Generated input
        """
        constraints = constraints or {}

        if input_type == 'auto':
            # Randomly select a type
            input_type = self.random.choice(list(self.generators.keys()))

        generator = self.generators.get(input_type, self._generate_string)
        return generator(constraints)

    def _generate_string(self, constraints: Dict[str, Any]) -> str:
        """Generate random string"""
        max_length = constraints.get('max_length', 100)
        min_length = constraints.get('min_length', 0)

        # Sometimes generate interesting strings
        if self.random.random() < 0.1:
            interesting = [
                '', 'null', 'undefined', 'NaN', 'Infinity',
                '\x00', '\n', '\r', '\t', '\\',
                '../../etc/passwd', '<script>alert(1)</script>',
                '1' * 10000, 'a' * 10000,
                '\x7f\x45\x4c\x46',  # ELF magic
                '%n' * 100,  # Format string
                '\x00' * 100,  # Null bytes
            ]
            return self.random.choice(interesting)

        length = self.random.randint(min_length, max_length)
        chars = string.ascii_letters + string.digits + string.punctuation + ' '
        return ''.join(self.random.choice(chars) for _ in range(length))

    def _generate_number(self, constraints: Dict[str, Any]) -> int:
        """Generate random number"""
        min_val = constraints.get('min', -1000000)
        max_val = constraints.get('max', 1000000)

        # Sometimes generate edge cases
        if self.random.random() < 0.1:
            edge_cases = [0, -1, 1, 2**31 - 1, -2**31, 2**63 - 1, -2**63,
                         float('inf'), float('-inf'), float('nan'),
                         3.14159, 1.61803398875, 2.71828]
            return self.random.choice(edge_cases)

        return self.random.randint(min_val, max_val)

    def _generate_boolean(self, constraints: Dict[str, Any]) -> bool:
        """Generate random boolean"""
        return self.random.choice([True, False])

    def _generate_array(self, constraints: Dict[str, Any]) -> list:
        """Generate random array"""
        max_length = constraints.get('max_length', 10)
        min_length = constraints.get('min_length', 0)

        length = self.random.randint(min_length, max_length)
        return [self.generate_input('auto') for _ in range(length)]

    def _generate_object(self, constraints: Dict[str, Any]) -> dict:
        """Generate random object"""
        max_keys = constraints.get('max_keys', 5)
        min_keys = constraints.get('min_keys', 0)

        num_keys = self.random.randint(min_keys, max_keys)
        obj = {}

        for _ in range(num_keys):
            key = self._generate_string({'max_length': 20})
            obj[key] = self.generate_input('auto')

        return obj

    def _generate_null(self, constraints: Dict[str, Any]) -> None:
        """Generate null/None"""
        return None

    def _generate_edge_case(self, constraints: Dict[str, Any]) -> Any:
        """Generate edge case inputs"""
        edge_cases = [
            None, '', 0, -0, False, [],
            {}, set(), float('nan'), float('inf'),
            -float('inf'), -0.0, '\x00',
            '\r\n\r\n\r\n', '\t\t\t\t',
            '      ', '⁤', 'א', '🔥',
        ]
        return self.random.choice(edge_cases)


class FuzzExecutor:
    """
    Executes solutions with fuzzed inputs and detects crashes.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    async def execute_with_fuzz(
        self,
        solution: Callable,
        fuzz_input: Any,
        input_name: str = None
    ) -> tuple[bool, Optional[Exception]]:
        """
        Execute solution with fuzzed input.

        Args:
            solution: Solution function to test
            fuzz_input: Fuzzed input to provide
            input_name: Optional name for the input

        Returns:
            Tuple of (success, exception)
        """
        input_name = input_name or str(fuzz_input)[:50]

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_solution(solution, fuzz_input),
                timeout=self.timeout
            )

            return (True, None)

        except asyncio.TimeoutError:
            logger.warning(f"Timeout with input: {input_name}")
            return (False, TimeoutError(f"Execution timed out after {self.timeout}s"))

        except Exception as e:
            logger.warning(f"Exception with input: {input_name} - {type(e).__name__}: {e}")
            return (False, e)

    async def _execute_solution(self, solution: Callable, fuzz_input: Any) -> Any:
        """Execute the solution function"""
        # Check if solution is async
        if asyncio.iscoroutinefunction(solution):
            return await solution(fuzz_input)
        else:
            # Run sync solution in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, solution, fuzz_input)


class SolutionFuzzer:
    """
    Main fuzzing engine for testing solutions.

    Manages fuzzing campaigns, corpus, and vulnerability tracking.
    """

    def __init__(
        self,
        iterations: int = 1000,
        timeout: float = 5.0,
        max_concurrent: int = 4,
        corpus_size: int = 100
    ):
        self.iterations = iterations
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.corpus_size = corpus_size

        self.generator = FuzzInputGenerator()
        self.executor = FuzzExecutor(timeout=timeout)

        # Corpus of interesting inputs
        self.corpus: List[Any] = []

        # Track unique crashes by stack trace
        self.unique_crashes: Dict[str, Vulnerability] = {}

    async def fuzz(
        self,
        solution: Callable,
        input_type: str = 'auto',
        constraints: Dict[str, Any] = None
    ) -> FuzzResult:
        """
        Fuzz a solution with random inputs.

        Args:
            solution: Solution function to fuzz
            input_type: Type of inputs to generate
            constraints: Constraints for input generation

        Returns:
            FuzzResult with findings
        """
        start_time = datetime.utcnow()
        crashes_found = 0
        vulnerabilities = []

        logger.info(f"Starting fuzzing: {self.iterations} iterations")

        # Create semaphore for concurrent execution
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fuzz_iteration(i):
            nonlocal crashes_found

            async with semaphore:
                # Generate input
                fuzz_input = self.generator.generate_input(input_type, constraints)

                # Execute with fuzzed input
                success, exception = await self.executor.execute_with_fuzz(
                    solution,
                    fuzz_input,
                    f"iteration_{i}"
                )

                if not success and exception is not None:
                    crashes_found += 1

                    # Create vulnerability if unique
                    vuln = self._create_vulnerability(exception, fuzz_input, i)
                    if vuln:
                        vulnerabilities.append(vuln)

                        # Add to corpus if interesting
                        if len(self.corpus) < self.corpus_size:
                            self.corpus.append(fuzz_input)

                return success

        # Run fuzzing iterations
        tasks = [fuzz_iteration(i) for i in range(self.iterations)]
        await asyncio.gather(*tasks, return_exceptions=True)

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        # Calculate unique crashes
        unique_crash_count = len(self.unique_crashes)

        result = FuzzResult(
            iterations=self.iterations,
            crashes_found=crashes_found,
            unique_crashes=unique_crash_count,
            vulnerabilities=vulnerabilities,
            corpus_size=len(self.corpus),
            execution_time=execution_time,
        )

        logger.info(
            f"Fuzzing complete: {crashes_found} crashes, "
            f"{unique_crash_count} unique vulnerabilities"
        )

        return result

    def _create_vulnerability(
        self,
        exception: Exception,
        fuzz_input: Any,
        iteration: int
    ) -> Optional[Vulnerability]:
        """Create a vulnerability from an exception"""
        # Create unique key from exception type and message
        exception_key = f"{type(exception).__name__}:{str(exception)[:100]}"

        if exception_key in self.unique_crashes:
            # Already seen this crash
            return None

        # Determine severity
        severity = self._assess_severity(exception)

        # Create vulnerability
        vuln = Vulnerability(
            vulnerability_id=f"vuln_{iteration}_{hash(exception_key) % 10000}",
            severity=severity,
            title=f"{type(exception).__name__} detected",
            description=str(exception),
            input_data=fuzz_input,
            crash_type=type(exception).__name__,
            stack_trace=traceback_str(exception),
            reproduction_steps=[
                f"1. Call solution with input: {repr(fuzz_input)[:200]}",
                f"2. Observe {type(exception).__name__}: {str(exception)[:100]}",
            ],
        )

        # Track unique crashes
        self.unique_crashes[exception_key] = vuln

        return vuln

    def _assess_severity(self, exception: Exception) -> VulnerabilitySeverity:
        """Assess severity of an exception"""
        exception_name = type(exception).__name__

        # Critical exceptions
        if exception_name in ['SystemExit', 'MemoryError', 'OverflowError']:
            return VulnerabilitySeverity.CRITICAL

        # High severity
        if exception_name in ['SegmentationFault', 'BusError', 'AssertionError']:
            return VulnerabilitySeverity.HIGH

        # Medium severity
        if exception_name in ['IndexError', 'KeyError', 'TypeError', 'ValueError']:
            return VulnerabilitySeverity.MEDIUM

        # Low severity
        if exception_name in ['AttributeError', 'NameError', 'ImportError']:
            return VulnerabilitySeverity.LOW

        # Default to info
        return VulnerabilitySeverity.INFO

    def get_corpus(self) -> List[Any]:
        """Get current corpus of interesting inputs"""
        return self.corpus.copy()

    def clear_corpus(self):
        """Clear the corpus"""
        self.corpus.clear()

    def clear_unique_crashes(self):
        """Clear tracked unique crashes"""
        self.unique_crashes.clear()


def traceback_str(exception: Exception) -> str:
    """Extract traceback from exception"""
    import traceback
    return ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))


async def fuzz_solution(
    solution: Callable,
    iterations: int = 1000,
    timeout: float = 5.0,
    max_concurrent: int = 4
) -> FuzzResult:
    """
    Convenience function to fuzz a solution.

    Args:
        solution: Solution function to fuzz
        iterations: Number of fuzzing iterations
        timeout: Per-execution timeout
        max_concurrent: Maximum concurrent executions

    Returns:
        FuzzResult with findings
    """
    fuzzer = SolutionFuzzer(
        iterations=iterations,
        timeout=timeout,
        max_concurrent=max_concurrent
    )

    return await fuzzer.fuzz(solution)


# Example usage
async def demo_fuzzing():
    """Demonstration of fuzzing capabilities"""

    # Example solution with a bug
    def buggy_solution(input_data):
        # Bug: doesn't handle None or empty strings
        if len(input_data) > 5:
            return input_data[:5]
        return input_data

    # Fuzz the solution
    result = await fuzz_solution(
        solution=buggy_solution,
        iterations=100,
        timeout=1.0,
        max_concurrent=4
    )

    print(f"Fuzzing Results:")
    print(f"  Iterations: {result.iterations}")
    print(f"  Crashes found: {result.crashes_found}")
    print(f"  Unique vulnerabilities: {result.unique_crashes}")
    print(f"  Execution time: {result.execution_time:.2f}s")

    if result.vulnerabilities:
        print(f"\nVulnerabilities:")
        for vuln in result.vulnerabilities[:5]:
            print(f"  - {vuln.severity.value.upper()}: {vuln.title}")
            print(f"    Input: {repr(str(vuln.input_data)[:50])}")


if __name__ == '__main__':
    asyncio.run(demo_fuzzing())
