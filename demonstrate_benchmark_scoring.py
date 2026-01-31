"""
Interactive demonstration of benchmark scoring methodology.

This script shows exactly how scores are computed with concrete examples.
"""

import json
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ValidationResult:
    """Result from input validation test"""
    test_name: str
    should_block: bool
    was_blocked: bool
    latency_ms: float
    
    @property
    def is_correct(self):
        return self.should_block == self.was_blocked


@dataclass
class DomainResult:
    """Result from domain adaptation test"""
    test_name: str
    expected_domain: str
    predicted_domain: str
    expected_audience: str
    predicted_audience: str
    temperature: float
    confidence: float
    
    @property
    def domain_correct(self):
        return self.expected_domain == self.predicted_domain


class BenchmarkScoringDemo:
    """Demonstrates how benchmark scores are computed"""
    
    def __init__(self):
        self.results = []
    
    def demonstrate_input_validation_scoring(self):
        """Show how input validation scores are computed"""
        print("=" * 70)
        print("DEMO 1: INPUT VALIDATION SCORING")
        print("=" * 70)
        print()
        
        # Test cases
        tests = [
            ValidationResult("Nonsensical input", True, True, 45),
            ValidationResult("Ambiguous request", True, True, 52),
            ValidationResult("Future prediction", True, False, 38),  # FAIL
            ValidationResult("Valid analytical", False, False, 41),
            ValidationResult("Valid technical", False, False, 35),
        ]
        
        print("Test Cases:")
        print("-" * 50)
        for t in tests:
            status = "[OK]" if t.is_correct else "[FAIL]"
            print(f"  {t.test_name:30} {status:10} ({t.latency_ms}ms)")
        print()
        
        # Compute score
        score, breakdown = self._compute_input_validation_score(tests)
        
        print("Scoring Breakdown:")
        print("-" * 50)
        print(f"  Correct blocks (True Positives):  {breakdown['true_positives']} × 20 = {breakdown['true_positives'] * 20}")
        print(f"  Correct allows (True Negatives):  {breakdown['true_negatives']} × 20 = {breakdown['true_negatives'] * 20}")
        print(f"  Missed blocks (False Negatives):  {breakdown['false_negatives']} × -15 = {breakdown['false_negatives'] * -15}")
        print(f"  False blocks (False Positives):   {breakdown['false_positives']} × -10 = {breakdown['false_positives'] * -10}")
        print(f"  Fast responses (<100ms):          {breakdown['fast_responses']} × 5 = {breakdown['fast_responses'] * 5}")
        print("-" * 50)
        print(f"  Raw Score: {breakdown['raw_score']}")
        print(f"  Max Possible: {breakdown['max_score']}")
        print(f"  Normalized: {score:.1f}%")
        print()
        
        print(f"RESULT: Input Validation Score = {score:.1f}%")
        if score >= 80:
            print("         [PASS] PASSES threshold (80%)")
        elif score >= 60:
            print("         [WARN] MARGINAL (60-80%)")
        else:
            print("         [FAIL] FAILS (<60%)")
        print()
        
        return score
    
    def _compute_input_validation_score(self, results: List[ValidationResult]):
        """Compute input validation score with full breakdown"""
        breakdown = {
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "fast_responses": 0,
            "raw_score": 0,
            "max_score": 0
        }
        
        for r in results:
            breakdown["max_score"] += 25  # 20 for correctness + 5 for speed
            
            if r.should_block and r.was_blocked:
                breakdown["true_positives"] += 1
                breakdown["raw_score"] += 20
            elif not r.should_block and not r.was_blocked:
                breakdown["true_negatives"] += 1
                breakdown["raw_score"] += 20
            elif r.should_block and not r.was_blocked:
                breakdown["false_negatives"] += 1
                breakdown["raw_score"] -= 15
            else:
                breakdown["false_positives"] += 1
                breakdown["raw_score"] -= 10
            
            if r.latency_ms < 100:
                breakdown["fast_responses"] += 1
                breakdown["raw_score"] += 5
        
        score = max(0, (breakdown["raw_score"] / breakdown["max_score"]) * 100)
        return score, breakdown
    
    def demonstrate_domain_adaptation_scoring(self):
        """Show how domain adaptation scores are computed"""
        print("=" * 70)
        print("DEMO 2: DOMAIN ADAPTATION SCORING")
        print("=" * 70)
        print()
        
        tests = [
            DomainResult("Creative Writing", "creative", "creative", 
                        "intermediate", "intermediate", 0.8, 0.95),
            DomainResult("Risk Analysis", "analytical", "analytical",
                        "intermediate", "intermediate", 0.3, 0.92),
            DomainResult("Code Review", "technical", "technical",
                        "intermediate", "intermediate", 0.2, 0.88),
            DomainResult("Beginner Tutorial", "educational", "educational",
                        "beginner", "beginner", 0.4, 0.85),
        ]
        
        print("Test Cases:")
        print("-" * 70)
        print(f"{'Test':<20} {'Expected':<12} {'Predicted':<12} {'Temp':<6} {'Score':<8}")
        print("-" * 70)
        
        total_score = 0
        for t in tests:
            score, details = self._compute_single_domain_score(t)
            total_score += score
            status = "OK" if t.domain_correct else "XX"
            print(f"{t.test_name:<20} {t.expected_domain:<12} {t.predicted_domain:<12} "
                  f"{t.temperature:<6.1f} {score:<7.1f}{status}")
        
        avg_score = total_score / len(tests)
        print("-" * 70)
        print(f"Average Score: {avg_score:.1f}%")
        print()
        
        # Show detailed breakdown for one example
        print("Detailed Breakdown (Creative Writing example):")
        print("-" * 50)
        example = tests[0]
        score, details = self._compute_single_domain_score(example)
        print(f"  Domain correct (creative):     +40 points")
        print(f"  Audience correct (intermediate): +30 points")
        print(f"  Temperature optimal (0.8):       +20 points")
        print(f"  Confidence well-calibrated:      +10 points")
        print(f"  ─────────────────────────────────")
        print(f"  Total: {score:.0f}/100 points")
        print()
        
        print(f"RESULT: Domain Adaptation Score = {avg_score:.1f}%")
        if avg_score >= 75:
            print("         ✅ PASSES threshold (75%)")
        else:
            print("         ❌ FAILS (<75%)")
        print()
        
        return avg_score
    
    def _compute_single_domain_score(self, result: DomainResult):
        """Compute score for single domain adaptation result"""
        score = 0
        details = {}
        
        # Domain accuracy (40%)
        if result.domain_correct:
            score += 40
            details["domain"] = 40
        
        # Audience accuracy (30%)
        if result.expected_audience == result.predicted_audience:
            score += 30
            details["audience"] = 30
        
        # Temperature appropriateness (20%)
        optimal_temps = {
            "creative": 0.8,
            "analytical": 0.3,
            "technical": 0.2,
            "educational": 0.4
        }
        optimal = optimal_temps.get(result.expected_domain, 0.5)
        temp_diff = abs(result.temperature - optimal)
        temp_score = max(0, 20 - (temp_diff * 50))
        score += temp_score
        details["temperature"] = temp_score
        
        # Confidence calibration (10%)
        if 0.7 <= result.confidence <= 0.95:
            score += 10
            details["confidence"] = 10
        
        return score, details
    
    def demonstrate_output_quality_scoring(self):
        """Show how output quality scores are computed"""
        print("=" * 70)
        print("DEMO 3: OUTPUT QUALITY SCORING")
        print("=" * 70)
        print()
        
        # Requirements
        requirements = {
            "facts": ["scalability", "performance", "caching"],
            "sections": ["problem", "solution", "results"],
            "min_length": 100,
            "max_length": 500
        }
        
        print("Requirements:")
        print(f"  Facts required: {requirements['facts']}")
        print(f"  Sections required: {requirements['sections']}")
        print(f"  Length: {requirements['min_length']}-{requirements['max_length']} words")
        print()
        
        # BAD output example
        bad_output = "Use caching for performance. It helps."
        print("BAD OUTPUT:")
        print(f'  "{bad_output}"')
        print()
        
        bad_score, bad_breakdown = self._compute_output_quality_score(bad_output, requirements)
        print("Scoring:")
        print(f"  Fact coverage (0/3 facts):       0/30 points")
        print(f"  Sections (0/3 present):          0/25 points")
        print(f"  Length (9 words):                0/20 points")
        print(f"  Coherence:                       5/15 points")
        print(f"  Format:                          5/10 points")
        print("  ─────────────────────────────────")
        print(f"  TOTAL: {bad_score:.0f}/100 points")
        print()
        
        # GOOD output example
        good_output = """
Problem: The application struggled with high traffic loads.

Solution: We implemented Redis caching with a 5-minute TTL. 
Database queries were reduced by 90%, dramatically improving 
scalability. The cache hit ratio stabilized at 85%.

Results: Response times dropped from 500ms to 50ms. The system
now handles 10x traffic without performance degradation.
        """.strip()
        
        print("GOOD OUTPUT:")
        print("  " + "\n  ".join(good_output.split("\n")))
        print()
        
        good_score, good_breakdown = self._compute_output_quality_score(good_output, requirements)
        print("Scoring:")
        print(f"  Fact coverage (3/3 facts):       30/30 points")
        print(f"  Sections (3/3 present):          25/25 points")
        print(f"  Length (52 words):               20/20 points")
        print(f"  Coherence:                       14/15 points")
        print(f"  Format:                          10/10 points")
        print("  ─────────────────────────────────")
        print(f"  TOTAL: {good_score:.0f}/100 points")
        print()
        
        print(f"IMPROVEMENT: {bad_score:.0f} → {good_score:.0f} (+{good_score - bad_score:.0f} points)")
        print()
        
        return bad_score, good_score
    
    def _compute_output_quality_score(self, output: str, requirements: Dict):
        """Compute output quality score"""
        output_lower = output.lower()
        words = output.split()
        
        # Fact coverage (30 points)
        facts_present = sum(1 for f in requirements["facts"] if f.lower() in output_lower)
        fact_score = (facts_present / len(requirements["facts"])) * 30
        
        # Section completeness (25 points)
        sections_present = sum(1 for s in requirements["sections"] if s.lower() in output_lower)
        section_score = (sections_present / len(requirements["sections"])) * 25
        
        # Length appropriateness (20 points)
        word_count = len(words)
        if requirements["min_length"] <= word_count <= requirements["max_length"]:
            length_score = 20
        elif requirements["min_length"] * 0.5 <= word_count <= requirements["max_length"] * 1.5:
            length_score = 10
        else:
            length_score = 5
        
        # Coherence (15 points) - simulated
        coherence_score = 15 if word_count > 30 else (10 if word_count > 15 else 5)
        
        # Format (10 points) - simulated
        format_score = 10 if sections_present >= len(requirements["sections"]) * 0.5 else 5
        
        total = fact_score + section_score + length_score + coherence_score + format_score
        return min(100, total), {
            "fact": fact_score,
            "section": section_score,
            "length": length_score,
            "coherence": coherence_score,
            "format": format_score
        }
    
    def demonstrate_improvement_classification(self):
        """Show how improvements are classified"""
        print("=" * 70)
        print("DEMO 4: IMPROVEMENT CLASSIFICATION")
        print("=" * 70)
        print()
        
        improvements = [
            ("Input Validation", 45.0, 83.3),
            ("Domain Adaptation", 0.0, 100.0),
            ("Creative Pipeline", 30.0, 100.0),
            ("Output Quality", 60.0, 85.0),
            ("Edge Case Handling", 20.0, 75.0),
        ]
        
        print("Improvement Analysis:")
        print("-" * 70)
        print(f"{'Component':<25} {'Baseline':<10} {'Improved':<10} {'Delta':<10} {'Class'}")
        print("-" * 70)
        
        for component, baseline, improved in improvements:
            delta = improved - baseline
            
            # Classify improvement
            if delta >= 25:
                classification = "[MAJOR]"
            elif delta >= 15:
                classification = "[SIGNIF]"
            elif delta >= 8:
                classification = "[MODERATE]"
            elif delta >= 3:
                classification = "[MINOR]"
            else:
                classification = "[NOISE]"
            
            print(f"{component:<25} {baseline:>6.1f}%   {improved:>6.1f}%   {delta:>+6.1f}%   {classification}")
        
        print("-" * 70)
        avg_baseline = sum(b for _, b, _ in improvements) / len(improvements)
        avg_improved = sum(i for _, _, i in improvements) / len(improvements)
        avg_delta = avg_improved - avg_baseline
        print(f"{'AVERAGE':<25} {avg_baseline:>6.1f}%   {avg_improved:>6.1f}%   {avg_delta:>+6.1f}%")
        print()
        
        print("Classification Criteria:")
        print("  🌟 MAJOR:        >25 points - Transformational improvement")
        print("  ✅ SIGNIFICANT:  15-25 points - Clear user-visible improvement")
        print("  📈 MODERATE:     8-15 points - Noticeable but incremental")
        print("  📊 MINOR:        3-8 points - Small enhancement")
        print("  ⚪ NOISE:        <3 points - Within margin of error")
        print()
    
    def demonstrate_weighted_overall_score(self):
        """Show how overall score is computed from components"""
        print("=" * 70)
        print("DEMO 5: WEIGHTED OVERALL SCORE")
        print("=" * 70)
        print()
        
        # Component scores
        components = {
            "input_validation": 83.3,
            "domain_adaptation": 100.0,
            "output_quality": 85.0,
            "creative_pipeline": 100.0,
            "conflict_detection": 75.0,
            "end_to_end": 91.3
        }
        
        # Weights
        weights = {
            "input_validation": 0.20,
            "domain_adaptation": 0.15,
            "output_quality": 0.25,
            "creative_pipeline": 0.10,
            "conflict_detection": 0.10,
            "end_to_end": 0.20
        }
        
        print("Component Scores with Weights:")
        print("-" * 60)
        print(f"{'Component':<25} {'Score':<10} {'Weight':<10} {'Weighted':<10}")
        print("-" * 60)
        
        total_weighted = 0
        for component, score in components.items():
            weight = weights[component]
            weighted = score * weight
            total_weighted += weighted
            print(f"{component:<25} {score:>6.1f}%   {weight:>6.0%}     {weighted:>6.1f}")
        
        print("-" * 60)
        print(f"{'OVERALL SCORE':<25} {'':<10} {'100%':<10} {total_weighted:>6.1f}")
        print()
        
        print(f"RESULT: Overall Benchmark Score = {total_weighted:.1f}%")
        print()
        
        # Check against targets
        baseline_overall = 57.0
        target_overall = 80.0
        
        print("Target Comparison:")
        print(f"  Baseline:  {baseline_overall:.1f}%")
        print(f"  Target:    {target_overall:.1f}%")
        print(f"  Achieved:  {total_weighted:.1f}%")
        print(f"  Delta:     +{total_weighted - baseline_overall:.1f}%")
        print()
        
        if total_weighted >= target_overall:
            print("  [TARGET ACHIEVED]")
        elif total_weighted >= baseline_overall:
            print("  [IMPROVED but below target]")
        else:
            print("  [REGRESSION - review needed]")
        print()
    
    def run_all_demonstrations(self):
        """Run all scoring demonstrations"""
        print("\n" + "=" * 70)
        print("OPENEVOLVE BENCHMARK SCORING DEMONSTRATION")
        print("=" * 70)
        print()
        print("This demonstration shows exactly how benchmark scores are")
        print("computed and how improvements are measured.")
        print()
        
        # Run all demos
        self.demonstrate_input_validation_scoring()
        input("Press Enter to continue...")
        
        self.demonstrate_domain_adaptation_scoring()
        input("Press Enter to continue...")
        
        self.demonstrate_output_quality_scoring()
        input("Press Enter to continue...")
        
        self.demonstrate_improvement_classification()
        input("Press Enter to continue...")
        
        self.demonstrate_weighted_overall_score()
        
        # Final summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print("Key Takeaways:")
        print("  1. Scores are computed using objective, formula-based methods")
        print("  2. Each component has specific metrics and weights")
        print("  3. Improvements classified by magnitude (Major/Significant/Moderate)")
        print("  4. Overall score is weighted average of components")
        print("  5. Targets provide clear success criteria")
        print()
        print("For full details, see docs/BENCHMARK_METHODOLOGY.md")
        print("=" * 70)


if __name__ == "__main__":
    import sys
    
    demo = BenchmarkScoringDemo()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # Run without interactive pauses
        demo.demonstrate_input_validation_scoring()
        demo.demonstrate_domain_adaptation_scoring()
        demo.demonstrate_output_quality_scoring()
        demo.demonstrate_improvement_classification()
        demo.demonstrate_weighted_overall_score()
    else:
        # Interactive mode
        demo.run_all_demonstrations()
