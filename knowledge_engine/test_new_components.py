"""
Integration Test Suite for New Components

Tests all 5 architectural gaps:
1. Execution Sandbox (Security)
2. Vision-Language Monitor (Multimodality)
3. Browser Research Agent (Live Web)
4. Complexity Router (Latency Optimization)
5. Chronicle (Temporal Memory)
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_engine.sandbox import SandboxManager, SandboxType, SecurityPolicy
from knowledge_engine.vision import VisionLanguageMonitor, VLMProvider
from knowledge_engine.browser import BrowserResearchAgent
from knowledge_engine.router import ComplexityRouter, ModelTier
from knowledge_engine.chronicle import Chronicle, EpisodeType, ChronicleIntegration


async def test_sandbox():
    """Test Execution Sandbox"""
    print("\n[TEST] Execution Sandbox")
    print("-" * 50)
    
    sandbox = SandboxManager(
        preferred_sandbox=SandboxType.SUBPROCESS,
        auto_cleanup=True
    )
    
    # Test Python execution
    result = await sandbox.execute_python(
        code="print('Hello from sandbox')\nx = 2 + 2\nprint(f'Result: {x}')",
        policy=SecurityPolicy(max_execution_time=10)
    )
    
    print(f"  Success: {result.success}")
    print(f"  Exit code: {result.exit_code}")
    print(f"  Output: {result.stdout.strip()}")
    print(f"  Sandbox ID: {result.sandbox_id}")
    print(f"  Security: {result.security_report}")
    
    assert result.success, "Sandbox execution failed"
    assert 'Hello from sandbox' in result.stdout
    print("  [OK] Sandbox test passed")


async def test_vision_monitor():
    """Test Vision-Language Monitor"""
    print("\n[TEST] Vision-Language Monitor")
    print("-" * 50)
    
    vlm = VisionLanguageMonitor(
        provider=VLMProvider.MOCK,
        screenshot_dir="./test_screenshots"
    )
    
    # Create a test screenshot
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 300], fill='green', outline='black')
    draw.text((120, 200), "Test Node", fill='black')
    
    test_path = "./test_screenshots/test_canvas.png"
    os.makedirs("./test_screenshots", exist_ok=True)
    img.save(test_path)
    
    # Analyze
    analysis = await vlm.analyze_screenshot(
        screenshot_path=test_path,
        verification_prompt="Is there a green node on the canvas?"
    )
    
    print(f"  Success: {analysis.success}")
    print(f"  Description: {analysis.description[:100]}...")
    print(f"  Elements: {len(analysis.elements_detected)}")
    print(f"  Confidence: {analysis.confidence}")
    
    assert analysis.success, "VLM analysis failed"
    print("  [OK] Vision monitor test passed")
    
    # Cleanup
    os.remove(test_path)


async def test_browser_agent():
    """Test Browser Research Agent"""
    print("\n[TEST] Browser Research Agent")
    print("-" * 50)
    
    agent = BrowserResearchAgent(
        rate_limit_delay=0.1
    )
    
    # Test search (mock if no API)
    results = await agent.search(
        query="Python asyncio tutorial",
        sources=['stackoverflow'],
        max_results=3
    )
    
    print(f"  Search results: {len(results)}")
    for r in results[:2]:
        print(f"    - {r.title[:50]}... ({r.source})")
    
    print("  [OK] Browser agent test passed")
    
    await agent.close()


def test_complexity_router():
    """Test Complexity Router"""
    print("\n[TEST] Complexity Router")
    print("-" * 50)
    
    router = ComplexityRouter(use_caching=True)
    
    # Test trivial query
    decision = router.route("What time is it?")
    print(f"  Trivial: '{decision.query}' -> {decision.selected_tier.value}")
    assert decision.selected_tier == ModelTier.FAST, "Trivial should route to FAST"
    
    # Test complex query (should route to DEEP or CAPABLE)
    decision = router.route(
        "Analyze the causal structure of this dataset and provide a comprehensive report on the relationships between variables"
    )
    print(f"  Complex: '{decision.query[:50]}...' -> {decision.selected_tier.value}")
    assert decision.selected_tier in [ModelTier.DEEP, ModelTier.CAPABLE], "Complex should route to DEEP or CAPABLE"
    
    # Test moderate query
    decision = router.route("Compare the differences between Python and JavaScript")
    print(f"  Moderate: '{decision.query}' -> {decision.selected_tier.value}")
    
    print(f"  Complexity score: {decision.complexity_score:.2f}")
    print(f"  Reasoning: {decision.reasoning}")
    
    # Check stats
    stats = router.get_routing_stats()
    print(f"  Total routed: {stats['total_routed']}")
    
    print("  [OK] Complexity router test passed")


def test_chronicle():
    """Test Chronicle (Temporal Memory)"""
    print("\n[TEST] Chronicle - Temporal Episodic Memory")
    print("-" * 50)
    
    chronicle = Chronicle(
        storage_path="./test_chronicle",
        max_episodes=1000
    )
    
    # Record episodes
    ep1 = chronicle.record_episode(
        agent="BlueTeam",
        action="Attempted Z3 timeout fix",
        episode_type=EpisodeType.FAILURE,
        outcome="Still timing out",
        lesson_learned="Need to increase solver timeout",
        tags=["z3", "timeout"],
        session_id="session_001"
    )
    
    ep2 = chronicle.record_episode(
        agent="BlueTeam",
        action="Increased Z3 timeout to 60s",
        episode_type=EpisodeType.SUCCESS,
        outcome="Solved successfully",
        lesson_learned="Default timeout was too low",
        tags=["z3", "timeout", "fix"],
        session_id="session_001",
        related_episodes=[ep1.episode_id]
    )
    
    print(f"  Recorded episodes: {len(chronicle.episodes)}")
    
    # Query
    from knowledge_engine.chronicle.chronicle import ChronicleQuery
    query = ChronicleQuery(
        agent="BlueTeam",
        tags=["z3"],
        limit=10
    )
    results = chronicle.query(query)
    print(f"  Z3-related episodes: {len(results)}")
    
    # Check for similar actions
    tried_before, lesson, similar = chronicle.have_we_tried_this(
        "Z3 timeout fix",
        time_window=__import__('datetime').timedelta(hours=1)
    )
    print(f"  Have we tried Z3 fix before? {tried_before}")
    if lesson:
        print(f"  Lesson learned: {lesson}")
    
    # Get stats
    stats = chronicle.get_stats()
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  By type: {stats['by_type']}")
    
    assert stats['total_episodes'] == 2, "Should have 2 episodes"
    print("  [OK] Chronicle test passed")
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_chronicle", ignore_errors=True)


async def run_all_tests():
    """Run all component tests"""
    print("=" * 60)
    print("NEW COMPONENT INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Execution Sandbox
        await test_sandbox()
        
        # Test 2: Vision-Language Monitor
        await test_vision_monitor()
        
        # Test 3: Browser Research Agent
        await test_browser_agent()
        
        # Test 4: Complexity Router (sync)
        test_complexity_router()
        
        # Test 5: Chronicle (sync)
        test_chronicle()
        
        print("\n" + "=" * 60)
        print("*** ALL COMPONENT TESTS PASSED ***")
        print("=" * 60)
        print("\nSummary:")
        print("  [OK] Execution Sandbox - Secure code execution")
        print("  [OK] Vision-Language Monitor - Visual verification")
        print("  [OK] Browser Research Agent - Live web research")
        print("  [OK] Complexity Router - Latency optimization")
        print("  [OK] Chronicle - Temporal episodic memory")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
