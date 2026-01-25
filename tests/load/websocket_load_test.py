"""
WebSocket load testing script
"""
import asyncio
import websockets
import time
import statistics
from typing import List
import json


async def test_single_websocket_connection(websocket_id: int, duration: int = 60):
    """
    Test a single WebSocket connection

    Args:
        websocket_id: Unique identifier for this connection
        duration: Test duration in seconds
    """
    messages_sent = 0
    messages_received = 0
    latencies = []

    try:
        async with websockets.connect(
            f"ws://localhost:8000/ws/evolution/test-{websocket_id}?user_id=test-user-{websocket_id}"
        ) as websocket:
            start_time = time.time()
            end_time = start_time + duration

            while time.time() < end_time:
                # Send a message
                send_time = time.time()
                await websocket.send_json({
                    "type": "test",
                    "data": {"message_id": messages_sent},
                })
                messages_sent += 1

                # Receive response
                response = await websocket.recv()
                receive_time = time.time()
                latency = (receive_time - send_time) * 1000  # Convert to ms
                latencies.append(latency)
                messages_received += 1

                # Small delay before next message
                await asyncio.sleep(0.1)

            return {
                "websocket_id": websocket_id,
                "messages_sent": messages_sent,
                "messages_received": messages_received,
                "avg_latency": statistics.mean(latencies) if latencies else 0,
                "min_latency": min(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0,
                "success": True,
            }

    except Exception as e:
        return {
            "websocket_id": websocket_id,
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "error": str(e),
            "success": False,
        }


async def test_websocket_concurrency(num_connections: int, duration: int = 60):
    """
    Test multiple concurrent WebSocket connections

    Args:
        num_connections: Number of concurrent connections
        duration: Test duration in seconds
    """
    print(f"\nTesting {num_connections} concurrent WebSocket connections...")
    print(f"Duration: {duration} seconds\n")

    start_time = time.time()

    # Create all connections concurrently
    tasks = [
        test_single_websocket_connection(i, duration)
        for i in range(num_connections)
    ]

    # Wait for all connections to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.time()
    total_duration = end_time - start_time

    # Calculate statistics
    successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed_results = [r for r in results if (isinstance(r, dict) and not r.get("success")) or isinstance(r, Exception)]

    total_messages_sent = sum(r.get("messages_sent", 0) for r in successful_results)
    total_messages_received = sum(r.get("messages_received", 0) for r in successful_results)

    all_latencies = [
        r.get("avg_latency")
        for r in successful_results
        if r.get("avg_latency") > 0
    ]

    print("\n=== WebSocket Load Test Results ===")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print(f"Concurrent Connections: {num_connections}")
    print(f"Successful Connections: {len(successful_results)}")
    print(f"Failed Connections: {len(failed_results)}")
    print(f"Total Messages Sent: {total_messages_sent}")
    print(f"Total Messages Received: {total_messages_received}")
    print(f"Messages per Second: {total_messages_sent / total_duration:.2f}")

    if all_latencies:
        print(f"\nLatency Statistics:")
        print(f"Average: {statistics.mean(all_latencies):.2f} ms")
        print(f"Median: {statistics.median(all_latencies):.2f} ms")
        print(f"Min: {min(all_latencies):.2f} ms")
        print(f"Max: {max(all_latencies):.2f} ms")

    if failed_results:
        print(f"\nFailed Connections:")
        for i, result in enumerate(failed_results[:5]):  # Show first 5 failures
            if isinstance(result, dict):
                print(f"  - Connection {result.get('websocket_id')}: {result.get('error', 'Unknown error')}")
            else:
                print(f"  - Exception: {str(result)}")

    print("\n" + "=" * 50 + "\n")

    return {
        "total_duration": total_duration,
        "num_connections": num_connections,
        "successful_connections": len(successful_results),
        "failed_connections": len(failed_results),
        "total_messages_sent": total_messages_sent,
        "total_messages_received": total_messages_received,
        "messages_per_second": total_messages_sent / total_duration,
        "avg_latency": statistics.mean(all_latencies) if all_latencies else 0,
    }


async def test_websocket_stress_test():
    """
    Run a stress test with increasing number of connections
    """
    print("\n=== WebSocket Stress Test ===\n")

    connection_counts = [10, 50, 100, 200, 500]
    results = []

    for count in connection_counts:
        result = await test_websocket_concurrency(count, duration=30)
        results.append(result)

        # Delay between tests
        await asyncio.sleep(5)

    print("\n=== Stress Test Summary ===")
    print(f"{'Connections':<15} {'Success Rate':<15} {'Msgs/sec':<15} {'Avg Latency (ms)':<15}")
    print("-" * 60)

    for result in results:
        success_rate = (result["successful_connections"] / result["num_connections"]) * 100
        print(f"{result['num_connections']:<15} {success_rate:<15.2f} {result['messages_per_second']:<15.2f} {result['avg_latency']:<15.2f}")


if __name__ == "__main__":
    # Run concurrency test
    asyncio.run(test_websocket_concurrency(num_connections=100, duration=60))

    # Run stress test
    # asyncio.run(test_websocket_stress_test())
