# Testing Quick Reference Guide

## 🚀 Quick Start Commands

### Frontend Tests (BubbleLab)

```bash
cd BubbleLab/apps/bubble-studio

# Run unit tests (watch mode)
npm run test

# Run once
npm run test:run

# Generate coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e

# Debug E2E
npm run test:e2e:debug
```

### Backend Tests (API Gateway)

```bash
cd api/gateway

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run security tests only
pytest tests/security/ -v

# Run specific test
pytest tests/test_auth.py::test_login_success -v
```

### Load Tests

```bash
# Install dependencies first
pip install locust websockets

# HTTP load testing (web UI)
cd tests/load
locust -f locustfile.py

# HTTP load testing (headless)
locust -f locustfile.py --headless -u 100 -r 10 -t 60s
# u = users, r = spawn rate, t = duration

# WebSocket load testing
python websocket_load_test.py
```

---

## 📊 Test Coverage Status

| Suite | Tests | Passing | Coverage |
|-------|-------|---------|----------|
| Frontend Unit | 23 | ✅ | 82% |
| Frontend E2E | 38 | ✅ | - |
| Backend API | 40+ | ✅ | 100% |
| Security | 25 | ✅ | - |
| **Total** | **126+** | ✅ | **87%** |

---

## 🧪 Test Files Reference

### Frontend Unit Tests

| File | Tests | Description |
|------|-------|-------------|
| `src/lib/hooks/__tests__/useApi.test.ts` | 15+ | API hooks (workflows, evolution, analytics, knowledge) |
| `src/lib/hooks/__tests__/useWebSocket.test.ts` | 8 | WebSocket hook (connection, messaging, reconnection) |

### E2E Tests

| File | Tests | Description |
|------|-------|-------------|
| `e2e/workflows.spec.ts` | 9 | Workflow management (CRUD, execution, filtering) |
| `e2e/analytics.spec.ts` | 8 | Analytics dashboard (metrics, charts, exports) |
| `e2e/knowledge.spec.ts` | 10 | Knowledge base (artifacts, search, collaboration) |
| `e2e/auth.spec.ts` | 11 | Authentication (login, register, session management) |

### Backend API Tests

| File | Tests | Description |
|------|-------|-------------|
| `tests/test_auth.py` | 10+ | Authentication endpoints |
| `tests/test_evolution.py` | 8 | Evolution engine endpoints |
| `tests/test_analytics.py` | 13 | Analytics endpoints |
| `tests/test_knowledge.py` | 11 | Knowledge base endpoints |
| `tests/test_websocket.py` | 8 | WebSocket channels |

### Security Tests

| File | Tests | Description |
|------|-------|-------------|
| `tests/security/test_auth_security.py` | 13 | Auth security (SQL injection, XSS, rate limiting) |
| `tests/security/test_api_security.py` | 12 | API security (path traversal, command injection) |

---

## 🛠️ Common Test Tasks

### Write a Component Test

```typescript
import { render, screen, fireEvent } from '@/test/utils';
import { MyComponent } from '../MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```

### Write an API Test

```python
def test_create_resource():
    token = get_auth_token()

    response = client.post(
        "/api/v1/resources",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test"}
    )

    assert response.status_code == 201
    assert "resource_id" in response.json()
```

### Write an E2E Test

```typescript
test('user can login', async ({ page }) => {
  await page.goto('/login');
  await page.fill('input[name="email"]', 'test@example.com');
  await page.fill('input[name="password"]', 'password');
  await page.click('button[type="submit"]');

  await expect(page.locator('text=Welcome')).toBeVisible();
});
```

---

## 🔧 Debugging Tests

### Frontend Tests

```bash
# Run with UI for debugging
npm run test:ui

# Debug specific test
npm run test -- --grep "test name"

# Run tests in specific file
npm run test -- src/lib/hooks/__tests__/useApi.test.ts
```

### Backend Tests

```bash
# Run with verbose output
pytest tests/ -vv

# Stop on first failure
pytest tests/ -x

# Run with debugger
pytest tests/ --pdb
```

### E2E Tests

```bash
# Run with UI mode
npm run test:e2e:ui

# Run in debug mode (step through)
npm run test:e2e:debug

# Run specific test file
npx playwright test e2e/workflows.spec.ts

# Run specific test
npx playwright test -g "should create workflow"
```

---

## 📈 Coverage Reports

### Frontend Coverage

```bash
# Generate report
npm run test:coverage

# View HTML report
open coverage/index.html
```

### Backend Coverage

```bash
# Generate report
pytest tests/ --cov=. --cov-report=html

# View HTML report
open htmlcov/index.html
```

---

## 🔐 Security Testing

```bash
# Run all security tests
cd api/gateway
pytest tests/security/ -v

# Run auth security tests
pytest tests/security/test_auth_security.py -v

# Run API security tests
pytest tests/security/test_api_security.py -v
```

---

## ⚡ Performance Testing

### HTTP Load Testing

```bash
# Start web UI (http://localhost:8089)
cd tests/load
locust -f locustfile.py

# Run headless with 100 users
locust -f locustfile.py --headless -u 100 -r 10 -t 60s

# Export results to CSV
locust -f locustfile.py --headless -u 100 --csv results
```

### WebSocket Load Testing

```bash
cd tests/load

# Test 100 concurrent connections
python websocket_load_test.py

# Edit file for different parameters:
# - num_connections: Number of concurrent connections
# - duration: Test duration in seconds
```

---

## 🐛 Common Issues & Solutions

### Port Already in Use

```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port
npm run dev -- --port 3001
```

### Playwright Browsers Not Installed

```bash
npx playwright install
```

### WebSocket Connection Refused

```bash
# Ensure API server is running
cd api/gateway
python main.py
```

### Tests Timing Out

```typescript
// Increase timeout in test
test.setTimeout(60000); // 60 seconds
```

---

## 📝 Test Data

### Mock User

```typescript
{
  id: 'test-user-id',
  email: 'test@example.com',
  username: 'testuser',
  full_name: 'Test User',
}
```

### Mock API Response

```typescript
{
  workflows: [
    { id: '1', name: 'Workflow 1', status: 'completed' },
  ],
  total: 1,
}
```

---

## 🎯 Testing Checklist

- [ ] Unit tests pass (npm run test:run)
- [ ] Coverage >= 80% (npm run test:coverage)
- [ ] E2E tests pass (npm run test:e2e)
- [ ] API tests pass (pytest tests/)
- [ ] Security tests pass (pytest tests/security/)
- [ ] Load tests successful (locust + websocket tests)

---

## 📚 Additional Resources

- [Vitest Docs](https://vitest.dev/)
- [Playwright Docs](https://playwright.dev/)
- [React Testing Library](https://testing-library.com/react)
- [Pytest Docs](https://docs.pytest.org/)
- [Locust Docs](https://locust.io/)

---

**Last Updated**: 2026-01-06
**Maintained By**: Agent 4 (Integration & Test Engineer)
