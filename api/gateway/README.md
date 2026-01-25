# OpenEvolve API Gateway

FastAPI-based REST API and WebSocket Gateway for the OpenEvolve Backend Engines.

## Overview

The API Gateway serves as the bridge between the BubbleLab React/TypeScript frontend and the Python backend engines (Evolution, Adversarial Testing, Analytics, etc.). It handles:

- REST API endpoints for all backend operations
- WebSocket connections for real-time updates
- JWT authentication and authorization
- Rate limiting and request validation
- Error handling and response formatting

## Architecture

```
┌─────────────────┐
│  BubbleLab UI   │
│  (React/TS)     │
└────────┬────────┘
         │ REST + WebSocket
         ▼
┌─────────────────────────────────────┐
│         API Gateway                 │
│  ┌──────────────────────────────┐  │
│  │  Authentication (JWT)        │  │
│  │  Rate Limiting              │  │
│  │  Request Validation         │  │
│  │  Response Formatting        │  │
│  └──────────────────────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      Backend Engines               │
│  ┌──────────────────────────────┐  │
│  │  Evolution Engine            │  │
│  │  Adversarial Engine          │  │
│  │  Analytics Engine            │  │
│  │  Knowledge Engine            │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Features

### Authentication
- JWT-based authentication
- User registration and login
- Token refresh mechanism
- Password hashing with bcrypt
- Optional Clerk JWT verification for BubbleLab UI tokens (RS256)

### Rate Limiting
- Configurable per-user rate limits
- Redis-based or in-memory storage
- Custom limits per endpoint

### WebSocket Channels
- `/ws/evolution/{id}` - Evolution progress updates
- `/ws/adversarial/{id}` - Adversarial testing updates
- `/ws/workflow/{id}` - Workflow execution updates
- `/ws/collaboration/{room}` - Real-time collaboration
- `/ws/monitoring` - System monitoring

### API Endpoints
- **Authentication**: `/api/v1/auth/*`
- **Evolution**: `/api/v1/evolution/*`
- **Adversarial**: `/api/v1/adversarial/*`
- **Analytics**: `/api/v1/analytics/*`
- **Content**: `/api/v1/content/*`
- **Collaboration**: `/api/v1/collaboration/*`

## Installation

### Prerequisites
- Python 3.9+
- pip or poetry

### Setup

1. **Clone the repository**
   ```bash
   cd api/gateway
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the server**
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Server host | `0.0.0.0` |
| `API_PORT` | Server port | `8000` |
| `SECRET_KEY` | JWT secret key | (auto-generated) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration | `30` |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:3000"]` |
| `RATE_LIMIT_PER_MINUTE` | Rate limit | `100` |
| `REDIS_URL` | Redis URL | `memory://` |
| `CLERK_ISSUER` | Clerk issuer URL | (empty) |
| `CLERK_JWKS_URL` | Clerk JWKS URL override | (empty) |
| `CLERK_AUDIENCE` | Clerk token audience | (empty) |
| `CLERK_JWKS_CACHE_TTL_SECONDS` | JWKS cache TTL (seconds) | `3600` |
| `EVOLUTION_ORCHESTRATOR_URL` | Evolution orchestrator endpoint | `http://localhost:8003/evolve` |

### Rate Limiting

Rate limiting is enabled by default and configured via environment variables:

```env
RATE_LIMIT_ENABLED=True
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST=10
```

Disable rate limiting for development:
```env
RATE_LIMIT_ENABLED=False
```

## API Usage

### Authentication

#### Register User
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123",
    "username": "johndoe",
    "full_name": "John Doe"
  }'
```

#### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Evolution Engine

#### Start Evolution
```bash
curl -X POST http://localhost:8000/api/v1/evolution/start \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "def hello():\n    print(\"Hello World\")",
    "mode": "standard",
    "parameters": {
      "max_iterations": 100,
      "population_size": 50,
      "temperature": 0.7
    },
    "models": [{
      "provider": "openai",
      "model": "gpt-4",
      "api_key": "sk-..."
  }]
  }'
```
If you are using BubbleLab with Clerk, you can pass the Clerk session token
as the `Authorization` bearer token once `CLERK_ISSUER` (or `CLERK_JWKS_URL`)
is configured in the gateway environment.

#### Get Evolution Status
```bash
curl -X GET http://localhost:8000/api/v1/evolution/{evolution_id} \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

#### List Evolutions
```bash
curl -X GET "http://localhost:8000/api/v1/evolution?status=running&limit=20" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### WebSocket Usage

#### JavaScript/TypeScript (BubbleLab Frontend)

```typescript
const ws = new WebSocket('ws://localhost:8000/ws/evolution/{evolution_id}?user_id={user_id}');

ws.onopen = () => {
  console.log('WebSocket connected');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);

  switch (message.type) {
    case 'progress_update':
      updateProgressBar(message.data.progress);
      break;
    case 'generation_complete':
      displayGeneration(message.data.generation);
      break;
    case 'complete':
      showResults(message.data.result);
      break;
    case 'error':
      showError(message.data.error);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket disconnected');
};

// Send message to server
ws.send(JSON.stringify({
  type: 'ping',
  data: {}
}));
```

#### Python Testing

```python
import asyncio
import websockets
import json

async def test_evolution_websocket():
    uri = "ws://localhost:8000/ws/evolution/{evolution_id}"
    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({
            "type": "ping",
            "data": {}
        }))

        # Receive messages
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received: {data}")

asyncio.run(test_evolution_websocket())
```

## Development

### Project Structure

```
api/gateway/
├── main.py                    # FastAPI application entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
├── middleware/
│   ├── auth.py               # JWT authentication
│   ├── cors.py               # CORS configuration
│   └── rate_limit.py         # Rate limiting
├── routes/
│   ├── auth.py               # Authentication endpoints
│   ├── evolution.py          # Evolution endpoints
│   ├── adversarial.py        # Adversarial testing endpoints
│   └── ...                   # Other route modules
├── realtime/
│   ├── manager.py            # WebSocket connection manager
│   └── handlers/
│       ├── workflow.py       # Workflow WebSocket handlers
│       ├── evolution.py      # Evolution WebSocket handlers
│       └── ...
├── models/
│   └── schemas.py            # Pydantic models
├── utils/
│   ├── errors.py             # Error handling utilities
│   ├── responses.py          # Response formatting
│   └── validators.py         # Request validation
└── tests/
    ├── test_auth.py          # Authentication tests
    ├── test_evolution.py     # Evolution tests
    └── test_websocket.py     # WebSocket tests
```

### Adding New Endpoints

1. **Create Pydantic models** in `models/schemas.py`
2. **Create route file** in `routes/`
3. **Register router** in `main.py`
4. **Add tests** in `tests/`
5. **Update documentation**

Example:

```python
# routes/myfeature.py
from fastapi import APIRouter, Depends
from models.schemas import MyRequest, MyResponse
from middleware.auth import get_current_user

router = APIRouter(prefix="/myfeature", tags=["MyFeature"])

@router.post("/", response_model=MyResponse)
async def create_myfeature(
    data: MyRequest,
    user: dict = Depends(get_current_user)
):
    # Implementation
    return MyResponse(...)

# main.py
from routes import myfeature
app.include_router(myfeature.router, prefix="/api/v1")
```

### Adding WebSocket Channels

1. **Create room manager** in `realtime/manager.py`
2. **Add WebSocket endpoint** in `main.py`
3. **Test connection**

Example:

```python
# realtime/manager.py
class MyFeatureRoomManager(RoomManager):
    def __init__(self):
        super().__init__("myfeature")

# main.py
@app.websocket("/ws/myfeature/{resource_id}")
async def websocket_myfeature(websocket: WebSocket, resource_id: str):
    room = f"myfeature:{resource_id}"
    await manager.connect(websocket, room)

    try:
        while True:
            data = await websocket.receive_text()
            # Handle messages
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
```

## Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_auth.py

# Run with verbose output
pytest -v
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/locustfile.py --host=http://localhost:8000
```

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t openevolve-api .
docker run -p 8000:8000 --env-file .env openevolve-api
```

### Production Considerations

1. **Use a production ASGI server** (Gunicorn + Uvicorn)
2. **Enable HTTPS** with a reverse proxy (Nginx)
3. **Use Redis** for rate limiting and session storage
4. **Set up monitoring** (Prometheus, Grafana)
5. **Configure logging** to use a centralized log service
6. **Enable request tracing** (Jaeger, Zipkin)

## Troubleshooting

### Common Issues

**Issue**: Import errors for backend engines
**Solution**: Ensure backend engines are in Python path or create adapters

**Issue**: WebSocket connection drops
**Solution**: Increase heartbeat interval, check proxy timeout settings

**Issue**: Rate limiting too aggressive
**Solution**: Adjust `RATE_LIMIT_PER_MINUTE` in `.env`

**Issue**: CORS errors from frontend
**Solution**: Add frontend URL to `CORS_ORIGINS` in `.env`

## Performance

- Async/await throughout for non-blocking I/O
- Connection pooling for database requests
- Redis caching for frequently accessed data
- WebSocket message broadcasting optimization

## Security

- Passwords hashed with bcrypt
- JWT tokens with expiration
- Rate limiting to prevent abuse
- CORS configuration
- Security headers (CSP, HSTS, X-Frame-Options)
- Input validation and sanitization

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [OpenEvolve/Frontend/issues](https://github.com/OpenEvolve/Frontend/issues)
- Documentation: [OpenEvolve Wiki](https://github.com/OpenEvolve/Frontend/wiki)
- Email: support@openevolve.org
