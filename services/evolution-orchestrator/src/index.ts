import { Hono, type Context } from 'hono';
import { z } from 'zod';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { EvolutionOrchestrator } from './orchestrator.js';
import type { EvolutionRunResponse } from './types.js';

const app = new Hono();
const orchestrator = new EvolutionOrchestrator();
const runs = new Map<string, { result: EvolutionRunResponse; createdAt: string }>();

app.onError((err, c) => {
  return c.json({ error: err.message || 'Unexpected error' }, 500);
});

orchestrator.events.on('event', (event) => {
  console.log(`[evolution] ${event.type}`, event.payload);
});

const evolutionRequestSchema = z.object({
  html: z.string().min(1),
  css: z.string().optional(),
  iterations: z.number().min(1).max(20).optional(),
  populationSize: z.number().min(1).max(50).optional(),
  criteria: z.string().optional(),
});

app.get('/health', (c) => c.json({ status: 'ok' }));

const handleEvolutionStart = async (c: Context) => {
  const payload = evolutionRequestSchema.parse(await c.req.json());
  const result = await orchestrator.runEvolution(payload);
  runs.set(result.runId, { result, createdAt: new Date().toISOString() });
  return c.json(result, 200);
};

app.post('/evolve', handleEvolutionStart);
app.post('/evolution', handleEvolutionStart);

app.get('/evolution/:id', (c) => {
  const runId = c.req.param('id');
  const stored = runs.get(runId);
  if (!stored) {
    return c.json({ error: 'Evolution run not found' }, 404);
  }
  return c.json(stored.result, 200);
});

app.delete('/evolution/:id', (c) => {
  const runId = c.req.param('id');
  if (!runs.has(runId)) {
    return c.json({ error: 'Evolution run not found' }, 404);
  }
  runs.delete(runId);
  return c.json({ status: 'deleted' }, 200);
});

const port = Number(process.env.PORT || 8003);
const server = createServer(app.fetch);
const io = new Server(server, {
  cors: {
    origin: '*',
  },
});

orchestrator.events.on('event', (event) => {
  io.emit('evolution_event', event);
});

server.listen(port, () => {
  console.log(`Evolution orchestrator running on ${port}`);
});
