import { OpenAPIHono } from '@hono/zod-openapi';
import {
  setupErrorHandler,
  validationErrorHook,
} from '../utils/error-handler.js';
import { mutateBatchRoute, mutateRoute } from '../schemas/evolution-mutate.js';
import { mutationApi } from '../services/evolution/mutation-api.js';

const app = new OpenAPIHono({
  defaultHook: validationErrorHook,
});
setupErrorHandler(app);

app.get('/health', (c) => c.json({ status: 'ok' }));

app.openapi(mutateRoute, async (c): Promise<any> => {
  const payload = c.req.valid('json');
  const result = await mutationApi.mutate<Record<string, unknown>>(payload);
  return c.json(result, 200);
});

app.openapi(mutateBatchRoute, async (c): Promise<any> => {
  const payload = c.req.valid('json');
  const result = await mutationApi.mutateBatch<Record<string, unknown>>(payload);
  return c.json(result, 200);
});

export default app;
