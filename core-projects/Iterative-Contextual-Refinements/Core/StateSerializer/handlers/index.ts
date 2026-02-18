/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Handler Registry - Registers all mode state handlers on import
 * Includes upstream handlers + local custom handlers (MathSolver, GenerativeUI, React)
 */

import { registerModeHandler } from '../ModeStateHandler';

// Upstream handlers
import { deepthinkStateHandler } from './DeepthinkStateHandler';
import { agenticStateHandler } from './AgenticStateHandler';
import { contextualStateHandler } from './ContextualStateHandler';
import { adaptiveDeepthinkStateHandler } from './AdaptiveDeepthinkStateHandler';
import { websiteModeStateHandler } from './WebsiteModeStateHandler';

// Local custom handlers
import { mathsolverStateHandler } from './MathSolverStateHandler';
import { generativeUIStateHandler } from './GenerativeUIStateHandler';
import { reactStateHandler } from './ReactStateHandler';

// Auto-register all handlers on module import
registerModeHandler(deepthinkStateHandler);
registerModeHandler(agenticStateHandler);
registerModeHandler(contextualStateHandler);
registerModeHandler(adaptiveDeepthinkStateHandler);
registerModeHandler(websiteModeStateHandler);

// Register local custom handlers
registerModeHandler(mathsolverStateHandler);
registerModeHandler(generativeUIStateHandler);
registerModeHandler(reactStateHandler);
