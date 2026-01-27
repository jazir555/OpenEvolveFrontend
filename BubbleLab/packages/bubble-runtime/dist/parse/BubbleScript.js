import { analyze, resetIds } from '@bubblelab/ts-scope-manager';
import { parse } from '@typescript-eslint/typescript-estree';
import { BubbleParser } from '../extraction/BubbleParser';
import { normalizeBracelessControlFlow } from '../utils/normalize-control-flow';
export class BubbleScript {
    ast;
    scopeManager;
    parsingErrors = [];
    // Stores parsed bubble information with variable $id as key
    parsedBubbles;
    originalParsedBubbles;
    workflow;
    scriptVariables; // Maps Variable.$id to Variable
    variableLocations; // Maps Variable.$id to location
    instanceMethodsLocation;
    bubbleScript;
    bubbleFactory;
    currentBubbleScript;
    trigger;
    /**
     * Reparse the AST and bubbles after the script has been modified
     * This is necessary when the script text changes but we need updated bubble locations
     */
    reparseAST() {
        // Reset ID generator to ensure deterministic variable IDs
        resetIds();
        // Parse the modified script into a new AST
        this.ast = parse(this.currentBubbleScript, {
            range: true, // Required for scope-manager
            loc: true, // Location info for line numbers
            sourceType: 'module', // Treat as ES module
            ecmaVersion: 2022, // Modern JS/TS features
        });
        // Analyze scope to build variable dependency graph
        this.scopeManager = analyze(this.ast, {
            sourceType: 'module',
        });
        this.variableLocations = {};
        // Build variable mapping first
        this.scriptVariables = this.buildVariableMapping();
        // Parse bubble dependencies from AST using the provided factory and scope manager
        const bubbleParser = new BubbleParser(this.currentBubbleScript);
        const parseResult = bubbleParser.parseBubblesFromAST(this.bubbleFactory, this.ast, this.scopeManager);
        this.instanceMethodsLocation = parseResult.instanceMethodsLocation;
        this.parsedBubbles = parseResult.bubbles;
        this.workflow = parseResult.workflow;
        this.trigger = this.getBubbleTriggerEventType() ?? { type: 'webhook/http' };
    }
    /**
     * Find the matching original bubble for a given bubble.
     * Used to restore original locations for user-facing data.
     */
    findOriginalBubble(bubble, originalBubblesByKey, normalizedBubblesByKey) {
        // For cloned bubbles, look up by clonedFromVariableId
        if (bubble.clonedFromVariableId !== undefined) {
            return this.originalParsedBubbles[bubble.clonedFromVariableId];
        }
        const key = `${bubble.bubbleName}:${bubble.variableName}`;
        const originalCandidates = originalBubblesByKey.get(key) || [];
        // Single candidate: use it directly
        if (originalCandidates.length === 1) {
            return originalCandidates[0];
        }
        // Multiple candidates: try exact variableId match first
        const match = originalCandidates.find((c) => c.variableId === bubble.variableId);
        if (match)
            return match;
        // Fallback: match by declaration order (index within same-named bubbles)
        // Normalization preserves declaration order, so the Nth bubble with this name
        // in normalized should correspond to the Nth bubble in original
        const normalizedCandidates = normalizedBubblesByKey.get(key) || [];
        const indexInNormalized = normalizedCandidates.findIndex((c) => c.variableId === bubble.variableId);
        if (indexInNormalized >= 0 &&
            indexInNormalized < originalCandidates.length) {
            return originalCandidates[indexInNormalized];
        }
        return undefined;
    }
    constructor(bubbleScript, bubbleFactory) {
        // Reset ID generator to ensure deterministic variable IDs
        resetIds();
        // First, parse the ORIGINAL script to get correct line numbers for originalParsedBubbles
        // This ensures user-facing locations match what they see in their IDE
        const originalAst = parse(bubbleScript, {
            range: true,
            loc: true,
            sourceType: 'module',
            ecmaVersion: 2022,
        });
        const originalScopeManager = analyze(originalAst, {
            sourceType: 'module',
        });
        const originalBubbleParser = new BubbleParser(bubbleScript);
        const originalParseResult = originalBubbleParser.parseBubblesFromAST(bubbleFactory, originalAst, originalScopeManager);
        // Store original bubbles with correct locations from original script
        this.originalParsedBubbles = originalParseResult.bubbles;
        // Reset IDs again before parsing normalized script to ensure consistent IDs
        resetIds();
        // Normalize braceless control flow statements to prevent injection issues
        const normalizedScript = normalizeBracelessControlFlow(bubbleScript);
        // Parse the normalized bubble script into AST for working bubbles
        this.bubbleScript = normalizedScript;
        this.currentBubbleScript = normalizedScript;
        this.bubbleFactory = bubbleFactory;
        this.ast = parse(normalizedScript, {
            range: true, // Required for scope-manager
            loc: true, // Location info for line numbers
            sourceType: 'module', // Treat as ES module
            ecmaVersion: 2022, // Modern JS/TS features
        });
        // Analyze scope to build variable dependency graph
        this.scopeManager = analyze(this.ast, {
            sourceType: 'module',
        });
        this.variableLocations = {};
        // Build variable mapping first
        this.scriptVariables = this.buildVariableMapping();
        // Parse bubble dependencies from AST using the provided factory and scope manager
        const bubbleParser = new BubbleParser(normalizedScript);
        const parseResult = bubbleParser.parseBubblesFromAST(bubbleFactory, this.ast, this.scopeManager);
        this.parsedBubbles = parseResult.bubbles;
        this.workflow = parseResult.workflow;
        this.instanceMethodsLocation = parseResult.instanceMethodsLocation;
        this.trigger = this.getBubbleTriggerEventType() ?? { type: 'webhook/http' };
    }
    // getter for bubblescript (computed property)
    get bubblescript() {
        // Regenerate the script
        return this.currentBubbleScript;
    }
    /** Print script with line numbers in pretty readable format */
    showScript(message) {
        const lines = this.currentBubbleScript.split('\n');
        console.debug(`###### ${message} ######`);
        console.debug('------------Script--------------');
        console.debug(lines.map((line, index) => `${index + 1}: ${line}`).join('\n'));
        // Show bubble paramer location (just the basic info)
        console.debug('---------------------------------');
        console.debug('--------Bubble Locations---------');
        const bubbles = this.getParsedBubbles();
        for (const bubble of Object.values(bubbles)) {
            console.debug(`Bubble ${bubble.bubbleName} location: ${bubble.location.startLine}-${bubble.location.endLine}`);
        }
        // Print instance methods locations
        console.debug('Instance methods locations:');
        for (const [methodName, location] of Object.entries(this.instanceMethodsLocation)) {
            console.debug(`  ${methodName}: ${location.bodyStartLine}-${location.endLine} (invocations: ${location.invocationLines.join(', ')})`);
        }
        console.debug('---------------------------------');
        console.debug(`##################`);
    }
    /**
     * Get all variable names available at a specific line (excluding globals)
     * This is like setting a debugger breakpoint at that line
     */
    getVarsForLine(lineNumber) {
        // Find ALL scopes that contain this line (not just one)
        const containingScopes = this.getAllScopesContainingLine(lineNumber);
        if (containingScopes.length === 0) {
            return [];
        }
        // Collect variables from all containing scopes
        const allAccessibleVars = new Set();
        for (const scope of containingScopes) {
            // Add variables from this scope
            for (const variable of scope.variables) {
                allAccessibleVars.add(variable);
            }
            // Walk up the parent chain for this scope
            let parentScope = scope.upper;
            while (parentScope) {
                for (const variable of parentScope.variables) {
                    allAccessibleVars.add(variable);
                }
                parentScope = parentScope.upper;
            }
        }
        // Convert to array and filter
        const accessibleVars = Array.from(allAccessibleVars);
        // Filter out global/built-in variables AND variables declared after this line
        return accessibleVars
            .filter((variable) => !this.isGlobalVariable(variable))
            .filter((variable) => this.isVariableDeclaredBeforeLine(variable, lineNumber))
            .map((variable) => variable);
    }
    /**
     * Find ALL scopes that contain the given line number
     * This is crucial because variables can be in sibling scopes (like block + for)
     */
    getAllScopesContainingLine(lineNumber) {
        const containingScopes = [];
        for (const scope of this.scopeManager.scopes) {
            const scopeStart = scope.block.loc?.start.line || 0;
            const scopeEnd = scope.block.loc?.end.line || 0;
            // Check if line is within this scope
            if (lineNumber >= scopeStart && lineNumber <= scopeEnd) {
                containingScopes.push(scope);
            }
        }
        // Sort by specificity (smaller ranges first, then by type priority)
        return containingScopes.sort((a, b) => {
            const rangeA = (a.block.loc?.end.line || 0) - (a.block.loc?.start.line || 0);
            const rangeB = (b.block.loc?.end.line || 0) - (b.block.loc?.start.line || 0);
            if (rangeA !== rangeB) {
                return rangeA - rangeB; // Smaller range first
            }
            // Same range, prefer by type priority
            const scopePriority = {
                block: 5,
                for: 4,
                function: 3,
                module: 2,
                global: 1,
            };
            const priorityA = scopePriority[a.type] || 0;
            const priorityB = scopePriority[b.type] || 0;
            return priorityB - priorityA; // Higher priority first
        });
    }
    /**
     * Find the most specific scope that contains the given line number
     */
    findScopeForLine(lineNumber) {
        let targetScope = null;
        let smallestRange = Infinity;
        for (const scope of this.scopeManager.scopes) {
            const scopeStart = scope.block.loc?.start.line || 0;
            const scopeEnd = scope.block.loc?.end.line || 0;
            // Check if line is within this scope
            if (lineNumber >= scopeStart && lineNumber <= scopeEnd) {
                const scopeRange = scopeEnd - scopeStart;
                // Prefer module scope over global scope when they have same range
                const isPreferredScope = scope.type === 'module' && targetScope?.type === 'global';
                // Find the most specific (smallest) scope containing this line
                if (scopeRange < smallestRange || isPreferredScope) {
                    smallestRange = scopeRange;
                    targetScope = scope;
                }
            }
        }
        return targetScope;
    }
    /**
     * Get all variables accessible from a scope (including parent scopes)
     * This mimics how debugger shows variables from current scope + outer scopes
     */
    getAllAccessibleVariables(scope) {
        const variables = [];
        let currentScope = scope;
        // Walk up the scope chain (like debugger scope stack)
        while (currentScope) {
            variables.push(...currentScope.variables);
            currentScope = currentScope.upper; // Parent scope
        }
        return variables;
    }
    /**
     * Check if a variable is declared before a given line number
     * This ensures we only return variables that actually exist at the breakpoint
     */
    isVariableDeclaredBeforeLine(variable, lineNumber) {
        // Get the line where this variable is declared
        const declarations = variable.defs;
        if (!declarations || declarations.length === 0) {
            return true; // If no declaration info, assume it's available (like function params)
        }
        // Check if any declaration is at or before the target line
        return declarations.some((def) => {
            const declLine = def.node?.loc?.start?.line;
            return declLine !== undefined && declLine <= lineNumber;
        });
    }
    /**
     * Check if a variable is a global/built-in (filter these out)
     */
    isGlobalVariable(variable) {
        // Filter out TypeScript/JavaScript built-ins
        const globalNames = new Set([
            'console',
            'Array',
            'Object',
            'String',
            'Number',
            'Boolean',
            'Date',
            'Math',
            'JSON',
            'Promise',
            'Error',
            'Function',
            'Symbol',
            'Map',
            'Set',
            'WeakMap',
            'WeakSet',
            'Proxy',
            'Reflect',
            'Buffer',
            'process',
            'global',
            'require',
            '__dirname',
            '__filename',
            'module',
            'exports',
            // TypeScript globals
            'Intl',
            'SymbolConstructor',
            'ArrayConstructor',
            'MapConstructor',
            'SetConstructor',
            'PromiseConstructor',
            'ErrorConstructor',
            'RegExp',
            'PropertyKey',
            'PropertyDescriptor',
            'Partial',
            'Required',
            'Readonly',
            'Pick',
            'Record',
            'Exclude',
            'Extract',
            'Omit',
            'NonNullable',
        ]);
        return (globalNames.has(variable.name) ||
            variable.scope.type === 'global' ||
            variable.name.includes('Constructor') ||
            variable.name.includes('Array') ||
            variable.name.includes('Iterator') ||
            variable.name.startsWith('Disposable') ||
            variable.name.startsWith('Async') ||
            variable.name.includes('Decorator'));
    }
    /**
     * Debug method: Get detailed scope info for a line
     */
    getScopeInfoForLine(lineNumber) {
        const targetScope = this.findScopeForLine(lineNumber);
        if (!targetScope) {
            return null;
        }
        const scopeVars = targetScope.variables
            .filter((v) => !this.isGlobalVariable(v))
            .map((v) => v.name);
        const allVars = this.getAllAccessibleVariables(targetScope)
            .filter((v) => !this.isGlobalVariable(v))
            .map((v) => v.name);
        return {
            scopeType: targetScope.type,
            variables: scopeVars,
            allAccessible: allVars,
            lineRange: `${targetScope.block.loc?.start.line}-${targetScope.block.loc?.end.line}`,
        };
    }
    /**
     * Build a mapping of all user-defined variables with unique IDs
     * Also cross-references with parsed bubbles
     * Fills variableLocations
     */
    buildVariableMapping() {
        const variableMap = {};
        this.variableLocations = {};
        // Collect all user-defined variables from all scopes
        for (const scope of this.scopeManager.scopes) {
            for (const variable of scope.variables) {
                if (!this.isGlobalVariable(variable)) {
                    // Use the Variable's built-in $id as the key
                    variableMap[variable.$id] = variable;
                    // Extract location information from the variable's definition
                    const location = this.extractVariableLocation(variable);
                    if (location) {
                        this.variableLocations[variable.$id] = location;
                    }
                }
            }
        }
        return variableMap;
    }
    /**
     * Extract precise location (line and column) for a variable
     */
    extractVariableLocation(variable) {
        // Get the primary definition of the variable
        const primaryDef = variable.defs[0];
        if (!primaryDef?.node?.loc)
            return null;
        const loc = primaryDef.node.loc;
        return {
            startLine: loc.start.line,
            startCol: loc.start.column,
            endLine: loc.end.line,
            endCol: loc.end.column,
        };
    }
    /**
     * Get Variable object by its $id
     */
    getVariableById(id) {
        return this.scriptVariables[id];
    }
    /**
     * Get all user-defined variables with their $ids
     */
    getAllVariablesWithIds() {
        return { ...this.scriptVariables };
    }
    /**
     * Get all user-defined variables in the entire script
     */
    getAllUserVariables() {
        const allVars = new Set();
        for (const scope of this.scopeManager.scopes) {
            for (const variable of scope.variables) {
                if (!this.isGlobalVariable(variable)) {
                    allVars.add(variable.name);
                }
            }
        }
        return Array.from(allVars);
    }
    /**
     * Get the parsed AST (for debugging or further analysis)
     */
    getAST() {
        return this.ast;
    }
    getOriginalParsedBubbles() {
        return this.originalParsedBubbles;
    }
    /**
     * Get the scope manager (for advanced analysis)
     */
    getScopeManager() {
        return this.scopeManager;
    }
    /**
     * Get the parsed bubbles with NORMALIZED locations (for internal use like injection).
     * These locations match the normalized script, not the original user script.
     */
    getParsedBubblesRaw() {
        return this.parsedBubbles;
    }
    /**
     * Get the parsed bubbles with original line numbers restored.
     * This returns a COPY of current bubbles (with clones, workflow updates, etc.)
     * with locations matching the original script that users see in their IDE.
     * Use this for frontend/user-facing data.
     */
    getParsedBubbles() {
        // Deep copy to avoid modifying internal state needed for injection
        const bubblesCopy = JSON.parse(JSON.stringify(this.parsedBubbles));
        // Build lookup maps for matching bubbles
        const originalBubblesByKey = new Map();
        for (const bubble of Object.values(this.originalParsedBubbles)) {
            const key = `${bubble.bubbleName}:${bubble.variableName}`;
            const existing = originalBubblesByKey.get(key) || [];
            existing.push(bubble);
            originalBubblesByKey.set(key, existing);
        }
        const normalizedBubblesByKey = new Map();
        for (const bubble of Object.values(this.parsedBubbles)) {
            const key = `${bubble.bubbleName}:${bubble.variableName}`;
            const existing = normalizedBubblesByKey.get(key) || [];
            existing.push(bubble);
            normalizedBubblesByKey.set(key, existing);
        }
        // Restore original locations for each bubble
        for (const bubble of Object.values(bubblesCopy)) {
            const originalBubble = this.findOriginalBubble(bubble, originalBubblesByKey, normalizedBubblesByKey);
            if (!originalBubble)
                continue;
            // Restore bubble's own location
            if (originalBubble.location) {
                bubble.location = { ...originalBubble.location };
            }
            // Restore parameter locations
            for (const param of bubble.parameters) {
                const originalParam = originalBubble.parameters.find((p) => p.name === param.name);
                if (originalParam?.location) {
                    param.location = { ...originalParam.location };
                }
            }
        }
        return bubblesCopy;
    }
    /**
     * Get the hierarchical workflow structure
     */
    getWorkflow() {
        return this.workflow;
    }
    /**
     * Get the handle method location (start and end lines)
     */
    getHandleMethodLocation() {
        // Backward compatibility: return handle method from instanceMethodsLocation
        const handleMethod = this.instanceMethodsLocation['handle'];
        if (handleMethod) {
            return {
                startLine: handleMethod.startLine,
                endLine: handleMethod.endLine,
                definitionStartLine: handleMethod.definitionStartLine,
                bodyStartLine: handleMethod.bodyStartLine,
            };
        }
        return null;
    }
    getInstanceMethodLocation(methodName) {
        return this.instanceMethodsLocation[methodName] || null;
    }
    /**
     * Get location information for a variable by its $id
     */
    getVariableLocation(variableId) {
        return this.variableLocations[variableId] || null;
    }
    /**
     * Get all variable locations
     */
    getAllVariableLocations() {
        return { ...this.variableLocations };
    }
    resetBubbleScript() {
        this.currentBubbleScript = this.bubbleScript;
    }
    /** Reassign variable to another value and assign to the new bubble script and return the new bubble script */
    reassignVariable(variableId, newValue) {
        const variable = this.getVariableById(variableId);
        if (!variable) {
            throw new Error(`Variable with ID ${variableId} not found`);
        }
        const location = this.getVariableLocation(variableId);
        if (!location) {
            throw new Error(`Location for variable ${variable.name} (ID: ${variableId}) not found`);
        }
        // Split the current script into lines
        const lines = this.currentBubbleScript.split('\n');
        // Get the line content (convert from 1-based to 0-based indexing)
        const lineIndex = location.startLine - 1;
        const originalLine = lines[lineIndex];
        // Find the variable declaration pattern and replace its value
        // Handle different patterns: const/let/var varName = value
        const variablePattern = new RegExp(`(\\b(?:const|let|var)\\s+${this.escapeRegExp(variable.name)}\\s*=\\s*)([^;,\\n]+)`, 'g');
        if (variablePattern.test(originalLine)) {
            // Replace the value part
            const newLine = originalLine.replace(variablePattern, `$1${newValue}`);
            lines[lineIndex] = newLine;
        }
        else {
            // If pattern doesn't match, try simpler assignment pattern
            const assignmentPattern = new RegExp(`(\\b${this.escapeRegExp(variable.name)}\\s*=\\s*)([^;,\\n]+)`, 'g');
            if (assignmentPattern.test(originalLine)) {
                const newLine = originalLine.replace(assignmentPattern, `$1${newValue}`);
                lines[lineIndex] = newLine;
            }
            else {
                throw new Error(`Could not find variable assignment pattern for ${variable.name} on line ${location.startLine}`);
            }
        }
        // Update the current script and return it
        this.currentBubbleScript = lines.join('\n');
        return this.currentBubbleScript;
    }
    /** Inject lines of script at particular locations and return the new bubble script */
    injectLines(lines, lineNumber) {
        if (lineNumber < 1) {
            throw new Error('Line number must be 1 or greater');
        }
        // Split the current script into lines
        const scriptLines = this.currentBubbleScript.split('\n');
        // Convert from 1-based to 0-based indexing
        const insertIndex = lineNumber - 1;
        // Validate the line number
        if (insertIndex > scriptLines.length) {
            throw new Error(`Line number ${lineNumber} exceeds script length (${scriptLines.length} lines)`);
        }
        // Insert the new lines at the specified position
        scriptLines.splice(insertIndex, 0, ...lines);
        // Update the current script and return it
        this.currentBubbleScript = scriptLines.join('\n');
        return this.currentBubbleScript;
    }
    /**
     * Helper method to escape special regex characters in variable names
     */
    escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    /**
     * Build a JSON Schema object for the payload parameter of the top-level `handle` entrypoint.
     * Delegates to BubbleParser for the actual implementation.
     */
    getPayloadJsonSchema() {
        const bubbleParser = new BubbleParser(this.currentBubbleScript);
        const schema = bubbleParser.getPayloadJsonSchema(this.ast);
        return schema;
    }
    /**
     * Detect the BubbleTriggerEventRegistry key from the class extends generic.
     * Example: class X extends BubbleFlow<'slack/bot_mentioned'> {}
     * Returns the string key (e.g., 'slack/bot_mentioned') or null if not found.
     */
    getBubbleTriggerEventType() {
        for (const stmt of this.ast.body) {
            const tryClass = (cls) => {
                if (!cls)
                    return null;
                const superClass = cls.superClass;
                if (!superClass || superClass.type !== 'Identifier')
                    return null;
                if (superClass.name !== 'BubbleFlow')
                    return null;
                // Extract the event type from generic parameter
                const params = cls.superTypeParameters;
                const firstParam = params?.params?.[0];
                if (!firstParam)
                    return null;
                let eventType = null;
                if (firstParam.type === 'TSLiteralType' &&
                    firstParam.literal.type === 'Literal') {
                    const v = firstParam.literal.value;
                    eventType = typeof v === 'string' ? v : null;
                }
                if (!eventType)
                    return null;
                // Extract cronSchedule if this is a schedule/cron event
                let cronSchedule = undefined;
                if (eventType === 'schedule/cron') {
                    // Look for cronSchedule property in the class body
                    for (const member of cls.body.body) {
                        if (member.type === 'PropertyDefinition' &&
                            member.key.type === 'Identifier' &&
                            member.key.name === 'cronSchedule') {
                            // Extract the string literal value
                            if (member.value &&
                                member.value.type === 'Literal' &&
                                typeof member.value.value === 'string') {
                                cronSchedule = member.value.value;
                                break;
                            }
                        }
                    }
                }
                return {
                    type: eventType,
                    cronSchedule,
                };
            };
            if (stmt.type === 'ClassDeclaration') {
                const result = tryClass(stmt);
                if (result) {
                    return {
                        type: result.type,
                        cronSchedule: result.cronSchedule,
                    };
                }
            }
            if (stmt.type === 'ExportNamedDeclaration' &&
                stmt.declaration?.type === 'ClassDeclaration') {
                const result = tryClass(stmt.declaration);
                if (result) {
                    return {
                        type: result.type,
                        cronSchedule: result.cronSchedule,
                    };
                }
            }
        }
        // Fallback: simple regex over the source to catch extends BubbleFlow<'event/key'>
        const match = this.currentBubbleScript.match(/extends\s+BubbleFlow\s*<\s*(['"`])([^'"`]+)\1\s*>/m);
        if (match && typeof match[2] === 'string') {
            const eventType = match[2];
            // Try to extract cronSchedule via regex if it's a cron event
            let cronSchedule = undefined;
            if (eventType === 'schedule/cron') {
                const cronMatch = this.currentBubbleScript.match(/readonly\s+cronSchedule\s*=\s*['"`]([^'"`]+)['"`]/);
                if (cronMatch) {
                    cronSchedule = cronMatch[1];
                }
            }
            return { type: eventType, cronSchedule };
        }
        return null;
    }
}
//# sourceMappingURL=BubbleScript.js.map