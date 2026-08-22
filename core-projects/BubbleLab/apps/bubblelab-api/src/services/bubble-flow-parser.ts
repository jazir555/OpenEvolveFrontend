import * as ts from 'typescript';
import { getBubbleFactory } from './bubble-factory-instance.js';
import { BUBBLE_CREDENTIAL_OPTIONS } from '@bubblelab/shared-schemas';
import {
  CredentialType,
  BubbleParameter,
  ParsedBubble,
  BubbleParameterType,
  BubbleName,
  ParsedBubbleWithInfo,
  SYSTEM_CREDENTIALS,
} from '@bubblelab/shared-schemas';

// Re-export ParsedBubble for use in other modules
export type { ParsedBubble };

/**
 * Built-in JavaScript classes that should be allowed in BubbleFlow code
 * These are not bubble classes but are standard JavaScript constructors
 */
export const ALLOWED_BUILTIN_CLASSES = new Set<string>([
  // Standard JS objects
  'Date',
  'Array',
  'Object',
  'Set',
  'Map',
  'WeakSet',
  'WeakMap',

  // Numbers and Math
  'Number',
  'BigInt',
  'Math',

  // Strings and RegExp
  'String',
  'RegExp',

  // Errors
  'Error',
  'TypeError',
  'ReferenceError',
  'SyntaxError',
  'RangeError',

  // Promises and async
  'Promise',

  // JSON and URL
  'JSON',
  'URL',
  'URLSearchParams',

  // TypedArrays
  'Int8Array',
  'Uint8Array',
  'Uint8ClampedArray',
  'Int16Array',
  'Uint16Array',
  'Int32Array',
  'Uint32Array',
  'Float32Array',
  'Float64Array',
  'BigInt64Array',
  'BigUint64Array',
  'ArrayBuffer',
  'SharedArrayBuffer',
  'DataView',

  // Other common classes
  'Symbol',
  'Boolean',
  'FormData',
  'Headers',
  'Request',
  'Response',
]);

/**
 * Check if a credential type is system-managed
 * @param credType - The credential type to check
 * @returns true if the credential is system-managed
 */
export function isSystemCredential(credType: CredentialType): boolean {
  return SYSTEM_CREDENTIALS.has(credType);
}

export interface BubbleFlowParseResult {
  success: boolean;
  bubbles: Record<string, ParsedBubble>;
  errors?: string[];
  warnings?: string[];
}

/**
 * Parses a BubbleFlow TypeScript code and extracts all bubble instantiations
 * with their parameters using the BubbleRegistry for accurate mapping.
 */
export async function parseBubbleFlow(
  code: string
): Promise<BubbleFlowParseResult> {
  try {
    // First, try to create the source file and check for basic syntax errors
    const sourceFile = ts.createSourceFile(
      'temp.ts',
      code,
      ts.ScriptTarget.ES2022,
      true
    );

    // Check if there are any parse diagnostics
    const syntaxDiagnostics = (
      sourceFile as unknown as {
        parseDiagnostics?: ts.DiagnosticWithLocation[];
      }
    ).parseDiagnostics;
    if (syntaxDiagnostics && syntaxDiagnostics.length > 0) {
      const errors = syntaxDiagnostics.map((diagnostic) => {
        const message = ts.flattenDiagnosticMessageText(
          diagnostic.messageText,
          '\n'
        );
        return `Syntax error: ${message}`;
      });
      return {
        success: false,
        bubbles: {},
        errors,
      };
    }

    const bubbles: Record<string, ParsedBubble> = {};
    const errors: string[] = [];
    const warnings: string[] = [];
    const unregisteredClasses = new Set<string>();

    // Build reverse lookup map from class names to bubble names using BubbleRegistry
    const classNameToBubbleName = await buildClassNameLookup();

    function visit(node: ts.Node) {
      // Look for variable declarations with bubble instantiations
      if (ts.isVariableDeclaration(node) && node.initializer) {
        const variableName = node.name.getText(sourceFile);
        const bubbleInfo = extractBubbleFromExpression(
          node.initializer,
          sourceFile,
          classNameToBubbleName,
          unregisteredClasses
        );

        if (bubbleInfo) {
          bubbles[variableName] = {
            variableName,
            ...bubbleInfo,
          };
        }
      }

      // Look for expression statements with anonymous bubble instantiations
      if (ts.isExpressionStatement(node)) {
        const bubbleInfo = extractBubbleFromExpression(
          node.expression,
          sourceFile,
          classNameToBubbleName,
          unregisteredClasses
        );

        if (bubbleInfo) {
          // Generate a synthetic variable name for anonymous calls
          const syntheticName = `_anonymous_${bubbleInfo.className}_${Object.keys(bubbles).length}`;
          bubbles[syntheticName] = {
            variableName: syntheticName,
            ...bubbleInfo,
          };
        }
      }

      // Note: We now parse both variable declarations and anonymous expression statements
      // Method calls like 'await postgres.action()' are NOT parsed as they are usage, not instantiation

      ts.forEachChild(node, visit);
    }

    visit(sourceFile);

    // Generate errors for unregistered classes
    if (unregisteredClasses.size > 0) {
      const registeredClasses = Array.from(classNameToBubbleName.keys());
      for (const className of unregisteredClasses) {
        errors.push(
          `Class '${className}' is not registered in the bubble factory. ` +
            `Available classes: ${registeredClasses.join(', ')}`
        );
      }
    }

    return {
      success: errors.length === 0,
      bubbles,
      errors: errors.length > 0 ? errors : undefined,
      warnings: warnings.length > 0 ? warnings : undefined,
    };
  } catch (error) {
    return {
      success: false,
      bubbles: {},
      errors: [
        `Parse error: ${error instanceof Error ? error.message : String(error)}`,
      ],
    };
  }
}

async function buildClassNameLookup(): Promise<
  Map<string, { bubbleName: string; className: string }>
> {
  const lookup = new Map<string, { bubbleName: string; className: string }>();

  // Get all registered bubbles and their metadata
  const factory = await getBubbleFactory();
  const allBubbles = factory.getAll();
  const bubbleNames = factory.list();

  bubbleNames.forEach((bubbleName, index) => {
    const bubbleClass = allBubbles[index];
    if (bubbleClass) {
      // Extract the class name from the constructor
      const className = bubbleClass.name;
      lookup.set(className, { bubbleName, className });
    }
  });

  return lookup;
}

function extractBubbleFromExpression(
  expression: ts.Expression,
  sourceFile: ts.SourceFile,
  classNameLookup: Map<string, { bubbleName: string; className: string }>,
  unregisteredClasses: Set<string>
): Omit<ParsedBubble, 'variableName'> | null {
  // Handle await expressions only if they contain instantiations
  if (ts.isAwaitExpression(expression)) {
    const awaitResult = extractBubbleFromExpression(
      expression.expression,
      sourceFile,
      classNameLookup,
      unregisteredClasses
    );
    if (awaitResult) {
      awaitResult.hasAwait = true;
    }
    return awaitResult;
  }

  // Handle direct new expressions
  if (ts.isNewExpression(expression)) {
    const result = extractBubbleFromNewExpression(
      expression,
      sourceFile,
      classNameLookup,
      unregisteredClasses
    );
    if (result) {
      result.hasAwait = false;
      result.hasActionCall = false;
    }
    return result;
  }

  // Handle call expressions (like .action())
  if (
    ts.isCallExpression(expression) &&
    ts.isPropertyAccessExpression(expression.expression)
  ) {
    const propertyAccess = expression.expression;
    if (
      propertyAccess.name.text === 'action' &&
      ts.isNewExpression(propertyAccess.expression)
    ) {
      const result = extractBubbleFromNewExpression(
        propertyAccess.expression,
        sourceFile,
        classNameLookup,
        unregisteredClasses
      );
      if (result) {
        result.hasAwait = false;
        result.hasActionCall = true;
      }
      return result;
    }
  }

  return null;
}

// Function removed - we only parse instantiations in variable declarations

function extractBubbleFromNewExpression(
  newExpr: ts.NewExpression,
  sourceFile: ts.SourceFile,
  classNameLookup: Map<string, { bubbleName: string; className: string }>,
  unregisteredClasses: Set<string>
): Omit<ParsedBubble, 'variableName'> | null {
  if (!newExpr.expression || !ts.isIdentifier(newExpr.expression)) {
    return null;
  }

  const className = newExpr.expression.text;

  // Check if it's a built-in JavaScript class - if so, ignore it
  if (ALLOWED_BUILTIN_CLASSES.has(className)) {
    return null; // Built-in class, not a bubble - don't parse or report as error
  }

  // Look up the bubble info using the registry
  const bubbleInfo = classNameLookup.get(className);
  if (!bubbleInfo) {
    // Track unregistered classes for error reporting (excluding built-ins)
    unregisteredClasses.add(className);
    return null; // Not a registered bubble
  }

  const parameters: BubbleParameter[] = [];

  // Extract parameters from the constructor call
  if (newExpr.arguments && newExpr.arguments.length > 0) {
    const firstArg = newExpr.arguments[0];

    if (ts.isObjectLiteralExpression(firstArg)) {
      firstArg.properties.forEach((prop) => {
        if (ts.isPropertyAssignment(prop) && ts.isIdentifier(prop.name)) {
          const paramName = prop.name.text;
          const paramValue = extractParameterValue(
            prop.initializer,
            sourceFile
          );
          parameters.push({
            name: paramName,
            ...paramValue,
          });
        } else if (
          ts.isShorthandPropertyAssignment(prop) &&
          ts.isIdentifier(prop.name)
        ) {
          // Handle ES6 shorthand property assignments like { userQuestion } instead of { userQuestion: userQuestion }
          const paramName = prop.name.text;
          const paramValue = extractParameterValue(
            prop.name, // In shorthand assignments, the name IS the value expression
            sourceFile
          );
          parameters.push({
            name: paramName,
            ...paramValue,
          });
        }
      });
    }
  }

  return {
    bubbleName: bubbleInfo.bubbleName as BubbleName,
    className: bubbleInfo.className,
    parameters,
    hasAwait: false, // Will be set by calling functions
    hasActionCall: false, // Will be set by calling functions
  };
}

function extractParameterValue(
  expression: ts.Expression,
  sourceFile: ts.SourceFile
): { value: string; type: BubbleParameter['type'] } {
  const valueText = expression.getText(sourceFile);

  // Check if it's an environment variable access - need to handle non-null assertion
  if (ts.isNonNullExpression(expression)) {
    const innerExpression = expression.expression;
    if (ts.isPropertyAccessExpression(innerExpression)) {
      const fullText = innerExpression.getText(sourceFile);
      if (fullText.startsWith('process.env.')) {
        return { value: valueText, type: BubbleParameterType.ENV };
      }
    }
  }

  // Check direct property access (without non-null assertion)
  if (
    ts.isPropertyAccessExpression(expression) ||
    ts.isElementAccessExpression(expression)
  ) {
    const fullText = expression.getText(sourceFile);
    if (fullText.startsWith('process.env.')) {
      return { value: fullText, type: BubbleParameterType.ENV };
    }
  }

  // Handle TypeScript type assertions (as expressions)
  if (ts.isAsExpression(expression)) {
    return { value: valueText, type: BubbleParameterType.UNKNOWN };
  }

  // Check TypeScript syntax kinds for type detection
  if (ts.isStringLiteral(expression) || ts.isTemplateExpression(expression)) {
    return { value: valueText, type: BubbleParameterType.STRING };
  }

  if (ts.isNumericLiteral(expression)) {
    return { value: valueText, type: BubbleParameterType.NUMBER };
  }

  if (
    expression.kind === ts.SyntaxKind.TrueKeyword ||
    expression.kind === ts.SyntaxKind.FalseKeyword
  ) {
    return { value: valueText, type: BubbleParameterType.BOOLEAN };
  }

  if (ts.isArrayLiteralExpression(expression)) {
    return { value: valueText, type: BubbleParameterType.ARRAY };
  }

  if (ts.isObjectLiteralExpression(expression)) {
    return { value: valueText, type: BubbleParameterType.OBJECT };
  }

  return { value: valueText, type: BubbleParameterType.UNKNOWN };
}

/**
 * Reconstructs a BubbleFlow code string from parsed bubbles and parameters.
 */
export async function reconstructBubbleFlow(
  originalCode: string,
  bubbleParameters: Record<string, ParsedBubble>
): Promise<{ success: boolean; code?: string; errors?: string[] }> {
  try {
    const sourceFile = ts.createSourceFile(
      'temp.ts',
      originalCode,
      ts.ScriptTarget.ES2022,
      true
    );

    const errors: string[] = [];
    let modifiedCode = originalCode;
    const modifications: Array<{
      start: number;
      end: number;
      replacement: string;
    }> = [];
    const classNameLookup = await buildClassNameLookup();

    // Track which anonymous bubble parameters have been used for matching
    const usedAnonymousBubbles = new Set<string>();

    function visit(node: ts.Node) {
      // Look for variable declarations with bubble instantiations
      if (ts.isVariableDeclaration(node) && node.initializer) {
        const variableName = node.name.getText(sourceFile);
        const param_by_id = Object.values(bubbleParameters).find(
          (param) => param.variableName === variableName
        );
        const newBubbleParams = bubbleParameters[variableName] || param_by_id;
        if (newBubbleParams) {
          const bubbleInfo = extractBubbleFromExpression(
            node.initializer,
            sourceFile,
            classNameLookup,
            new Set() // Empty set for reconstruction, we don't track errors here
          );
          if (bubbleInfo) {
            // Validate bubble name matches
            if (bubbleInfo.bubbleName !== newBubbleParams.bubbleName) {
              errors.push(
                `Bubble name mismatch for variable '${variableName}': expected '${bubbleInfo.bubbleName}', got '${newBubbleParams.bubbleName}'`
              );
            } else {
              // Generate new bubble instantiation
              const newInstantiation =
                generateBubbleInstantiation(newBubbleParams);

              // Find the range of the original instantiation
              const start = node.initializer.getStart(sourceFile);
              const end = node.initializer.getEnd();

              modifications.push({
                start,
                end,
                replacement: newInstantiation,
              });
            }
          }
        }
      }

      // Look for expression statements with anonymous bubble instantiations
      if (ts.isExpressionStatement(node)) {
        const bubbleInfo = extractBubbleFromExpression(
          node.expression,
          sourceFile,
          classNameLookup,
          new Set()
        );

        if (bubbleInfo) {
          // Find matching anonymous bubble parameters by comparing bubble properties
          // and ensuring we use each anonymous bubble only once (in order)
          const expressionStart = node.expression.getStart(sourceFile);
          const expressionEnd = node.expression.getEnd();

          for (const [paramKey, bubbleParam] of Object.entries(
            bubbleParameters
          )) {
            if (
              (paramKey.startsWith('_anonymous_') || Number(paramKey) < 0) &&
              !usedAnonymousBubbles.has(paramKey) &&
              bubbleParam.bubbleName === bubbleInfo.bubbleName &&
              bubbleParam.className === bubbleInfo.className
            ) {
              // Mark this anonymous bubble as used
              usedAnonymousBubbles.add(paramKey);

              // Generate new bubble instantiation
              const newInstantiation = generateBubbleInstantiation(bubbleParam);

              modifications.push({
                start: expressionStart,
                end: expressionEnd,
                replacement: newInstantiation,
              });

              // Break after finding the first unused match
              break;
            }
          }
        }
      }

      ts.forEachChild(node, visit);
    }

    visit(sourceFile);

    // Return errors if any validation failed
    if (errors.length > 0) {
      return { success: false, errors };
    }

    // Apply modifications in reverse order to maintain positions
    modifications.reverse().forEach((mod) => {
      modifiedCode =
        modifiedCode.slice(0, mod.start) +
        mod.replacement +
        modifiedCode.slice(mod.end);
    });

    return { success: true, code: modifiedCode };
  } catch (error) {
    return {
      success: false,
      errors: [
        `Reconstruction error: ${error instanceof Error ? error.message : String(error)}`,
      ],
    };
  }
}

function generateBubbleInstantiation(bubble: ParsedBubble): string {
  const paramStrings = bubble.parameters.map((param) => {
    // Format the value based on its type
    let valueStr: string;

    if (typeof param.value === 'string') {
      // Check if the string already looks like a code literal (starts with quote, number, boolean keyword, etc.)
      const trimmed = param.value.trim();
      const looksLikeCode =
        trimmed.startsWith('"') ||
        trimmed.startsWith("'") ||
        trimmed.startsWith('`') ||
        trimmed.startsWith('{') ||
        trimmed.startsWith('[') ||
        trimmed === 'true' ||
        trimmed === 'false' ||
        trimmed === 'null' ||
        trimmed === 'undefined' ||
        /^\d/.test(trimmed); // Starts with a digit

      if (looksLikeCode) {
        // Already formatted as code, use as-is
        valueStr = param.value;
      } else if (param.type === BubbleParameterType.STRING) {
        // Raw string value that needs quotes
        valueStr = JSON.stringify(param.value);
      } else {
        // Other types, use as-is
        valueStr = param.value;
      }
    } else {
      // Non-string values (objects, numbers, etc.)
      valueStr = JSON.stringify(param.value);
    }

    return `${param.name}: ${valueStr}`;
  });

  const hasParams = bubble.parameters.length > 0;
  const paramsString = hasParams ? `{\n  ${paramStrings.join(',\n  ')}\n}` : '';

  // Reconstruct based on original pattern
  if (bubble.hasAwait && bubble.hasActionCall) {
    // Original: await new BubbleName({...}).action()
    return `await new ${bubble.className}(${paramsString}).action()`;
  } else if (bubble.hasAwait && !bubble.hasActionCall) {
    // Original: await new BubbleName({...})
    return `await new ${bubble.className}(${paramsString})`;
  } else if (!bubble.hasAwait && bubble.hasActionCall) {
    // Original: new BubbleName({...}).action()
    return `new ${bubble.className}(${paramsString}).action()`;
  } else {
    // Original: new BubbleName({...})
    return `new ${bubble.className}(${paramsString})`;
  }
}

/**
 * Validates bubble parameters against their schema from the BubbleRegistry
 */
export async function validateBubbleParameters(
  bubbleName: BubbleName,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _parameters: BubbleParameter[] // intentionally unused, reserved for future validation
): Promise<{ valid: boolean; errors?: string[] }> {
  const factory = await getBubbleFactory();
  const bubbleMetadata = factory.getMetadata(bubbleName);
  if (!bubbleMetadata) {
    return { valid: false, errors: [`Unknown bubble: ${bubbleName}`] };
  }

  // For now, return success - full schema validation would require
  // evaluating the parameter values in context
  return { valid: true };
}

/**
 * Gets available parameter schema for a bubble from the registry
 */
export async function getBubbleParameterSchema(
  bubbleName: BubbleName
): Promise<unknown> {
  const factory = await getBubbleFactory();
  return factory.getMetadata(bubbleName);
}

/**
 * Extracts required credential types from parsed bubble parameters
 * @param bubbleParameters - Parsed bubble parameters
 * @returns Record mapping bubble variable names to their required credential types (excluding system credentials)
 */
export function extractRequiredCredentials(
  bubbleParameters: Record<string, ParsedBubble>
): Record<string, CredentialType[]> {
  const requiredCredentials: Record<string, CredentialType[]> = {};

  // Iterate through each bubble and check its credential requirements
  for (const [bubbleName, bubble] of Object.entries(bubbleParameters)) {
    const allCredentialTypes = new Set<CredentialType>();

    // Get bubble-level credentials
    const credentialOptions =
      BUBBLE_CREDENTIAL_OPTIONS[
        bubble.bubbleName as keyof typeof BUBBLE_CREDENTIAL_OPTIONS
      ];
    if (credentialOptions && Array.isArray(credentialOptions)) {
      for (const credType of credentialOptions) {
        allCredentialTypes.add(credType);
      }
    }

    // For AI agent bubbles, also collect tool-level credential requirements
    if (bubble.bubbleName === 'ai-agent') {
      const toolCredentials = extractToolCredentials(bubble);
      for (const credType of toolCredentials) {
        allCredentialTypes.add(credType);
      }
    }

    // Filter out system credentials that users don't need to provide
    const userCredentials = Array.from(allCredentialTypes);

    //.filter(
    //   (credType) => !SYSTEM_CREDENTIALS.has(credType)
    // );

    // Only add the bubble if it has non-system credentials
    if (userCredentials.length > 0) {
      requiredCredentials[bubbleName] = userCredentials;
    }
  }

  return requiredCredentials;
}

/**
 * Extracts tool credential requirements from AI agent bubble parameters
 * @param bubble - The parsed bubble to extract tool requirements from
 * @returns Array of credential types required by the bubble's tools
 */
export function extractToolCredentials(bubble: ParsedBubble): CredentialType[] {
  if (bubble.bubbleName !== 'ai-agent') {
    return [];
  }

  const toolCredentials: Set<CredentialType> = new Set();

  // Find the tools parameter in the bubble
  const toolsParam = bubble.parameters.find((param) => param.name === 'tools');
  if (!toolsParam || typeof toolsParam.value !== 'string') {
    return [];
  }

  try {
    // Parse the tools array from the parameter value
    // The value can be either JSON or JavaScript array literal
    let toolsArray: Array<{ name: string; [key: string]: unknown }>;

    // First try to safely evaluate as JavaScript (for cases like [{"name": "web-search-tool"}])
    try {
      // Use Function constructor to safely evaluate the expression in isolation
      const safeEval = new Function('return ' + toolsParam.value);
      const evaluated = safeEval();

      if (Array.isArray(evaluated)) {
        toolsArray = evaluated;
      } else {
        // Single object, wrap in array
        toolsArray = [evaluated];
      }
    } catch {
      // Fallback to JSON.parse for cases where it's valid JSON
      if (toolsParam.value.startsWith('[')) {
        toolsArray = JSON.parse(toolsParam.value);
      } else {
        toolsArray = [JSON.parse(toolsParam.value)];
      }
    }

    // For each tool, get its credential requirements
    for (const tool of toolsArray) {
      if (!tool.name || typeof tool.name !== 'string') {
        continue;
      }

      const toolBubbleName = tool.name as BubbleName;
      const toolCredentialOptions = BUBBLE_CREDENTIAL_OPTIONS[toolBubbleName];

      if (toolCredentialOptions && Array.isArray(toolCredentialOptions)) {
        for (const credType of toolCredentialOptions) {
          toolCredentials.add(credType);
        }
      }
    }
  } catch (error) {
    // If we can't parse the tools parameter, silently ignore
    // This handles cases where the tools parameter contains complex TypeScript expressions
    console.debug(
      `Failed to parse tools parameter for credential extraction: ${error}`
    );
  }

  return Array.from(toolCredentials);
}

/**
 * Merges credentials from user selection into bubble parameters
 * @param bubbleParameters - The parsed bubble parameters
 * @param credentials - Credentials mapping: bubble name -> credential type -> credential ID
 * @returns Updated bubble parameters with credentials injected
 */
export function mergeCredentialsIntoBubbleParameters(
  bubbleParameters: Record<string | number, ParsedBubbleWithInfo>,
  credentials: Record<string | number, Record<string, number>>
): Record<number, ParsedBubbleWithInfo> {
  const updatedParameters = { ...bubbleParameters };

  // For each bubble that has credentials
  for (const [bubbleName, credMapping] of Object.entries(credentials)) {
    const bubble = updatedParameters[bubbleName];
    if (!bubble || !credMapping || Object.keys(credMapping).length === 0) {
      continue;
    }

    // Find existing credentials parameter or create new one
    let credentialsParam = bubble.parameters.find(
      (p) => p.name === 'credentials'
    );

    if (!credentialsParam) {
      // Create new credentials parameter
      credentialsParam = {
        name: 'credentials',
        value: {},
        type: BubbleParameterType.OBJECT,
      };
      bubble.parameters.push(credentialsParam);
    }

    // Ensure the value is an object
    if (
      typeof credentialsParam.value !== 'object' ||
      credentialsParam.value === null
    ) {
      credentialsParam.value = {};
    }

    // Merge the credential IDs into the credentials object
    const credentialsObj = credentialsParam.value as Record<string, number>;
    for (const [credType, credId] of Object.entries(credMapping)) {
      credentialsObj[credType] = credId;
    }

    credentialsParam.value = credentialsObj;
  }

  return updatedParameters;
}

/**
 * Decomposes bubble parameters into a workflow of interconnected service bubbles.
 * 
 * This function analyzes the bubble parameters and creates a decomposition based on:
 * 1. Bubble types and their natural execution order
 * 2. Data dependencies between bubbles
 * 3. Service categories (input, processing, output)
 * 
 * @param bubbleParameters - Record of parsed bubble parameters
 * @returns Record of decomposed service bubbles in execution order
 */
export function generateDisplayedBubbleParameters(
  bubbleParameters: Record<string, ParsedBubble>
): Record<string, ParsedBubble> {
  // If no parameters provided, return empty decomposition
  if (!bubbleParameters || Object.keys(bubbleParameters).length === 0) {
    return {};
  }

  // Categorize bubbles by their service type
  const categorizedBubbles = categorizeBubbles(bubbleParameters);
  
  // Generate decomposition based on categorization
  const decomposition: Record<string, ParsedBubble> = {};
  let stepIndex = 0;

  // Step 1: Input/Listener bubbles (event sources)
  for (const [, bubble] of categorizedBubbles.input) {
    const stepKey = `step-${String(stepIndex++).padStart(2, '0')}-${bubble.bubbleName}-listener`;
    decomposition[stepKey] = {
      ...bubble,
      parameters: bubble.parameters.map(p => ({
        ...p,
        value:
          typeof p.value === 'string' && p.value.startsWith('payload')
            ? p.value
            : `payload.${p.name}`,
      })),
      hasAwait: false,
      hasActionCall: false,
    };
  }

  // Step 2: Formatter/Parser bubbles (data transformation)
  for (const [, bubble] of categorizedBubbles.formatter) {
    const stepKey = `step-${String(stepIndex++).padStart(2, '0')}-${bubble.bubbleName}-parser`;
    decomposition[stepKey] = {
      ...bubble,
      parameters: generateDataFlowParameters(bubble, decomposition),
      hasAwait: true,
      hasActionCall: true,
    };
  }

  // Step 3: Database/Storage bubbles
  for (const [, bubble] of categorizedBubbles.database) {
    const stepKey = `step-${String(stepIndex++).padStart(2, '0')}-${bubble.bubbleName}-query`;
    decomposition[stepKey] = {
      ...bubble,
      parameters: [
        ...bubble.parameters,
        {
          name: 'connectionString',
          value: 'process.env.DATABASE_URL',
          type: BubbleParameterType.STRING,
        },
      ],
      hasAwait: true,
      hasActionCall: true,
    };
  }

  // Step 4: AI/Analysis bubbles
  for (const [, bubble] of categorizedBubbles.ai) {
    const stepKey = `step-${String(stepIndex++).padStart(2, '0')}-${bubble.bubbleName}-analysis`;
    decomposition[stepKey] = {
      ...bubble,
      parameters: generateDataFlowParameters(bubble, decomposition, {
        model: 'gemini-2.0-flash-exp',
      }),
      hasAwait: true,
      hasActionCall: true,
    };
  }

  // Step 5: Output/Responder bubbles
  for (const [, bubble] of categorizedBubbles.output) {
    const stepKey = `step-${String(stepIndex++).padStart(2, '0')}-${bubble.bubbleName}-responder`;
    decomposition[stepKey] = {
      ...bubble,
      parameters: generateDataFlowParameters(bubble, decomposition),
      hasAwait: true,
      hasActionCall: true,
    };
  }

  // Step 6: Any uncategorized bubbles
  for (const [, bubble] of categorizedBubbles.other) {
    const stepKey = `step-${String(stepIndex++).padStart(2, '0')}-${bubble.bubbleName}`;
    decomposition[stepKey] = {
      ...bubble,
      parameters: generateDataFlowParameters(bubble, decomposition),
      hasAwait: bubble.hasAwait ?? true,
      hasActionCall: bubble.hasActionCall ?? true,
    };
  }

  return decomposition;
}

/**
 * Categorizes bubbles by their service type for proper workflow ordering.
 */
function categorizeBubbles(
  bubbleParameters: Record<string, ParsedBubble>
): {
  input: Map<string, ParsedBubble>;
  formatter: Map<string, ParsedBubble>;
  database: Map<string, ParsedBubble>;
  ai: Map<string, ParsedBubble>;
  output: Map<string, ParsedBubble>;
  other: Map<string, ParsedBubble>;
} {
  const categories = {
    input: new Map<string, ParsedBubble>(),
    formatter: new Map<string, ParsedBubble>(),
    database: new Map<string, ParsedBubble>(),
    ai: new Map<string, ParsedBubble>(),
    output: new Map<string, ParsedBubble>(),
    other: new Map<string, ParsedBubble>(),
  };

  for (const [key, bubble] of Object.entries(bubbleParameters)) {
    const bubbleName = bubble.bubbleName.toLowerCase();
    const className = bubble.className.toLowerCase();

    // Input/Listener category
    if (
      bubbleName.includes('slack') && 
      (className.includes('listener') || className.includes('event') || key.includes('listener'))
    ) {
      categories.input.set(key, bubble);
    }
    // Formatter/Parser category
    else if (
      bubbleName.includes('formatter') ||
      bubbleName.includes('parser') ||
      className.includes('parser') ||
      className.includes('formatter')
    ) {
      categories.formatter.set(key, bubble);
    }
    // Database category
    else if (
      bubbleName.includes('database') ||
      bubbleName.includes('db') ||
      bubbleName.includes('storage') ||
      className.includes('database') ||
      className.includes('query')
    ) {
      categories.database.set(key, bubble);
    }
    // AI category
    else if (
      bubbleName.includes('ai') ||
      bubbleName.includes('agent') ||
      bubbleName.includes('llm') ||
      className.includes('ai') ||
      className.includes('agent')
    ) {
      categories.ai.set(key, bubble);
    }
    // Output category
    else if (
      (bubbleName.includes('slack') && (className.includes('sender') || className.includes('responder'))) ||
      bubbleName.includes('sender') ||
      bubbleName.includes('responder') ||
      bubbleName.includes('output') ||
      className.includes('sender')
    ) {
      categories.output.set(key, bubble);
    }
    // Uncategorized
    else {
      categories.other.set(key, bubble);
    }
  }

  return categories;
}

/**
 * Generates parameters with proper data flow references to previous steps.
 */
function generateDataFlowParameters(
  bubble: ParsedBubble,
  previousDecomposition: Record<string, ParsedBubble>,
  defaults: Record<string, unknown> = {}
): BubbleParameter[] {
  const params = [...bubble.parameters];
  const previousSteps = Object.keys(previousDecomposition);
  const lastStepKey = previousSteps[previousSteps.length - 1];

  // Add default parameters if not already present
  for (const [key, value] of Object.entries(defaults)) {
    const exists = params.some(p => p.name === key);
    if (!exists) {
      params.push({
        name: key,
        value: typeof value === 'string' ? value : JSON.stringify(value),
        type: typeof value === 'string' ? BubbleParameterType.STRING : BubbleParameterType.OBJECT,
      });
    }
  }

  // Update parameter values to reference previous steps if appropriate
  return params.map(param => {
    // If parameter value already references a step, keep it
    if (
      typeof param.value === 'string' &&
      (param.value.includes('.') || param.value.startsWith('process.env'))
    ) {
      return param;
    }

    // If there's a previous step, try to reference its output
    if (lastStepKey) {
      const stepName = lastStepKey.replace(/^step-\d+-/, '').replace(/-(listener|parser|query|analysis|responder)$/, '');
      return {
        ...param,
        value: `${stepName}.${param.name}`,
      };
    }

    return param;
  });
}

/**
 * Carry over stored credentials from an existing BubbleFlow's bubbles onto a
 * freshly parsed set of bubble parameters. Matching is done by `bubbleName`
 * (not variable id) so credentials survive when variable ids change but the
 * logical bubbles stay the same.
 *
 * @param params - Newly parsed bubble parameters (variable id -> bubble)
 * @param existingFlowParams - Previously stored bubble parameters to copy credentials from
 * @param _credentials - Reserved for callers that also pass a credential map
 * @returns Bubble parameters with credentials merged in by bubble name
 */
export function mergeCredentialsByBubbleName(
  params: Record<string | number, ParsedBubbleWithInfo>,
  existingFlowParams: Record<string | number, ParsedBubbleWithInfo> | null,
  _credentials?: unknown
): Record<string | number, ParsedBubbleWithInfo> {
  if (!existingFlowParams) {
    return params;
  }

  // Index existing credential values by bubble name for fast lookup.
  const credentialsByBubbleName = new Map<string, Record<string, unknown>>();
  for (const bubble of Object.values(existingFlowParams)) {
    const credParam = (bubble.parameters ?? []).find(
      (p) => p.name === 'credentials'
    );
    if (
      credParam &&
      credParam.value &&
      typeof credParam.value === 'object' &&
      !Array.isArray(credParam.value)
    ) {
      credentialsByBubbleName.set(
        bubble.bubbleName,
        credParam.value as Record<string, unknown>
      );
    }
  }

  if (credentialsByBubbleName.size === 0) {
    return params;
  }

  const merged: Record<string | number, ParsedBubbleWithInfo> = {};
  for (const [varId, bubble] of Object.entries(params)) {
    const carried = credentialsByBubbleName.get(bubble.bubbleName);
    const alreadyHasCredentials = (bubble.parameters ?? []).some(
      (p) => p.name === 'credentials'
    );

    if (carried && !alreadyHasCredentials) {
      merged[varId] = {
        ...bubble,
        parameters: [
          ...(bubble.parameters ?? []),
          {
            name: 'credentials',
            type: BubbleParameterType.OBJECT,
            value: carried,
          },
        ],
      };
    } else {
      merged[varId] = bubble;
    }
  }

  return merged;
}
