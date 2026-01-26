/**
 * Language Detector - Identifies project programming language based on file structure
 *
 * Detects whether a project is primarily Python, TypeScript, or JavaScript
 * by analyzing file extensions and configuration files.
 */

export interface LanguageDetectionResult {
  language: 'TYPESCRIPT' | 'JAVASCRIPT' | 'PYTHON';
  confidence: number; // 0-1
  reason: string;
}

/**
 * Detect the primary language of a project based on its files
 * @param files - Array of file paths in the project
 * @returns LanguageDetectionResult with language, confidence, and reasoning
 */
export function detectProjectLanguage(files: string[]): LanguageDetectionResult {
  const pyFiles = files.filter(f => f.endsWith('.py')).length;
  const tsFiles = files.filter(f => f.endsWith('.ts')).length;
  const jsFiles = files.filter(f => f.endsWith('.js') || f.endsWith('.jsx')).length;
  const totalFiles = files.length;

  // Check for Python-specific configuration files
  const hasRequirementsTxt = files.some(f =>
    f.endsWith('requirements.txt') || f.includes('requirements.txt')
  );
  const hasPyprojectToml = files.some(f =>
    f.endsWith('pyproject.toml') || f.includes('pyproject.toml')
  );
  const hasPipfile = files.some(f =>
    f.endsWith('Pipfile') || f.includes('Pipfile')
  );
  const hasSetupPy = files.some(f =>
    f.endsWith('setup.py') || f.includes('setup.py')
  );

  // Check for TypeScript/JavaScript configuration files
  const hasPackageJson = files.some(f =>
    f.endsWith('package.json') || f.includes('package.json')
  );
  const hasTsConfig = files.some(f =>
    f.endsWith('tsconfig.json') || f.includes('tsconfig.json')
  );

  // Calculate scores
  const pyConfigScore = (hasRequirementsTxt ? 1 : 0) +
                        (hasPyprojectToml ? 1 : 0) +
                        (hasPipfile ? 1 : 0) +
                        (hasSetupPy ? 1 : 0);

  const tsConfigScore = (hasPackageJson ? 1 : 0) +
                        (hasTsConfig ? 2 : 0); // tsconfig.json is a strong TypeScript indicator

  // Determine language with confidence
  const totalSourceFiles = pyFiles + tsFiles + jsFiles;

  if (totalSourceFiles === 0) {
    // Only config files to go on
    if (pyConfigScore > tsConfigScore) {
      return {
        language: 'PYTHON',
        confidence: 0.6,
        reason: `Detected Python project via config files (${pyConfigScore} Python indicators vs ${tsConfigScore} TS/JS indicators)`,
      };
    } else if (tsConfigScore > 0) {
      return {
        language: hasTsConfig ? 'TYPESCRIPT' : 'JAVASCRIPT',
        confidence: 0.6,
        reason: `Detected ${hasTsConfig ? 'TypeScript' : 'JavaScript'} project via config files (${tsConfigScore} indicators vs ${pyConfigScore} Python indicators)`,
      };
    }

    return {
      language: 'TYPESCRIPT', // Default assumption for DevilDev
      confidence: 0.1,
      reason: 'Unable to determine language - no source files or strong config indicators found',
    };
  }

  const pyRatio = pyFiles / totalSourceFiles;
  const tsRatio = tsFiles / totalSourceFiles;
  const jsRatio = jsFiles / totalSourceFiles;
  const tsJsRatio = tsRatio + jsRatio;

  // Strong signals from config files
  if (pyConfigScore >= 2 && pyRatio > 0.1) {
    return {
      language: 'PYTHON',
      confidence: 0.9,
      reason: `Strong Python indicators: ${pyFiles} .py files (${(pyRatio * 100).toFixed(1)}%) and ${pyConfigScore} Python config files`,
    };
  }

  if (hasTsConfig && tsRatio > 0.1) {
    return {
      language: 'TYPESCRIPT',
      confidence: 0.9,
      reason: `Strong TypeScript indicators: ${tsFiles} .ts files (${(tsRatio * 100).toFixed(1)}%) and tsconfig.json present`,
    };
  }

  // Ratio-based detection with 60% threshold
  if (pyRatio >= 0.6) {
    return {
      language: 'PYTHON',
      confidence: 0.8 + (pyConfigScore * 0.05),
      reason: `${pyFiles} .py files (${(pyRatio * 100).toFixed(1)}%) out of ${totalSourceFiles} total source files`,
    };
  }

  if (tsRatio >= 0.6) {
    return {
      language: 'TYPESCRIPT',
      confidence: 0.8 + (tsConfigScore * 0.05),
      reason: `${tsFiles} .ts files (${(tsRatio * 100).toFixed(1)}%) out of ${totalSourceFiles} total source files`,
    };
  }

  if (jsRatio >= 0.6) {
    return {
      language: 'JAVASCRIPT',
      confidence: 0.8 + (tsConfigScore * 0.05),
      reason: `${jsFiles} .js files (${(jsRatio * 100).toFixed(1)}%) out of ${totalSourceFiles} total source files`,
    };
  }

  // Combined TS/JS detection
  if (tsJsRatio >= 0.6) {
    const isTypeScript = tsRatio > jsRatio || hasTsConfig;
    return {
      language: isTypeScript ? 'TYPESCRIPT' : 'JAVASCRIPT',
      confidence: 0.7,
      reason: `Combined ${tsFiles + jsFiles} TS/JS files (${(tsJsRatio * 100).toFixed(1)}%) out of ${totalSourceFiles} total source files`,
    };
  }

  // Low confidence - mixed project
  if (pyRatio > tsJsRatio) {
    return {
      language: 'PYTHON',
      confidence: 0.5,
      reason: `Slightly more Python files (${pyFiles} vs ${tsFiles + jsFiles} TS/JS files)`,
    };
  } else {
    const isTypeScript = tsRatio >= jsRatio || hasTsConfig;
    return {
      language: isTypeScript ? 'TYPESCRIPT' : 'JAVASCRIPT',
      confidence: 0.5,
      reason: `Slightly more ${isTypeScript ? 'TypeScript' : 'JavaScript'} files (${tsFiles + jsFiles} vs ${pyFiles} Python files)`,
    };
  }
}

/**
 * Detect the language of a single file based on its extension
 * @param filename - The filename to analyze
 * @returns Language identifier or null if unknown
 */
export function detectFileLanguage(filename: string): string | null {
  const ext = filename.split('.').pop()?.toLowerCase();

  switch (ext) {
    case 'py':
      return 'PYTHON';
    case 'ts':
      return 'TYPESCRIPT';
    case 'js':
    case 'jsx':
    case 'mjs':
    case 'cjs':
      return 'JAVASCRIPT';
    default:
      return null;
  }
}
