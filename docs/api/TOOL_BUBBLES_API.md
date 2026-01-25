# Tool Bubbles API Documentation

Complete API reference for all Tool Bubbles in BubbleLab.

**Table of Contents:**
- [Overview](#overview)
- [Code Edit Tool](#code-edit-tool)
- [Chart.js Tool](#chartjs-tool)
- [Google Maps Tool](#google-maps-tool)
- [Instagram Tool](#instagram-tool)
- [LinkedIn Tool](#linkedin-tool)
- [Research Agent Tool](#research-agent-tool)
- [SQL Query Tool](#sql-query-tool)
- [Twitter Tool](#twitter-tool)
- [YouTube Tool](#youtube-tool)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Overview

Tool Bubbles are specialized components that provide advanced data processing, analysis, and manipulation capabilities. Unlike Service Bubbles that connect to external services, Tool Bubbles implement sophisticated algorithms and transformations directly within BubbleLab.

### Common Features

All Tool Bubbles support:

- **Data Validation**: Input schema validation
- **Error Handling**: Comprehensive error reporting
- **Progress Tracking**: Real-time progress updates
- **Cancellation Support**: Graceful operation cancellation
- **Memory Management**: Efficient resource usage
- **Logging**: Detailed operation logs

### Tool Categories

1. **Code Analysis**: Code editing, linting, transformation
2. **Data Visualization**: Chart generation, graphing
3. **Social Media**: Platform-specific data extraction
4. **Research**: Advanced research and analysis
5. **Database**: Query execution and analysis
6. **Geospatial**: Location-based data processing

---

## Code Edit Tool

**Purpose**: Advanced code editing with linting, formatting, and transformation capabilities.

### API Operations

#### `execute(context: BubbleContext)`

Performs code editing operations.

**Parameters (Lint):**

```typescript
{
  operation: 'lint';
  code: string;            // Source code to lint
  language: 'javascript' | 'typescript' | 'python' | 'java' | 'cpp' | 'csharp' | 'go' | 'rust' | 'php' | 'ruby';
  rules?: string[];        // Custom linting rules
  fix?: boolean;           // Auto-fix issues (default: false)
  timeout?: number;        // Operation timeout in milliseconds (default: 30000)
}
```

**Parameters (Format):**

```typescript
{
  operation: 'format';
  code: string;            // Source code to format
  language: string;        // Programming language
  options?: {              // Formatter options
    indentSize?: number;   // Indent size (default: 2)
    indentStyle?: 'spaces' | 'tabs';
    maxWidth?: number;     // Line width (default: 80)
    semicolons?: boolean;  // Insert semicolons (language-dependent)
  };
  timeout?: number;
}
```

**Parameters (Transform):**

```typescript
{
  operation: 'transform';
  code: string;            // Source code
  from: string;            // Source language/framework
  to: string;              // Target language/framework
  preserveLogic?: boolean; // Preserve application logic (default: true)
  timeout?: number;
}
```

**Parameters (Refactor):**

```typescript
{
  operation: 'refactor';
  code: string;            // Source code
  language: string;
  pattern: string;         // Refactoring pattern
    // 'extract-function' | 'inline-variable' | 'rename' |
    // 'convert-arrow-to-function' | 'convert-to-arrow' |
    // 'simplify-conditional' | 'merge-declarations'
  target?: string;         // Target identifier (for rename, etc.)
  timeout?: number;
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    result: string;        // Processed code
    issues?: Array<{       // Linting issues
      line: number;
      column: number;
      severity: 'error' | 'warning' | 'info';
      message: string;
      ruleId: string;
      fixable: boolean;
    }>;
    stats: {
      lines: number;       // Total lines
      functions: number;   // Number of functions
      classes: number;     // Number of classes
      complexity: number;  // Cyclomatic complexity
    };
    executionTime: number; // Operation time in milliseconds
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const codeEditTool = new CodeEditTool();

// Lint JavaScript code
const lintResult = await codeEditTool.execute({
  operation: 'lint',
  code: 'function foo() { var x = 1; return x; }',
  language: 'javascript',
  fix: true
});

// Format Python code
const formatResult = await codeEditTool.execute({
  operation: 'format',
  code: 'def foo(x,y): return x+y',
  language: 'python',
  options: {
    indentSize: 4,
    maxWidth: 100
  }
});

// Refactor: Extract function
const refactorResult = await codeEditTool.execute({
  operation: 'refactor',
  code: `
    function calculateTotal(price, quantity, tax) {
      return price * quantity * (1 + tax);
    }
  `,
  language: 'javascript',
  pattern: 'extract-function',
  target: 'calculateTotal'
});
```

**Supported Languages:**

- **JavaScript**: ESLint, Prettier
- **TypeScript**: TSLint, Prettier
- **Python**: Pylint, Black, autopep8
- **Java**: Checkstyle, Google Java Format
- **C++**: Clang-Tidy, ClangFormat
- **C#**: Roslyn analyzers
- **Go**: gofmt, golint
- **Rust**: rustfmt, clippy
- **PHP**: PHP CS Fixer
- **Ruby**: RuboCop

**Error Responses:**

- `400`: Invalid operation or parameters
- `404`: Unsupported language
- `422`: Parse error in code
- `500`: Internal tool error

**Performance:**

- Linting: ~100ms per 1000 lines
- Formatting: ~50ms per 1000 lines
- Transformation: ~500ms per 1000 lines
- Refactoring: ~200ms per operation

---

## Chart.js Tool

**Purpose**: Generate charts and visualizations using Chart.js library.

### API Operations

#### `execute(context: BubbleContext)`

Generates charts from data.

**Parameters:**

```typescript
{
  type: 'bar' | 'line' | 'pie' | 'doughnut' | 'radar' | 'polarArea' |
       'scatter' | 'bubble' | 'heatmap' | 'boxplot';
  data: {
    labels: string[];      // X-axis labels or category labels
    datasets: Array<{
      label: string;       // Dataset label
      data: number[];      // Data points
      backgroundColor?: string | string[]; // Color(s)
      borderColor?: string | string[];    // Border color(s)
      borderWidth?: number;               // Border width
      fill?: boolean;        // Fill area under line
      tension?: number;      // Line tension (0-1)
    }>;
  };
  options?: {              // Chart.js options
    responsive?: boolean;
    maintainAspectRatio?: boolean;
    plugins?: {
      title?: {
        display: boolean;
        text: string;
        font?: { size: number; family: string };
      };
      legend?: {
        display: boolean;
        position: 'top' | 'bottom' | 'left' | 'right';
      };
      tooltip?: {
        enabled: boolean;
        mode: 'index' | 'point' | 'nearest' | 'x' | 'y';
        intersect: boolean;
      };
    };
    scales?: {
      x?: {
        display: boolean;
        title?: { display: boolean; text: string };
      };
      y?: {
        display: boolean;
        title?: { display: boolean; text: string };
        beginAtZero?: boolean;
      };
    };
  };
  format: 'png' | 'svg' | 'json'; // Output format
  width?: number;           // Chart width in pixels (default: 800)
  height?: number;          // Chart height in pixels (default: 600)
  timeout?: number;         // Generation timeout in milliseconds (default: 10000)
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    chart: string;         // Base64-encoded image (for png/svg)
      // OR chart config (for json)
    type: string;          // Chart type
    datasets: number;      // Number of datasets
    dataPoints: number;    // Total data points
    dimensions: {
      width: number;
      height: number;
    };
    generationTime: number; // Generation time in milliseconds
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const chartTool = new ChartJsTool();

// Bar chart
const barChart = await chartTool.execute({
  type: 'bar',
  data: {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    datasets: [{
      label: 'Sales',
      data: [120, 190, 300, 250, 280],
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      borderColor: 'rgba(75, 192, 192, 1)',
      borderWidth: 1
    }]
  },
  options: {
    plugins: {
      title: {
        display: true,
        text: 'Monthly Sales'
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Revenue ($)'
        }
      }
    }
  },
  format: 'png',
  width: 800,
  height: 600
});

// Line chart with multiple datasets
const lineChart = await chartTool.execute({
  type: 'line',
  data: {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    datasets: [
      {
        label: 'Product A',
        data: [65, 59, 80, 81, 56],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      },
      {
        label: 'Product B',
        data: [28, 48, 40, 19, 86],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1
      }
    ]
  },
  format: 'svg'
});

// Pie chart
const pieChart = await chartTool.execute({
  type: 'pie',
  data: {
    labels: ['Red', 'Blue', 'Yellow'],
    datasets: [{
      data: [300, 50, 100],
      backgroundColor: [
        'rgb(255, 99, 132)',
        'rgb(54, 162, 235)',
        'rgb(255, 205, 86)'
      ]
    }]
  },
  format: 'png'
});
```

**Chart Types:**

- **Bar**: Vertical/horizontal bars
- **Line**: Connected points with optional filling
- **Pie**: Circular proportional representation
- **Doughnut**: Pie with hole in center
- **Radar**: Spider web chart
- **Polar Area**: Polar coordinate chart
- **Scatter**: X-Y coordinate points
- **Bubble**: Scatter with bubble size
- **Heatmap**: Color-coded grid
- **Box Plot**: Statistical distribution

**Error Responses:**

- `400`: Invalid chart type or data format
- `422`: Data validation failed
- `500`: Chart generation error

**Performance:**

- Generation time: ~100-500ms depending on complexity
- Memory usage: ~10-50MB depending on data size
- Max data points: 10,000 per dataset

**Best Practices:**

1. **Limit data points**: For performance, limit to ~1000 points per dataset
2. **Use appropriate colors**: Ensure color contrast and accessibility
3. **Label clearly**: Always include labels and titles
4. **Choose right chart type**: Match chart type to data characteristics
5. **Optimize for size**: Use appropriate dimensions for intended use

---

## Google Maps Tool

**Purpose**: Interact with Google Maps API for location-based operations.

### API Operations

#### `execute(context: BubbleContext)`

Performs Google Maps operations.

**Parameters (Geocode):**

```typescript
{
  operation: 'geocode';
  address: string;         // Address to geocode
  bounds?: {               // Bias results to area
    northeast: { lat: number; lng: number };
    southwest: { lat: number; lng: number };
  };
  language?: string;       // Language code (default: 'en')
  timeout?: number;
}

// Credentials:
{
  apiKey: string;          // Google Maps API key
}
```

**Parameters (Reverse Geocode):**

```typescript
{
  operation: 'reverse-geocode';
  lat: number;             // Latitude
  lng: number;             // Longitude
  language?: string;
  timeout?: number;
}
```

**Parameters (Distance Matrix):**

```typescript
{
  operation: 'distance-matrix';
  origins: Array<{ lat: number; lng: number } | string>;
  destinations: Array<{ lat: number; lng: number } | string>;
  mode?: 'driving' | 'walking' | 'bicycling' | 'transit';
  units?: 'metric' | 'imperial';
  avoid?: 'tolls' | 'highways' | 'ferries';
  timeout?: number;
}
```

**Parameters (Places):**

```typescript
{
  operation: 'places-search';
  query: string;           // Search query
  location?: { lat: number; lng: number }; // Search near location
  radius?: number;         // Search radius in meters (max: 50000)
  type?: string;           // Place type (e.g., 'restaurant', 'gas_station')
  minPrice?: number;       // Price level (0-4)
  maxPrice?: number;
  openNow?: boolean;       // Only open places
  timeout?: number;
}
```

**Parameters (Directions):**

```typescript
{
  operation: 'directions';
  origin: { lat: number; lng: number } | string;
  destination: { lat: number; lng: number } | string;
  mode?: 'driving' | 'walking' | 'bicycling' | 'transit';
  waypoints?: Array<{ lat: number; lng: number } | string>;
  optimize?: boolean;      // Optimize route order
  avoid?: 'tolls' | 'highways' | 'ferries';
  departureTime?: string;  // ISO-8601 timestamp
  timeout?: number;
}
```

**Response:**

```typescript
// Geocode:
{
  success: boolean;
  data: {
    address: string;       // Formatted address
    location: {
      lat: number;
      lng: number;
    };
    viewport: {            // Recommended viewport
      northeast: { lat: number; lng: number };
      southwest: { lat: number; lng: number };
    };
    components: {          // Address components
      street_number?: string;
      street?: string;
      city?: string;
      county?: string;
      state?: string;
      country?: string;
      postal_code?: string;
    };
    types: string[];       // Address types
  };
  error?: string;
  correlationId: string;
}

// Distance Matrix:
{
  success: boolean;
  data: {
    origins: string[];
    destinations: string[];
    rows: Array<{
      elements: Array<{
        status: string;
        distance?: {
          text: string;    // Human-readable distance
          value: number;   // Distance in meters
        };
        duration?: {
          text: string;    // Human-readable duration
          value: number;   // Duration in seconds
        };
      }>;
    }>;
  };
  error?: string;
  correlationId: string;
}

// Places:
{
  success: boolean;
  data: {
    places: Array<{
      placeId: string;
      name: string;
      address?: string;
      location: { lat: number; lng: number };
      types: string[];
      rating?: number;
      priceLevel?: number;
      openNow?: boolean;
      photos?: Array<{
        photoReference: string;
        width: number;
        height: number;
      }>;
    }>;
    totalCount: number;
  };
  error?: string;
  correlationId: string;
}

// Directions:
{
  success: boolean;
  data: {
    routes: Array<{
      summary: string;
      distance: {
        text: string;
        value: number;
      };
      duration: {
        text: string;
        value: number;
      };
      steps: Array<{
        instruction: string;
        distance: { text: string; value: number };
        duration: { text: string; value: number };
      }>;
      polyline: string;     // Encoded polyline
    }>;
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const mapsTool = new GoogleMapsTool();

// Geocode address
const geocodeResult = await mapsTool.execute({
  operation: 'geocode',
  address: '1600 Amphitheatre Parkway, Mountain View, CA'
});

// Get directions
const directionsResult = await mapsTool.execute({
  operation: 'directions',
  origin: 'San Francisco, CA',
  destination: 'Los Angeles, CA',
  mode: 'driving',
  optimize: true
});

// Search for nearby restaurants
const placesResult = await mapsTool.execute({
  operation: 'places-search',
  query: 'restaurant',
  location: { lat: 37.7749, lng: -122.4194 },
  radius: 1000,
  minPrice: 2,
  maxPrice: 4,
  openNow: true
});

// Calculate distance matrix
const matrixResult = await mapsTool.execute({
  operation: 'distance-matrix',
  origins: ['San Francisco, CA', 'San Jose, CA'],
  destinations: ['Los Angeles, CA', 'San Diego, CA'],
  mode: 'driving'
});
```

**Error Responses:**

- `400`: Invalid request parameters
- `401`: Invalid API key
- `403`: API key quota exceeded
- `404`: Location not found
- `429`: Rate limit exceeded
- `500`: Google Maps API error

**Rate Limits:**

- Geocoding: 50 requests per second
- Directions: 50 requests per second
- Distance Matrix: 1000 elements per request
- Places: 150 requests per day (free tier)

**Best Practices:**

1. **Cache results**: Geocoding results are cacheable
2. **Use place IDs**: Use place IDs instead of addresses when possible
3. **Optimize routes**: Use optimize parameter for multi-stop routes
4. **Handle rate limits**: Implement exponential backoff
5. **Validate addresses**: Validate addresses before geocoding

---

## Instagram Tool

**Purpose**: Extract data from Instagram via Apify actors.

### API Operations

#### `execute(context: BubbleContext)`

Extracts Instagram data.

**Parameters (Profile Posts):**

```typescript
{
  operation: 'profile-posts';
  username: string;        // Instagram username
  resultsLimit?: number;   // Max posts to return (default: 30)
  addParentData?: boolean; // Include parent data (default: false)
  timeout?: number;        // Operation timeout in milliseconds (default: 300000)
}

// Credentials:
{
  apifyApiToken: string;   // Apify API token
}
```

**Parameters (Hashtag Posts):**

```typescript
{
  operation: 'hashtag-posts';
  hashtag: string;         // Hashtag without # (e.g., 'nature')
  resultsLimit?: number;
  addParentData?: boolean;
  timeout?: number;
}
```

**Parameters (Profile Details):**

```typescript
{
  operation: 'profile-details';
  username: string;
  timeout?: number;
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    posts?: Array<{
      id: string;
      text?: string;
      type: 'photo' | 'video' | 'carousel' | 'story';
      url: string;
      likes: number;
      comments: number;
      timestamp: string;    // ISO-8601
      media: Array<{
        type: string;
        url: string;
        thumbnailUrl?: string;
      }>;
      mentions?: string[];
      hashtags?: string[];
    }>;
    profile?: {
      username: string;
      fullName: string;
      bio: string;
      followers: number;
      following: number;
      posts: number;
      verified: boolean;
      profilePicUrl: string;
      url: string;
    };
    extractedAt: string;   // Extraction timestamp
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const instagramTool = new InstagramTool();

// Get profile posts
const postsResult = await instagramTool.execute({
  operation: 'profile-posts',
  username: 'instagram',
  resultsLimit: 20
});

// Get hashtag posts
const hashtagResult = await instagramTool.execute({
  operation: 'hashtag-posts',
  hashtag: 'nature',
  resultsLimit: 50
});

// Get profile details
const profileResult = await instagramTool.execute({
  operation: 'profile-details',
  username: 'instagram'
});
```

**Data Fields:**

- **Post**: Text, type, URL, likes, comments, timestamp
- **Media**: URLs, thumbnails
- **Mentions**: Tagged users
- **Hashtags**: Used hashtags
- **Profile**: Bio, stats, verification

**Error Responses:**

- `400`: Invalid username or hashtag
- `404`: Profile not found
- `429`: Rate limit exceeded
- `500`: Instagram API error

**Rate Limits:**

- Limited by Apify quotas
- Recommend 1 request per second
- Cache results when possible

**Best Practices:**

1. **Respect rate limits**: Don't scrape too aggressively
2. **Cache results**: Instagram data doesn't change frequently
3. **Handle errors gracefully**: Profiles may be private or deleted
4. **Use appropriate limits**: Only extract needed data
5. **Validate input**: Ensure usernames are valid

---

## LinkedIn Tool

**Purpose**: Extract LinkedIn data via Apify actors.

### API Operations

#### `execute(context: BubbleContext)`

Extracts LinkedIn data.

**Parameters (Profile Posts):**

```typescript
{
  operation: 'profile-posts';
  profileUrl: string;     // LinkedIn profile URL
  resultsLimit?: number;   // Max posts (default: 30)
  timeout?: number;        // Operation timeout in milliseconds (default: 300000)
}

// Credentials:
{
  apifyApiToken: string;   // Apify API token
}
```

**Parameters (Jobs Search):**

```typescript
{
  operation: 'jobs-search';
  query: string;           // Job search keywords
  location?: string;       // Job location
  experienceLevel?: string[]; // ['entry', 'associate', 'mid', 'director', 'executive']
  jobType?: string[];      // ['full-time', 'part-time', 'contract', 'internship']
  remoteFilter?: string;   // ['on-site', 'remote', 'hybrid']
  resultsLimit?: number;
  timeout?: number;
}
```

**Parameters (Posts Search):**

```typescript
{
  operation: 'posts-search';
  query: string;           // Search keywords
  resultsLimit?: number;
  timeout?: number;
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    posts?: Array<{
      id: string;
      text: string;
      url: string;
      author: {
        name: string;
        profileUrl: string;
        headline?: string;
      };
      likes: number;
      comments: number;
      shares: number;
      publishedAt: string;  // ISO-8601
      media?: Array<{
        type: string;
        url: string;
      }>;
    }>;
    jobs?: Array<{
      title: string;
      company: string;
      location: string;
      url: string;
      description: string;
      postedAt: string;
      applicants?: number;
      experienceLevel?: string;
      jobType?: string;
      remote?: string;
    }>;
    extractedAt: string;
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const linkedinTool = new LinkedInTool();

// Get profile posts
const postsResult = await linkedinTool.execute({
  operation: 'profile-posts',
  profileUrl: 'https://www.linkedin.com/in/some-profile/',
  resultsLimit: 20
});

// Search for jobs
const jobsResult = await linkedinTool.execute({
  operation: 'jobs-search',
  query: 'software engineer',
  location: 'San Francisco, CA',
  experienceLevel: ['mid', 'senior'],
  jobType: ['full-time'],
  remoteFilter: ['remote', 'hybrid'],
  resultsLimit: 50
});

// Search posts
const searchResult = await linkedinTool.execute({
  operation: 'posts-search',
  query: 'artificial intelligence',
  resultsLimit: 30
});
```

**Data Fields:**

- **Posts**: Text, author, engagement, media
- **Jobs**: Title, company, location, description, requirements
- **Profiles**: Name, headline, company, education

**Error Responses:**

- `400`: Invalid URL or parameters
- `404`: Profile or job not found
- `429`: Rate limit exceeded
- `500`: LinkedIn API error

**Rate Limits:**

- Limited by Apify quotas
- LinkedIn has strict rate limits
- Use conservative limits

**Best Practices:**

1. **Be conservative**: LinkedIn has aggressive rate limiting
2. **Cache aggressively**: Job postings don't change frequently
3. **Respect privacy**: Don't scrape private profiles
4. **Use specific queries**: Narrow searches to relevant results
5. **Handle errors**: LinkedIn often returns errors

---

## Research Agent Tool

**Purpose**: Advanced research and information gathering using AI-powered agents.

### API Operations

#### `execute(context: BubbleContext)`

Performs research tasks.

**Parameters (General Research):**

```typescript
{
  operation: 'research';
  query: string;           // Research query
  depth: 'shallow' | 'medium' | 'deep'; // Research depth
  sources?: Array<'web' | 'academic' | 'news' | 'books'>;
  maxResults?: number;     // Max sources to analyze (default: 10)
  timeout?: number;        // Research timeout in milliseconds (default: 300000)
}

// Credentials (optional, depending on sources):
{
  openaiApiKey?: string;   // For AI analysis
  serpApiKey?: string;     // For web search
  scholarApiKey?: string;  // For academic search
}
```

**Parameters (Fact Check):**

```typescript
{
  operation: 'fact-check';
  claim: string;           // Claim to verify
  sources?: string[];      // Specific sources to check
  timeout?: number;
}
```

**Parameters (Summary):**

```typescript
{
  operation: 'summarize';
  topic: string;           // Topic to summarize
  length: 'brief' | 'standard' | 'detailed';
  format: 'bullet' | 'paragraph' | 'structured';
  timeout?: number;
}
```

**Parameters (Comparison):**

```typescript
{
  operation: 'compare';
  subjects: string[];      // Subjects to compare (2-5 items)
  criteria?: string[];     // Comparison criteria
  timeout?: number;
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    findings: Array<{
      source: string;
      url: string;
      title: string;
      snippet: string;
      relevance: number;   // 0-1 score
      publishedAt?: string;
      author?: string;
    }>;
    summary: string;        // Research summary
    keyPoints: string[];    // Key findings
    confidence: number;     // 0-1 confidence score
    sources: {
      total: number;
      web: number;
      academic: number;
      news: number;
      books: number;
    };
    researchTime: number;  // Research time in milliseconds
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const researchAgent = new ResearchAgentTool();

// General research
const researchResult = await researchAgent.execute({
  operation: 'research',
  query: 'Impact of artificial intelligence on healthcare',
  depth: 'deep',
  sources: ['academic', 'news', 'web'],
  maxResults: 20
});

// Fact check
const factCheckResult = await researchAgent.execute({
  operation: 'fact-check',
  claim: 'Climate change is caused by human activity'
});

// Summarize topic
const summaryResult = await researchAgent.execute({
  operation: 'summarize',
  topic: 'Quantum computing applications',
  length: 'detailed',
  format: 'structured'
});

// Compare options
const compareResult = await researchAgent.execute({
  operation: 'compare',
  subjects: ['Python', 'JavaScript', 'Rust'],
  criteria: ['performance', 'ease of use', 'ecosystem', 'adoption']
});
```

**Research Capabilities:**

- **Web Search**: General web information
- **Academic**: Scholarly articles and papers
- **News**: Recent news articles
- **Books**: Published books and references
- **Fact Checking**: Verify claims against sources
- **Summarization**: Condense information
- **Comparison**: Compare multiple subjects

**Error Responses:**

- `400`: Invalid query or parameters
- `404`: No results found
- `429**: Rate limit exceeded
- `500`: Research tool error

**Rate Limits:**

- Web search: 100 requests per day
- Academic search: 50 requests per day
- AI analysis: 1000 requests per day

**Best Practices:**

1. **Use specific queries**: More specific = better results
2. **Choose appropriate depth**: Deeper = more time
3. **Diversify sources**: Use multiple source types
4. **Verify findings**: Cross-reference important information
5. **Cache results**: Research results are cacheable

---

## SQL Query Tool

**Purpose**: Execute and analyze SQL queries across multiple database types.

### API Operations

#### `execute(context: BubbleContext)`

Executes SQL queries.

**Parameters (Execute Query):**

```typescript
{
  operation: 'execute';
  query: string;           // SQL query
  database: 'postgresql' | 'mysql' | 'sqlite' | 'mssql' | 'oracle';
  params?: any[];          // Query parameters
  timeout?: number;        // Query timeout in milliseconds (default: 30000)
  maxRows?: number;        // Max rows to return (default: 1000)
}

// Credentials (stored securely):
{
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
  ssl?: boolean;
}
```

**Parameters (Analyze Query):**

```typescript
{
  operation: 'analyze';
  query: string;
  database: string;
  timeout?: number;
}
```

**Parameters (Optimize Query):**

```typescript
{
  operation: 'optimize';
  query: string;
  database: string;
  schema?: any;           // Database schema information
  timeout?: number;
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    rows?: any[];          // Query results
    rowCount: number;      // Rows affected/returned
    columns?: Array<{
      name: string;
      type: string;
    }>;
    executionTime: number; // Execution time in milliseconds
    analysis?: {
      complexity: string;  // Query complexity
      tables: string[];    // Tables accessed
      indexes: string[];   // Indexes used
      recommendations: string[];
    };
    optimization?: {
      originalQuery: string;
      optimizedQuery: string;
      improvements: string[];
      estimatedSpeedup: number; // Percentage improvement
    };
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const sqlTool = new SqlQueryTool();

// Execute query
const result = await sqlTool.execute({
  operation: 'execute',
  query: 'SELECT * FROM users WHERE status = $1',
  database: 'postgresql',
  params: ['active'],
  maxRows: 100
});

// Analyze query
const analysis = await sqlTool.execute({
  operation: 'analyze',
  query: `
    SELECT u.name, o.order_date
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE u.status = 'active'
  `,
  database: 'postgresql'
});

// Optimize query
const optimization = await sqlTool.execute({
  operation: 'optimize',
  query: 'SELECT * FROM users WHERE LOWER(name) LIKE LOWER($1)',
  database: 'postgresql',
  schema: {
    tables: {
      users: {
        columns: [
          { name: 'id', type: 'integer', indexed: true },
          { name: 'name', type: 'text', indexed: false },
          { name: 'status', type: 'text', indexed: true }
        ]
      }
    }
  }
});
```

**Supported Databases:**

- **PostgreSQL**: Full support
- **MySQL**: Full support
- **SQLite**: Full support
- **MSSQL**: Full support
- **Oracle**: Basic support

**Query Analysis Features:**

- **Complexity Assessment**: Simple, medium, complex
- **Table Access**: Tables and indexes used
- **Performance Issues**: Potential bottlenecks
- **Recommendations**: Optimization suggestions

**Optimization Features:**

- **Index suggestions**: Missing indexes
- **Query rewriting**: More efficient SQL
- **Join optimization**: Better join strategies
- **Subquery optimization**: Flatten subqueries

**Error Responses:**

- `400`: Invalid SQL syntax
- `401`: Authentication failed
- `403`: Permission denied
- `500`: Database error

**Rate Limits:**

- 50 queries per minute per flow
- Max 1000 rows per query
- Query timeout: 30 seconds

**Best Practices:**

1. **Use parameters**: Prevent SQL injection
2. **Limit results**: Use maxRows and LIMIT
3. **Analyze before optimizing**: Understand bottlenecks
4. **Test optimizations**: Verify optimizations help
5. **Monitor performance**: Track slow queries

---

## Twitter Tool

**Purpose**: Extract Twitter/X data via Apify actors.

### API Operations

#### `execute(context: BubbleContext)`

Extracts Twitter data.

**Parameters (User Tweets):**

```typescript
{
  operation: 'user-tweets';
  username: string;        // Twitter username (without @)
  resultsLimit?: number;   // Max tweets (default: 30)
  includeReplies?: boolean; // Include replies (default: false)
  includeRetweets?: boolean; // Include retweets (default: true)
  timeout?: number;        // Operation timeout in milliseconds (default: 300000)
}

// Credentials:
{
  apifyApiToken: string;   // Apify API token
}
```

**Parameters (Search Tweets):**

```typescript
{
  operation: 'search';
  query: string;           // Search query
  resultsLimit?: number;
  language?: string;       // Filter by language (e.g., 'en', 'es')
  timeout?: number;
}
```

**Parameters (Trending):**

```typescript
{
  operation: 'trending';
  location?: string;       // Location for trends (e.g., 'US', 'UK')
  timeout?: number;
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    tweets?: Array<{
      id: string;
      text: string;
      url: string;
      author: {
        username: string;
        name: string;
        followers: number;
        verified: boolean;
      };
      createdAt: string;   // ISO-8601
      likes: number;
      retweets: number;
      replies: number;
      quotes: number;
      hashtags: string[];
      mentions: string[];
      media?: Array<{
        type: string;
        url: string;
      }>;
      isReply: boolean;
      isRetweet: boolean;
      isQuote: boolean;
    }>;
    trending?: Array<{
      rank: number;
      topic: string;
      tweetCount: number;
    }>;
    extractedAt: string;
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const twitterTool = new TwitterTool();

// Get user tweets
const tweetsResult = await twitterTool.execute({
  operation: 'user-tweets',
  username: 'elonmusk',
  resultsLimit: 50,
  includeReplies: false,
  includeRetweets: false
});

// Search tweets
const searchResult = await twitterTool.execute({
  operation: 'search',
  query: 'artificial intelligence',
  resultsLimit: 100,
  language: 'en'
});

// Get trending topics
const trendingResult = await twitterTool.execute({
  operation: 'trending',
  location: 'US'
});
```

**Data Fields:**

- **Tweet**: Text, author, engagement, media
- **Author**: Username, name, followers, verification
- **Engagement**: Likes, retweets, replies, quotes
- **Content**: Hashtags, mentions, media
- **Metadata**: Creation time, URL, type

**Error Responses:**

- `400`: Invalid username or query
- `404**: User not found
- `429`: Rate limit exceeded
- `500`: Twitter API error

**Rate Limits:**

- Limited by Apify quotas
- Twitter has strict rate limits
- Use conservative limits

**Best Practices:**

1. **Respect rate limits**: Twitter has aggressive rate limiting
2. **Filter carefully**: Only extract needed data
3. **Cache results**: Tweets don't change after posting
4. **Handle private accounts**: Some data may be inaccessible
5. **Use specific queries**: Narrow search terms

---

## YouTube Tool

**Purpose**: Extract YouTube data via Apify actors.

### API Operations

#### `execute(context: BubbleContext)`

Extracts YouTube data.

**Parameters (Channel Videos):**

```typescript
{
  operation: 'channel-videos';
  channelId: string;      // Channel ID or URL
  resultsLimit?: number;   // Max videos (default: 50)
  timeout?: number;        // Operation timeout in milliseconds (default: 300000)
}

// Credentials:
{
  apifyApiToken: string;   // Apify API token
}
```

**Parameters (Video Details):**

```typescript
{
  operation: 'video-details';
  videoId: string;         // Video ID or URL
  includeComments?: boolean; // Include comments (default: false)
  commentsLimit?: number;  // Max comments (default: 20)
  timeout?: number;
}
```

**Parameters (Search):**

```typescript
{
  operation: 'search';
  query: string;           // Search query
  resultsLimit?: number;
  uploadDate?: 'hour' | 'today' | 'week' | 'month' | 'year';
  duration?: 'short' | 'medium' | 'long';
  timeout?: number;
}
```

**Response:**

```typescript
{
  success: boolean;
  data: {
    videos?: Array<{
      id: string;
      title: string;
      description: string;
      url: string;
      thumbnail: string;
      channel: {
        id: string;
        name: string;
        url: string;
      };
      publishedAt: string;  // ISO-8601
      duration: string;     // ISO-8601 duration
      views: number;
      likes: number;
      comments: number;
      tags: string[];
      categoryId: string;
      liveBroadcast: 'live' | 'none' | 'upcoming';
    }>;
    comments?: Array<{
      id: string;
      text: string;
      author: string;
      authorUrl: string;
      likes: number;
      publishedAt: string;
      replies: number;
    }>;
    extractedAt: string;
  };
  error?: string;
  correlationId: string;
}
```

**Example:**

```typescript
const youtubeTool = new YouTubeTool();

// Get channel videos
const videosResult = await youtubeTool.execute({
  operation: 'channel-videos',
  channelId: 'UC_x5XG1OV2P6uZZ5FSM9Ttw', // Google Developers channel
  resultsLimit: 100
});

// Get video details with comments
const detailsResult = await youtubeTool.execute({
  operation: 'video-details',
  videoId: 'dQw4w9WgXcQ',
  includeComments: true,
  commentsLimit: 50
});

// Search videos
const searchResult = await youtubeTool.execute({
  operation: 'search',
  query: 'machine learning tutorial',
  resultsLimit: 50,
  uploadDate: 'year',
  duration: 'medium'
});
```

**Data Fields:**

- **Video**: Title, description, duration, views
- **Channel**: Name, ID, URL
- **Engagement**: Likes, comments
- **Metadata**: Tags, category, publish date
- **Comments**: Text, author, likes

**Error Responses:**

- `400`: Invalid video/channel ID
- `404**: Video or channel not found
- `429`: Rate limit exceeded
- `500`: YouTube API error

**Rate Limits:**

- Limited by Apify quotas
- YouTube has generous rate limits
- Still recommend 1 request per second

**Best Practices:**

1. **Use video IDs**: More reliable than URLs
2. **Filter by duration**: Only extract needed duration
3. **Limit comments**: Comments can be numerous
4. **Cache aggressively**: Video data doesn't change frequently
5. **Handle errors**: Videos may be private or deleted

---

## Error Handling

All Tool Bubbles use standardized error handling.

### Error Categories

**Validation Errors:**
- `INVALID_PARAMETERS`: Invalid input parameters
- `MISSING_REQUIRED_FIELD`: Required field missing
- `TYPE_MISMATCH`: Field has wrong type
- `VALUE_OUT_OF_RANGE`: Value outside acceptable range

**Execution Errors:**
- `OPERATION_FAILED`: Operation execution failed
- `TIMEOUT`: Operation timeout
- `RESOURCE_EXHAUSTED`: Resource limits exceeded
- `CANNOT_CANCEL`: Operation cannot be cancelled

**Data Errors:**
- `DATA_NOT_FOUND`: Requested data not found
- `DATA_CORRUPTED`: Data is corrupted
- `PARSE_ERROR`: Failed to parse data
- `VALIDATION_ERROR`: Data validation failed

**External Errors:**
- `API_ERROR`: External API error
- `NETWORK_ERROR`: Network connectivity issue
- `AUTHENTICATION_FAILED`: Authentication failed
- `RATE_LIMITED`: Rate limit exceeded

### Error Handling Best Practices

1. **Always check success field**: Verify operation succeeded
2. **Log correlation IDs**: Include in logs for debugging
3. **Handle specific errors**: Different handling for different error types
4. **Implement retries**: For transient errors
5. **Fail fast**: For permanent errors

---

## Best Practices

### General

1. **Set appropriate timeouts**: Balance responsiveness and completion
2. **Validate input**: Validate before calling tool
3. **Handle errors gracefully**: Always handle error cases
4. **Log operations**: Use correlation IDs for tracking
5. **Test thoroughly**: Test with various inputs

### Performance

1. **Batch operations**: When possible, batch multiple operations
2. **Use caching**: Cache results when appropriate
3. **Limit data size**: Only request needed data
4. **Monitor performance**: Track execution times
5. **Optimize queries**: Use indexes and efficient queries

### Security

1. **Validate input**: Sanitize all input data
2. **Use parameters**: Prevent injection attacks
3. **Secure credentials**: Never log credentials
4. **Limit access**: Apply principle of least privilege
5. **Audit usage**: Track tool usage

### Data Quality

1. **Verify results**: Validate output data
2. **Handle edge cases**: Test boundary conditions
3. **Document assumptions**: Document data format assumptions
4. **Version schemas**: Use schema versioning
5. **Test with real data**: Test with production-like data

---

**Last Updated:** 2026-01-18
**Version:** 1.0.0
**Maintained By:** BubbleLab Core Team
