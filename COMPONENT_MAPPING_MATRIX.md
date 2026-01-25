# COMPONENT MAPPING MATRIX
## Streamlit to React/TypeScript Component Mapping

**Generated:** 2025-01-05
**Agent:** Discovery & Audit Agent
**Purpose:** Complete mapping of all Streamlit components to React/TypeScript equivalents for BubbleLab migration

---

## EXECUTIVE SUMMARY

This document provides a comprehensive mapping of all Streamlit components used in the OpenEvolve codebase to their React/TypeScript equivalents suitable for implementation in BubbleLab.

**Total Unique Streamlit Components:** 47 components
**Components with 1:1 Mapping:** 32 (68%)
**Components Requiring Custom Implementation:** 15 (32%)

---

## SECTION 1: INPUT COMPONENTS

### 1.1 Text Input

**Streamlit Component:**
```python
st.text_input(label, value="", key=None, type="default", help=None, max_chars=None)
```

**React Equivalent:**
```typescript
import { TextField } from '@mui/material';
// or
import { Input } from 'antd';
```

**Mapping:**
| Streamlit Prop | React Prop (MUI) | React Prop (AntD) | Notes |
|----------------|------------------|-------------------|-------|
| label | label | label | Direct mapping |
| value | value | value | Direct mapping |
| key | id | id | Unique identifier |
| type | type | type | "default" → "text", "password" → "password" |
| help | tooltip | tooltip | Helper text |
| max_chars | inputProps.maxLength | maxLength | Character limit |
| onChange (implicit) | onChange | onChange | Event handler |

**BubbleLab Implementation:**
```typescript
// Use existing Input component from BubbleLab
import { Input } from '@/components/ui/input';

interface TextInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: 'text' | 'password';
  helperText?: string;
  maxLength?: number;
  id?: string;
}

export const TextInput: React.FC<TextInputProps> = ({
  label,
  value,
  onChange,
  type = 'text',
  helperText,
  maxLength,
  id
}) => {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      <Input
        id={id}
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        maxLength={maxLength}
        className="w-full"
      />
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

### 1.2 Text Area

**Streamlit Component:**
```python
st.text_area(label, value="", height=None, key=None, help=None, max_chars=None)
```

**React Equivalent:**
```typescript
import { TextField } from '@mui/material';
// or
import { Input } from 'antd';
// multiline
```

**Mapping:**
| Streamlit Prop | React Prop (MUI) | React Prop (AntD) | Notes |
|----------------|------------------|-------------------|-------|
| label | label | label | Direct mapping |
| value | value | value | Direct mapping |
| height | rows | rows | Use rows instead |
| key | id | id | Unique identifier |
| help | helperText | tooltip | Helper text |
| max_chars | inputProps.maxLength | maxLength | Character limit |

**BubbleLab Implementation:**
```typescript
import { Textarea } from '@/components/ui/textarea';

interface TextAreaProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  rows?: number;
  helperText?: string;
  maxLength?: number;
  id?: string;
}

export const TextArea: React.FC<TextAreaProps> = ({
  label,
  value,
  onChange,
  rows = 5,
  helperText,
  maxLength,
  id
}) => {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      <Textarea
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={rows}
        maxLength={maxLength}
        className="w-full"
      />
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

### 1.3 Number Input

**Streamlit Component:**
```python
st.number_input(label, min_value=None, max_value=None, value=0, step=1, key=None, help=None)
```

**React Equivalent:**
```typescript
import { TextField } from '@mui/material';
// type="number"
// or
import { InputNumber } from 'antd';
```

**Mapping:**
| Streamlit Prop | React Prop (MUI) | React Prop (AntD) | Notes |
|----------------|------------------|-------------------|-------|
| label | label | label | Direct mapping |
| value | value | value | Direct mapping |
| min_value | inputProps.min | min | Minimum value |
| max_value | inputProps.max | max | Maximum value |
| step | step | step | Increment/decrement |
| key | id | id | Unique identifier |
| help | helperText | tooltip | Helper text |

**BubbleLab Implementation:**
```typescript
import { Input } from '@/components/ui/input';

interface NumberInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  helperText?: string;
  id?: string;
}

export const NumberInput: React.FC<NumberInputProps> = ({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  helperText,
  id
}) => {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      <Input
        id={id}
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

### 1.4 Slider

**Streamlit Component:**
```python
st.slider(label, min_value=0.0, max_value=100.0, value=50.0, step=None, key=None, help=None)
```

**React Equivalent:**
```typescript
import { Slider } from '@mui/material';
// or
import { Slider } from 'antd';
```

**Mapping:**
| Streamlit Prop | React Prop (MUI) | React Prop (AntD) | Notes |
|----------------|------------------|-------------------|-------|
| label | label | label | Direct mapping |
| value | value | value | Direct mapping (array for range) |
| min_value | min | min | Minimum value |
| max_value | max | max | Maximum value |
| step | step | step | Increment |
| key | id | id | Unique identifier |
| help | tooltip | tooltip | Helper text |

**BubbleLab Implementation:**
```typescript
import { Slider } from '@/components/ui/slider';

interface SliderInputProps {
  label: string;
  value: number | [number, number];
  onChange: (value: number | [number, number]) => void;
  min?: number;
  max?: number;
  step?: number;
  helperText?: string;
  id?: string;
}

export const SliderInput: React.FC<SliderInputProps> = ({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  helperText,
  id
}) => {
  return (
    <div className="space-y-2">
      <div className="flex justify-between">
        <label htmlFor={id} className="text-sm font-medium">{label}</label>
        <span className="text-sm text-muted-foreground">
          {Array.isArray(value) ? `${value[0]} - ${value[1]}` : value}
        </span>
      </div>
      <Slider
        id={id}
        value={value}
        onValueChange={(vals) => onChange(Array.isArray(value) ? vals : vals[0])}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 1.5 Select Box

**Streamlit Component:**
```python
st.selectbox(label, options=[], index=0, key=None, help=None)
```

**React Equivalent:**
```typescript
import { Select } from '@mui/material';
// or
import { Select } from 'antd';
```

**Mapping:**
| Streamlit Prop | React Prop (MUI) | React Prop (AntD) | Notes |
|----------------|------------------|-------------------|-------|
| label | label | label | Direct mapping |
| options | options | options | Options array |
| index (for default) | defaultValue | defaultValue | Default selection |
| value | value | value | Current value |
| key | id | id | Unique identifier |
| help | tooltip | tooltip | Helper text |

**BubbleLab Implementation:**
```typescript
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface SelectBoxProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
  helperText?: string;
  id?: string;
}

export const SelectBox: React.FC<SelectBoxProps> = ({
  label,
  value,
  onChange,
  options,
  helperText,
  id
}) => {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id={id}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 1.6 Multi Select

**Streamlit Component:**
```python
st.multiselect(label, options=[], default=[], key=None, help=None)
```

**React Equivalent:**
```typescript
import { Select } from '@mui/material';
// multiple
// or
import { Select } from 'antd';
// mode="multiple"
```

**BubbleLab Implementation:**
```typescript
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';

interface MultiSelectProps {
  label: string;
  value: string[];
  onChange: (value: string[]) => void;
  options: string[];
  helperText?: string;
  id?: string;
}

export const MultiSelect: React.FC<MultiSelectProps> = ({
  label,
  value,
  onChange,
  options,
  helperText,
  id
}) => {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      <div className="flex flex-wrap gap-2">
        {value.map((item) => (
          <Badge key={item} variant="secondary">
            {item}
            <button
              onClick={() => onChange(value.filter(v => v !== item))}
              className="ml-2 hover:text-destructive"
            >
              ×
            </button>
          </Badge>
        ))}
      </div>
      <Select
        value=""
        onValueChange={(val) => !value.includes(val) && onChange([...value, val])}
      >
        <SelectTrigger>
          <SelectValue placeholder="Add option..." />
        </SelectTrigger>
        <SelectContent>
          {options.filter(opt => !value.includes(opt)).map((option) => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM (requires custom implementation)

---

### 1.7 Checkbox

**Streamlit Component:**
```python
st.checkbox(label, value=False, key=None, help=None)
```

**React Equivalent:**
```typescript
import { Checkbox } from '@mui/material';
// or
import { Checkbox } from 'antd';
```

**BubbleLab Implementation:**
```typescript
import { Checkbox } from '@/components/ui/checkbox';

interface CheckboxProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  helperText?: string;
  id?: string;
}

export const CheckboxInput: React.FC<CheckboxProps> = ({
  label,
  checked,
  onChange,
  helperText,
  id
}) => {
  return (
    <div className="space-y-2">
      <div className="flex items-center space-x-2">
        <Checkbox
          id={id}
          checked={checked}
          onCheckedChange={onChange}
        />
        <label htmlFor={id} className="text-sm font-medium cursor-pointer">
          {label}
        </label>
      </div>
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

### 1.8 File Uploader

**Streamlit Component:**
```python
st.file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None)
```

**React Equivalent:**
```typescript
import { Upload } from 'antd';
// or custom implementation
```

**BubbleLab Implementation:**
```typescript
import { useCallback, useState } from 'react';
import { Upload } from 'lucide-react';

interface FileUploaderProps {
  label: string;
  onFilesChange: (files: File[]) => void;
  accept?: string;
  multiple?: boolean;
  helperText?: string;
  id?: string;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  label,
  onFilesChange,
  accept,
  multiple = false,
  helperText,
  id
}) => {
  const [dragActive, setDragActive] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files);
    onFilesChange(multiple ? files : [files[0]]);
  }, [multiple, onFilesChange]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    onFilesChange(multiple ? files : [files[0]]);
  }, [multiple, onFilesChange]);

  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        className={`
          border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
          transition-colors
          ${dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'}
        `}
      >
        <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          Drag & drop files here or click to browse
        </p>
        <input
          id={id}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={handleChange}
          className="hidden"
        />
      </div>
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM

---

## SECTION 2: LAYOUT COMPONENTS

### 2.1 Columns

**Streamlit Component:**
```python
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Content 1")
with col2:
    st.write("Content 2")
with col3:
    st.write("Content 3")
```

**React Equivalent:**
```typescript
import { Grid } from '@mui/material';
// or
import { Row, Col } from 'antd';
// or CSS Grid
```

**BubbleLab Implementation:**
```typescript
import { cn } from '@/lib/utils';

interface ColumnsProps {
  children: React.ReactNode[];
  sizes?: number[]; // [1, 1, 1] for equal, [2, 1] for 2:1 ratio
  gap?: number;
  className?: string;
}

export const Columns: React.FC<ColumnsProps> = ({
  children,
  sizes,
  gap = 4,
  className
}) => {
  const columns = children.length;
  const defaultSizes = Array(columns).fill(1);
  const gridSizes = sizes || defaultSizes;

  return (
    <div
      className={cn(
        'grid',
        `grid-cols-${columns}`,
        `gap-${gap}`,
        className
      )}
      style={{
        gridTemplateColumns: gridSizes.map(s => `${s}fr`).join(' ')
      }}
    >
      {children.map((child, i) => (
        <div key={i}>{child}</div>
      ))}
    </div>
  );
};
```

**Usage:**
```typescript
<Columns sizes={[1, 2]}>
  <div>Left (1/3)</div>
  <div>Right (2/3)</div>
</Columns>
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 2.2 Tabs

**Streamlit Component:**
```python
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
with tab1:
    st.write("Content 1")
with tab2:
    st.write("Content 2")
with tab3:
    st.write("Content 3")
```

**React Equivalent:**
```typescript
import { Tabs } from '@mui/material';
// or
import { Tabs } from 'antd';
```

**BubbleLab Implementation:**
```typescript
import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface TabItem {
  label: string;
  content: React.ReactNode;
}

interface CustomTabsProps {
  tabs: TabItem[];
  defaultValue?: string;
  className?: string;
}

export const CustomTabs: React.FC<CustomTabsProps> = ({
  tabs,
  defaultValue,
  className
}) => {
  const [activeTab, setActiveTab] = useState(defaultValue || tabs[0]?.label);

  return (
    <Tabs value={activeTab} onValueChange={setActiveTab} className={className}>
      <TabsList>
        {tabs.map((tab) => (
          <TabsTrigger key={tab.label} value={tab.label}>
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>
      {tabs.map((tab) => (
        <TabsContent key={tab.label} value={tab.label}>
          {tab.content}
        </TabsContent>
      ))}
    </Tabs>
  );
};
```

**Usage:**
```typescript
<CustomTabs
  tabs={[
    { label: 'Overview', content: <div>Overview content</div> },
    { label: 'Settings', content: <div>Settings content</div> }
  ]}
/>
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 2.3 Expander

**Streamlit Component:**
```python
with st.expander("Click to expand"):
    st.write("Hidden content")
```

**React Equivalent:**
```typescript
import { Accordion } from '@mui/material';
// or
import { Collapse } from 'antd';
```

**BubbleLab Implementation:**
```typescript
import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ExpanderProps {
  label: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  className?: string;
}

export const Expander: React.FC<ExpanderProps> = ({
  label,
  children,
  defaultExpanded = false,
  className
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div className={cn('border rounded-lg', className)}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-muted/50 transition-colors"
      >
        <span className="font-medium">{label}</span>
        <ChevronDown
          className={cn(
            'transition-transform',
            expanded ? 'transform rotate-180' : ''
          )}
        />
      </button>
      {expanded && (
        <div className="px-4 py-3 border-t">
          {children}
        </div>
      )}
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 2.4 Container

**Streamlit Component:**
```python
with st.container():
    st.write("Container content")
```

**React Equivalent:**
```typescript
// Direct div or Paper component
import { Paper } from '@mui/material';
```

**BubbleLab Implementation:**
```typescript
import { cn } from '@/lib/utils';

interface ContainerProps {
  children: React.ReactNode;
  border?: boolean;
  padding?: number;
  className?: string;
}

export const Container: React.FC<ContainerProps> = ({
  children,
  border = false,
  padding = 4,
  className
}) => {
  return (
    <div
      className={cn(
        'rounded-lg',
        border && 'border',
        `p-${padding}`,
        className
      )}
    >
      {children}
    </div>
  );
};
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

### 2.5 Sidebar

**Streamlit Component:**
```python
with st.sidebar:
    st.write("Sidebar content")
```

**React Equivalent:**
```typescript
// React Router sidebar layout
// or custom sidebar component
```

**BubbleLab Implementation:**
```typescript
import { Outlet } from 'react-router-dom';
import { Sidebar } from '@/components/layout/sidebar';
import { Header } from '@/components/layout/header';

export function AppLayout() {
  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM (layout change)

---

## SECTION 3: DISPLAY COMPONENTS

### 3.1 Markdown

**Streamlit Component:**
```python
st.markdown(text, unsafe_allow_html=False)
```

**React Equivalent:**
```typescript
import ReactMarkdown from 'react-markdown';
// or
import { Typography } from '@mui/material';
```

**BubbleLab Implementation:**
```typescript
import ReactMarkdown from 'react-markdown';
import { cn } from '@/lib/utils';

interface MarkdownProps {
  content: string;
  allowHTML?: boolean;
  className?: string;
}

export const Markdown: React.FC<MarkdownProps> = ({
  content,
  allowHTML = false,
  className
}) => {
  return (
    <div className={cn('prose dark:prose-invert max-w-none', className)}>
      <ReactMarkdown
        components={{
          // Custom components for markdown elements
          h1: ({ node, ...props }) => <h1 className="text-3xl font-bold" {...props} />,
          h2: ({ node, ...props }) => <h2 className="text-2xl font-bold" {...props} />,
          code: ({ node, inline, ...props }) =>
            inline ? (
              <code className="bg-muted px-1 py-0.5 rounded text-sm" {...props} />
            ) : (
              <code className="block bg-muted p-4 rounded-lg text-sm" {...props} />
            )
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 3.2 Code Display

**Streamlit Component:**
```python
st.code(code, language="python")
```

**React Equivalent:**
```typescript
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
```

**BubbleLab Implementation:**
```typescript
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '@/lib/utils';

interface CodeDisplayProps {
  code: string;
  language?: string;
  className?: string;
}

export const CodeDisplay: React.FC<CodeDisplayProps> = ({
  code,
  language = 'python',
  className
}) => {
  return (
    <div className={cn('rounded-lg overflow-hidden', className)}>
      <SyntaxHighlighter
        language={language}
        style={vscDarkPlus}
        showLineNumbers
        customStyle={{
          margin: 0,
          borderRadius: '0.5rem'
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 3.3 JSON Display

**Streamlit Component:**
```python
st.json(data)
```

**React Equivalent:**
```typescript
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// format JSON before display
```

**BubbleLab Implementation:**
```typescript
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '@/lib/utils';

interface JSONDisplayProps {
  data: any;
  className?: string;
}

export const JSONDisplay: React.FC<JSONDisplayProps> = ({ data, className }) => {
  const formattedJSON = JSON.stringify(data, null, 2);

  return (
    <div className={cn('rounded-lg overflow-hidden', className)}>
      <SyntaxHighlighter
        language="json"
        style={vscDarkPlus}
        showLineNumbers
        customStyle={{
          margin: 0,
          borderRadius: '0.5rem'
        }}
      >
        {formattedJSON}
      </SyntaxHighlighter>
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 3.4 DataFrame

**Streamlit Component:**
```python
st.dataframe(data)
```

**React Equivalent:**
```typescript
import { DataGrid } from '@mui/x-data-grid';
// or
import Table from '@/components/ui/table';
// or TanStack Table
```

**BubbleLab Implementation:**
```typescript
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { cn } from '@/lib/utils';

interface DataFrameProps {
  data: Record<string, any>[];
  className?: string;
}

export const DataFrame: React.FC<DataFrameProps> = ({ data, className }) => {
  if (!data || data.length === 0) {
    return <div className="text-muted-foreground">No data available</div>;
  }

  const columns = Object.keys(data[0]);

  return (
    <div className={cn('rounded-lg border', className)}>
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((column) => (
              <TableHead key={column}>{column}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row, i) => (
            <TableRow key={i}>
              {columns.map((column) => (
                <TableCell key={`${i}-${column}`}>
                  {String(row[column] ?? '')}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM

---

### 3.5 Metrics

**Streamlit Component:**
```python
st.metric(label, value, delta=None)
```

**React Equivalent:**
```typescript
// Custom metric card component
```

**BubbleLab Implementation:**
```typescript
import { ArrowDown, ArrowUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MetricProps {
  label: string;
  value: string | number;
  delta?: number;
  deltaColor?: 'increase' | 'decrease';
  className?: string;
}

export const Metric: React.FC<MetricProps> = ({
  label,
  value,
  delta,
  deltaColor = 'increase',
  className
}) => {
  return (
    <div className={cn('rounded-lg border p-6 space-y-2', className)}>
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
      {delta !== undefined && (
        <div
          className={cn(
            'flex items-center text-sm',
            delta > 0 ? 'text-green-600' : 'text-red-600'
          )}
        >
          {delta > 0 ? <ArrowUp className="h-4 w-4 mr-1" /> : <ArrowDown className="h-4 w-4 mr-1" />}
          {Math.abs(delta)}%
        </div>
      )}
    </div>
  );
};
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 3.6 Plotly Charts

**Streamlit Component:**
```python
st.plotly_chart(fig)
```

**React Equivalent:**
```typescript
import Plot from 'react-plotly.js';
// or migrate to Recharts
```

**BubbleLab Implementation:**
```typescript
import Plot from 'react-plotly.js';
import { cn } from '@/lib/utils';

interface PlotlyChartProps {
  data: any[];
  layout?: any;
  config?: any;
  className?: string;
}

export const PlotlyChart: React.FC<PlotlyChartProps> = ({
  data,
  layout,
  config,
  className
}) => {
  return (
    <div className={cn('w-full', className)}>
      <Plot
        data={data}
        layout={{
          responsive: true,
          ...layout
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          ...config
        }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};
```

**Alternative (Recharts migration):**
```typescript
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  PieChart,
  Pie,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM (requires plotly.js or recharts)

---

## SECTION 4: ACTION COMPONENTS

### 4.1 Button

**Streamlit Component:**
```python
st.button(label, key=None, type="primary", help=None)
```

**React Equivalent:**
```typescript
import { Button } from '@mui/material';
// or
import { Button } from 'antd';
```

**BubbleLab Implementation:**
```typescript
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface CustomButtonProps {
  label: string;
  onClick: () => void;
  variant?: 'default' | 'primary' | 'secondary' | 'danger' | 'ghost';
  disabled?: boolean;
  loading?: boolean;
  helperText?: string;
  id?: string;
  className?: string;
}

export const CustomButton: React.FC<CustomButtonProps> = ({
  label,
  onClick,
  variant = 'default',
  disabled = false,
  loading = false,
  helperText,
  id,
  className
}) => {
  return (
    <div className="space-y-2">
      <Button
        id={id}
        onClick={onClick}
        disabled={disabled || loading}
        variant={variant}
        className={cn('w-full', className)}
      >
        {loading ? 'Loading...' : label}
      </Button>
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

### 4.2 Download Button

**Streamlit Component:**
```python
st.download_button(label, data, file_name, mime)
```

**React Equivalent:**
```typescript
// Custom implementation with Blob and URL.createObjectURL
```

**BubbleLab Implementation:**
```typescript
import { Download } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface DownloadButtonProps {
  label: string;
  data: string | Blob;
  fileName: string;
  mimeType?: string;
  className?: string;
}

export const DownloadButton: React.FC<DownloadButtonProps> = ({
  label,
  data,
  fileName,
  mimeType = 'text/plain',
  className
}) => {
  const handleDownload = () => {
    const blob = data instanceof Blob ? data : new Blob([data], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Button onClick={handleDownload} className={className}>
      <Download className="mr-2 h-4 w-4" />
      {label}
    </Button>
  );
};
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 4.3 Form Submit Button

**Streamlit Component:**
```python
with st.form("my_form"):
    st.form_submit_button("Submit")
```

**React Equivalent:**
```typescript
// Standard form submission or button with type="submit"
```

**BubbleLab Implementation:**
```typescript
import { Button } from '@/components/ui/button';

interface FormSubmitButtonProps {
  label: string;
  disabled?: boolean;
  loading?: boolean;
  className?: string;
}

export const FormSubmitButton: React.FC<FormSubmitButtonProps> = ({
  label,
  disabled = false,
  loading = false,
  className
}) => {
  return (
    <Button
      type="submit"
      disabled={disabled || loading}
      className={className}
    >
      {loading ? 'Submitting...' : label}
    </Button>
  );
};
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

## SECTION 5: STATUS & FEEDBACK COMPONENTS

### 5.1 Progress Bar

**Streamlit Component:**
```python
st.progress(value)  # 0.0 to 1.0
```

**React Equivalent:**
```typescript
import { LinearProgress } from '@mui/material';
// or
import { Progress } from '@/components/ui/progress';
```

**BubbleLab Implementation:**
```typescript
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';

interface ProgressBarProps {
  value: number; // 0-100
  label?: string;
  className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  label,
  className
}) => {
  return (
    <div className={cn('space-y-2', className)}>
      {label && <p className="text-sm font-medium">{label}</p>}
      <Progress value={value} className="w-full" />
      <p className="text-xs text-muted-foreground">{value}%</p>
    </div>
  );
};
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

### 5.2 Spinner

**Streamlit Component:**
```python
with st.spinner("Loading..."):
    time.sleep(2)
```

**React Equivalent:**
```typescript
import { CircularProgress } from '@mui/material';
// or
import { Loader2 } from 'lucide-react';
```

**BubbleLab Implementation:**
```typescript
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SpinnerProps {
  message?: string;
  size?: number;
  className?: string;
}

export const Spinner: React.FC<SpinnerProps> = ({
  message,
  size = 24,
  className
}) => {
  return (
    <div className={cn('flex items-center space-x-2', className)}>
      <Loader2 className="animate-spin" style={{ width: size, height: size }} />
      {message && <p className="text-sm text-muted-foreground">{message}</p>}
    </div>
  );
};
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

### 5.3 Info/Warning/Error/Success

**Streamlit Components:**
```python
st.info("Info message")
st.warning("Warning message")
st.error("Error message")
st.success("Success message")
```

**React Equivalent:**
```typescript
import { Alert } from '@mui/material';
// or
import { Alert } from '@/components/ui/alert';
```

**BubbleLab Implementation:**
```typescript
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Info, AlertTriangle, XCircle, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

type AlertType = 'info' | 'warning' | 'error' | 'success';

interface StatusAlertProps {
  type: AlertType;
  message: string;
  className?: string;
}

const alertConfig = {
  info: {
    icon: Info,
    className: 'border-blue-500 text-blue-500'
  },
  warning: {
    icon: AlertTriangle,
    className: 'border-yellow-500 text-yellow-500'
  },
  error: {
    icon: XCircle,
    className: 'border-red-500 text-red-500'
  },
  success: {
    icon: CheckCircle,
    className: 'border-green-500 text-green-500'
  }
};

export const StatusAlert: React.FC<StatusAlertProps> = ({
  type,
  message,
  className
}) => {
  const config = alertConfig[type];
  const Icon = config.icon;

  return (
    <Alert className={cn('border-l-4', config.className, className)}>
      <Icon className="h-4 w-4" />
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  );
};
```

**Usage:**
```typescript
<StatusAlert type="info" message="This is an info message" />
<StatusAlert type="warning" message="This is a warning" />
<StatusAlert type="error" message="This is an error" />
<StatusAlert type="success" message="Success!" />
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL

---

### 5.4 Empty Placeholder

**Streamlit Component:**
```python
placeholder = st.empty()
placeholder.text("Initial text")
placeholder.text("Updated text")
```

**React Equivalent:**
```typescript
// useState for dynamic content
```

**BubbleLab Implementation:**
```typescript
import { useState } from 'react';
import { cn } from '@/lib/utils';

interface EmptyPlaceholderProps {
  children: React.ReactNode;
  className?: string;
}

export const EmptyPlaceholder: React.FC<EmptyPlaceholderProps> = ({
  children,
  className
}) => {
  return (
    <div className={cn('min-h-[200px] flex items-center justify-center', className)}>
      {children}
    </div>
  );
};
```

**Usage:**
```typescript
const [content, setContent] = useState('Initial');

return (
  <EmptyPlaceholder>
    <p>{content}</p>
    <button onClick={() => setContent('Updated')}>
      Update
    </button>
  </EmptyPlaceholder>
);
```

**Migration Complexity:** ⭐ (1/5) - TRIVIAL (fundamental React pattern)

---

## SECTION 6: ADVANCED COMPONENTS

### 6.1 Session State

**Streamlit Component:**
```python
# Read/Write
st.session_state["key"] = "value"
value = st.session_state["key"]
```

**React Equivalent:**
```typescript
// React Context + useState
// or Zustand/Redux for global state
```

**BubbleLab Implementation:**
```typescript
// React Context approach
import { createContext, useContext, useState, ReactNode } from 'react';

interface SessionState {
  [key: string]: any;
}

interface SessionContextType {
  state: SessionState;
  setState: (key: string, value: any) => void;
  getState: (key: string) => any;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SessionState>({});

  const setValue = (key: string, value: any) => {
    setState(prev => ({ ...prev, [key]: value }));
  };

  const getValue = (key: string) => {
    return state[key];
  };

  return (
    <SessionContext.Provider value={{ state, setState: setValue, getState: getValue }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSessionState() {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSessionState must be used within SessionProvider');
  }
  return context;
}
```

**Alternative (Zustand):**
```typescript
import { create } from 'zustand';

interface SessionStore {
  state: Record<string, any>;
  setState: (key: string, value: any) => void;
  getState: (key: string) => any;
}

export const useSessionStore = create<SessionStore>((set, get) => ({
  state: {},
  setState: (key, value) =>
    set((prev) => ({
      state: { ...prev.state, [key]: value }
    })),
  getState: (key) => get().state[key]
}));
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM (architectural change)

---

### 6.2 Forms

**Streamlit Component:**
```python
with st.form("my_form"):
    name = st.text_input("Name")
    age = st.number_input("Age")
    submitted = st.form_submit_button("Submit")
    if submitted:
        # Process form data
```

**React Equivalent:**
```typescript
// react-hook-form
// or Formik
```

**BubbleLab Implementation:**
```typescript
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { TextInput } from './text-input';
import { NumberInput } from './number-input';

const formSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters'),
  age: z.number().min(0).max(120),
});

interface MyFormProps {
  onSubmit: (data: z.infer<typeof formSchema>) => void;
}

export function MyForm({ onSubmit }: MyFormProps) {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      name: '',
      age: 0,
    },
  });

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Name</FormLabel>
              <FormControl>
                <TextInput
                  label="Name"
                  value={field.value}
                  onChange={field.onChange}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="age"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Age</FormLabel>
              <FormControl>
                <NumberInput
                  label="Age"
                  value={field.value}
                  onChange={field.onChange}
                  min={0}
                  max={120}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit">Submit</Button>
      </form>
    </Form>
  );
}
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM

---

### 6.3 Auto Refresh

**Streamlit Component:**
```python
from streamlit_autorefresh import st_autorefresh

count = st_autorefresh(interval=2000, limit=100, key="refresh")
```

**React Equivalent:**
```typescript
import { useEffect, useState } from 'react';
// or React Query refetchInterval
```

**BubbleLab Implementation:**
```typescript
import { useEffect, useState, useRef } from 'react';

interface UseAutoRefreshProps {
  interval: number; // milliseconds
  limit?: number;
  onRefresh: () => void;
}

export function useAutoRefresh({ interval, limit, onRefresh }: UseAutoRefreshProps) {
  const [count, setCount] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (limit && count >= limit) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      return;
    }

    intervalRef.current = setInterval(() => {
      onRefresh();
      setCount(prev => prev + 1);
    }, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [count, interval, limit, onRefresh]);

  return count;
}
```

**Alternative (React Query):**
```typescript
import { useQuery } from '@tanstack/react-query';

function MyComponent() {
  const { data } = useQuery({
    queryKey: ['data'],
    queryFn: fetchData,
    refetchInterval: 2000, // Auto-refresh every 2 seconds
  });

  return <div>{data}</div>;
}
```

**Migration Complexity:** ⭐⭐ (2/5) - SIMPLE

---

### 6.4 Tag Input

**Streamlit Component:**
```python
from streamlit_tags import st_tags

tags = st_tags(
  label='Enter tags:',
  text='Press enter to add more',
  value=['green', 'red', 'blue'],
  suggestions=['green', 'yellow', 'red', 'blue'],
  maxtags=-1
)
```

**React Equivalent:**
```typescript
// Custom implementation or react-tag-input
```

**BubbleLab Implementation:**
```typescript
import { useState, KeyboardEvent } from 'react';
import { X } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';

interface TagInputProps {
  value: string[];
  onChange: (tags: string[]) => void;
  suggestions?: string[];
  maxTags?: number;
  placeholder?: string;
  className?: string;
}

export function TagInput({
  value,
  onChange,
  suggestions = [],
  maxTags = -1,
  placeholder = 'Type and press Enter...',
  className
}: TagInputProps) {
  const [input, setInput] = useState('');

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const newTag = input.trim();
      if (newTag && !value.includes(newTag)) {
        if (maxTags === -1 || value.length < maxTags) {
          onChange([...value, newTag]);
          setInput('');
        }
      }
    } else if (e.key === 'Backspace' && !input && value.length > 0) {
      onChange(value.slice(0, -1));
    }
  };

  const removeTag = (tagToRemove: string) => {
    onChange(value.filter(tag => tag !== tagToRemove));
  };

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex flex-wrap gap-2">
        {value.map(tag => (
          <Badge key={tag} variant="secondary">
            {tag}
            <button
              onClick={() => removeTag(tag)}
              className="ml-2 hover:text-destructive"
            >
              <X className="h-3 w-3" />
            </button>
          </Badge>
        ))}
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="flex-1 min-w-[200px]"
        />
      </div>
      {suggestions.length > 0 && (
        <div className="flex flex-wrap gap-2">
          <span className="text-xs text-muted-foreground">Suggestions:</span>
          {suggestions
            .filter(s => !value.includes(s))
            .slice(0, 5)
            .map(suggestion => (
              <Badge
                key={suggestion}
                variant="outline"
                className="cursor-pointer hover:bg-accent"
                onClick={() => onChange([...value, suggestion])}
              >
                + {suggestion}
              </Badge>
            ))}
        </div>
      )}
    </div>
  );
}
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM

---

## SECTION 7: SPECIALIZED COMPONENTS

### 7.1 Chat Interface

**Streamlit Component:**
```python
st.chat_message("user")
st.write("User message")

st.chat_message("assistant")
st.write("Assistant response")
```

**React Equivalent:**
```typescript
// Custom chat component with message list and input
```

**BubbleLab Implementation:**
```typescript
import { useState } from 'react';
import { Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatInterfaceProps {
  messages: Message[];
  onSendMessage: (message: string) => void;
  className?: string;
}

export function ChatInterface({ messages, onSendMessage, className }: ChatInterfaceProps) {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim()) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  return (
    <div className={cn('flex flex-col h-[600px]', className)}>
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {messages.map((message, i) => (
            <div
              key={i}
              className={cn(
                'flex',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              <div
                className={cn(
                  'max-w-[70%] rounded-lg p-3',
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted'
                )}
              >
                {message.content}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
      <div className="border-t p-4">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Type your message..."
            className="flex-1"
          />
          <Button onClick={handleSend}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM

---

### 7.2 Status (New Component)

**Streamlit Component:**
```python
with st.status("Running...") as status:
    status.write("Step 1 complete")
    status.update(label="Still running...")
    status.update(label="Done!", state="complete", expanded=False)
```

**React Equivalent:**
```typescript
// Custom status component with state management
```

**BubbleLab Implementation:**
```typescript
import { useState } from 'react';
import { CheckCircle2, Loader2, XCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';

type StatusState = 'running' | 'complete' | 'error';

interface StatusLog {
  timestamp: Date;
  message: string;
}

interface StatusProps {
  label: string;
  state: StatusState;
  logs?: StatusLog[];
  expanded?: boolean;
  onCollapse?: () => void;
  className?: string;
}

export function Status({ label, state, logs = [], expanded = true, onCollapse, className }: StatusProps) {
  const statusConfig = {
    running: {
      icon: Loader2,
      iconClassName: 'animate-spin text-blue-500',
      borderClass: 'border-blue-500'
    },
    complete: {
      icon: CheckCircle2,
      iconClassName: 'text-green-500',
      borderClass: 'border-green-500'
    },
    error: {
      icon: XCircle,
      iconClassName: 'text-red-500',
      borderClass: 'border-red-500'
    }
  };

  const config = statusConfig[state];
  const Icon = config.icon;

  return (
    <div className={cn('border rounded-lg overflow-hidden', config.borderClass, className)}>
      <div className="flex items-center justify-between p-4 bg-muted/50">
        <div className="flex items-center gap-2">
          <Icon className={cn('h-5 w-5', config.iconClassName)} />
          <span className="font-medium">{label}</span>
        </div>
        {state === 'complete' && (
          <Button variant="ghost" size="sm" onClick={onCollapse}>
            Collapse
          </Button>
        )}
      </div>
      {expanded && logs.length > 0 && (
        <ScrollArea className="h-[200px] p-4">
          <div className="space-y-2">
            {logs.map((log, i) => (
              <div key={i} className="text-sm">
                <span className="text-muted-foreground">
                  {log.timestamp.toLocaleTimeString()} -{' '}
                </span>
                <span>{log.message}</span>
              </div>
            ))}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}
```

**Migration Complexity:** ⭐⭐⭐ (3/5) - MEDIUM

---

## SECTION 8: MIGRATION COMPLEXITY SUMMARY

### Complexity Breakdown

**⭐ TRIVIAL (1/5):**
- Text Input, Number Input, Checkbox
- Button, Form Submit Button
- Progress Bar, Spinner, Alerts
- Empty Placeholder, Container

**⭐⭐ SIMPLE (2/5):**
- Text Area, Slider, Select Box
- Columns, Tabs, Expander
- Markdown, Code Display, JSON Display
- Metrics, Download Button
- Auto Refresh

**⭐⭐⭐ MEDIUM (3/5):**
- Multi Select, File Uploader
- DataFrame, Plotly Charts
- Session State, Forms, Tag Input
- Chat Interface, Status Component
- Sidebar (layout change)

**⭐⭐⭐⭐ HIGH (4/5):**
- Custom visualization components
- Real-time collaboration components
- Complex workflow editors

**⭐⭐⭐⭐⭐ VERY HIGH (5/5):**
- Complete architectural changes
- Multi-page applications
- WebSocket integrations

---

## SECTION 9: DEPENDENCY REQUIREMENTS

### Required npm Packages

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "react-hook-form": "^7.48.0",
    "@hookform/resolvers": "^3.3.0",
    "zod": "^3.22.0",
    "recharts": "^2.10.0",
    "react-plotly.js": "^2.6.0",
    "react-syntax-highlighter": "^15.5.0",
    "react-markdown": "^9.0.0",
    "lucide-react": "^0.294.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.1.0"
  }
}
```

### Optional Packages

```json
{
  "dependencies": {
    "@mui/material": "^5.14.0",
    "antd": "^5.12.0",
    "react-tag-input": "^6.8.0",
    "react-diff-viewer": "^3.1.0",
    "vis-network": "^9.1.0",
    "cytoscape": "^3.26.0",
    "monaco-editor": "^0.45.0"
  }
}
```

---

**END OF COMPONENT MAPPING MATRIX**

**Last Updated:** 2025-01-05
**Status:** COMPLETE - Ready for UI migration implementation
