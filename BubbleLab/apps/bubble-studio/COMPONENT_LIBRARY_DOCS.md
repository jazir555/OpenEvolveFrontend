# Component Library Documentation

OpenEvolve Frontend - BubbleLab UI Component Library

---

## Table of Contents

- [Form Components](#form-components)
- [Layout Components](#layout-components)
- [Feedback Components](#feedback-components)
- [Navigation Components](#navigation-components)
- [Data Display](#data-display)
- [Button Components](#button-components)
- [Modal Components](#modal-components)

---

## Form Components

### Input

Basic text input field.

```tsx
import { Input } from '@/components/common/Input';

<Input
  label="Email"
  value={email}
  onChange={(e) => setEmail(e.target.value)}
  placeholder="Enter your email"
  required
  error={error}
/>
```

**Props:**
- `label?: string` - Field label
- `value: string` - Input value
- `onChange: (value: string) => void` - Change handler
- `placeholder?: string` - Placeholder text
- `type?: string` - Input type (default: 'text')
- `required?: boolean` - Required field indicator
- `disabled?: boolean` - Disable input
- `error?: string` - Error message
- `helperText?: string` - Helper text
- `className?: string` - Additional classes

---

### Textarea

Multi-line text input.

```tsx
import { Textarea } from '@/components/common/Textarea';

<Textarea
  label="Description"
  value={description}
  onChange={(e) => setDescription(e.target.value)}
  rows={4}
  placeholder="Enter a description"
/>
```

**Props:** Same as Input, plus:
- `rows?: number` - Number of rows (default: 3)

---

### Select

Dropdown select component.

```tsx
import { Select } from '@/components/common/Select';

<Select
  label="Category"
  value={category}
  onChange={setCategory}
  options={[
    { value: 'math', label: 'Mathematics' },
    { value: 'code', label: 'Coding' },
  ]}
  placeholder="Select a category"
/>
```

**Props:**
- `label?: string`
- `value: string`
- `onChange: (value: string) => void`
- `options: { value: string; label: string }[]`
- `placeholder?: string`
- `disabled?: boolean`

---

### MultiSelect

Select multiple items from a list.

```tsx
import { MultiSelect } from '@/components/common/MultiSelect';

<MultiSelect
  label="Teams"
  value={selectedTeams}
  onChange={setSelectedTeams}
  options={teamOptions}
  placeholder="Select teams"
/>
```

---

### Checkbox

Checkbox input.

```tsx
import { Checkbox } from '@/components/common/Checkbox';

<Checkbox
  label="Accept terms"
  checked={accepted}
  onChange={setAccepted}
/>
```

---

### Slider

Range slider.

```tsx
import { Slider } from '@/components/common/Slider';

<Slider
  label="Temperature"
  value={temperature}
  onChange={setTemperature}
  min={0}
  max={1}
  step={0.1}
/>
```

---

### ToggleSwitch

On/off toggle switch.

```tsx
import { ToggleSwitch } from '@/components/common/ToggleSwitch';

<ToggleSwitch
  label="Enable notifications"
  checked={enabled}
  onChange={setEnabled}
/>
```

---

## Layout Components

### Card

Container with elevation.

```tsx
import { Card } from '@/components/common/Card';

<Card>
  <h3>Card Title</h3>
  <p>Card content goes here.</p>
</Card>
```

---

### Container

Constrained width container.

```tsx
import { Container } from '@/components/common/Container';

<Container size="lg">
  <p>Content is centered and max-width constrained.</p>
</Container>
```

---

### Divider

Visual separator.

```tsx
import { Divider } from '@/components/common/Divider';

<Divider />
<Divider orientation="vertical" />
```

---

### Stepper

Multi-step progress indicator.

```tsx
import { Stepper } from '@/components/common/Stepper';

<Stepper
  steps={[
    { id: '1', label: 'Step 1', status: 'complete' },
    { id: '2', label: 'Step 2', status: 'current' },
    { id: '3', label: 'Step 3', status: 'pending' },
  ]}
/>
```

---

## Feedback Components

### Alert

Alert banner for messages.

```tsx
import { Alert } from '@/components/common/Alert';

<Alert variant="success" title="Success!">
  Your changes have been saved.
</Alert>
```

**Variants:** `success`, `error`, `warning`, `info`

---

### Toast

Temporary notification.

```tsx
import { Toast, ToastContainer } from '@/components/common/Toast';

const toasts = [
  { id: '1', type: 'success', title: 'Saved!', message: 'Changes saved' },
];

<ToastContainer toasts={toasts} onClose={(id) => removeToast(id)} />
```

---

### LoadingSpinner

Loading indicator.

```tsx
import { LoadingSpinner } from '@/components/common/LoadingSpinner';

<LoadingSpinner size="lg" />
```

---

### Progress

Progress bar.

```tsx
import { Progress } from '@/components/common/Progress';

<Progress value={75} max={100} showLabel />
```

---

### EmptyState

Placeholder when no data.

```tsx
import { EmptyState } from '@/components/common/EmptyState';

<EmptyState
  title="No workflows found"
  description="Create your first workflow to get started"
  action={{ label: 'Create Workflow', onClick: handleCreate }}
/>
```

---

## Navigation Components

### Tabs

Tabbed content interface.

```tsx
import { Tabs } from '@/components/common/Tabs';

<Tabs
  tabs={[
    { id: 'tab1', label: 'Tab 1', content: <div>Content 1</div> },
    { id: 'tab2', label: 'Tab 2', content: <div>Content 2</div> },
  ]}
  defaultTab="tab1"
/>
```

---

### Breadcrumbs

Navigation breadcrumb trail.

```tsx
import { Breadcrumbs } from '@/components/common/Breadcrumbs';

<Breadcrumbs
  items={[
    { label: 'Home', to: '/' },
    { label: 'Workflows', to: '/workflows' },
    { label: 'Detail', to: '/workflows/1' },
  ]}
/>
```

---

### Tooltip

Hover tooltip.

```tsx
import { Tooltip } from '@/components/common/Tooltip';

<Tooltip content="This is a tooltip">
  <button>Hover me</button>
</Tooltip>
```

---

## Data Display

### Badge

Status badge.

```tsx
import { Badge } from '@/components/common/Badge';

<Badge variant="success">Active</Badge>
```

**Variants:** `success`, `error`, `warning`, `info`, `gray`

---

### Avatar

User avatar.

```tsx
import { Avatar } from '@/components/common/Avatar';

<Avatar
  src="/avatar.jpg"
  name="John Doe"
  size="md"
/>
```

---

### Table

Data table.

```tsx
import { DataTable } from '@/components/common/DataTable';

<DataTable
  columns={[
    { key: 'name', label: 'Name' },
    { key: 'email', label: 'Email' },
  ]}
  data={users}
/>
```

---

### CodeBlock

Code display.

```tsx
import { CodeBlock } from '@/components/common/CodeBlock';

<CodeBlock
  language="typescript"
  code="const x = 1;"
/>
```

---

## Button Components

### Button

Button with variants.

```tsx
import { Button } from '@/components/common/Button';

<Button variant="primary" onClick={handleClick}>
  Click me
</Button>
```

**Variants:** `primary`, `secondary`, `success`, `danger`, `ghost`

**Sizes:** `xs`, `sm`, `md`, `lg`, `xl`

---

## Modal Components

### Modal

Dialog modal.

```tsx
import { Modal } from '@/components/common/Modal';

<Modal
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  title="Modal Title"
  size="md"
>
  <p>Modal content goes here.</p>
</Modal>
```

---

### Drawer

Slide-out panel.

```tsx
import { Drawer, DrawerHeader, DrawerBody, DrawerFooter } from '@/components/common/Drawer';

<Drawer isOpen={isOpen} onClose={() => setIsOpen(false)}>
  <DrawerHeader title="Drawer" onClose={() => setIsOpen(false)} />
  <DrawerBody>
    <p>Drawer content</p>
  </DrawerBody>
  <DrawerFooter>
    <Button onClick={() => setIsOpen(false)}>Close</Button>
  </DrawerFooter>
</Drawer>
```

---

## Performance Components

### VirtualList

Efficiently render large lists.

```tsx
import { VirtualList } from '@/components/common/VirtualList';

<VirtualList
  items={largeDataset}
  itemHeight={50}
  renderItem={(item) => <div>{item.name}</div>}
  containerHeight={400}
/>
```

---

### LazyLoad

Lazy load content when in viewport.

```tsx
import { LazyLoad } from '@/components/common/LazyLoad';

<LazyLoad fallback={<Skeleton />}>
  <ExpensiveComponent />
</LazyLoad>
```

---

### InfiniteScroll

Load more as user scrolls.

```tsx
import { InfiniteScroll } from '@/components/common/InfiniteScroll';

<InfiniteScroll
  hasMore={hasMore}
  isLoading={isLoading}
  onLoadMore={loadMore}
>
  {items.map((item) => <Item key={item.id} data={item} />)}
</InfiniteScroll>
```

---

## Custom Hooks

### useForm

Form state management.

```tsx
import { useForm } from '@/hooks/useForm';

const { values, errors, isValid, handleSubmit } = useForm({
  initialValues: { name: '' },
  validate: (values) => ({
    name: values.name ? null : 'Name is required',
  }),
  onSubmit: async (values) => {
    await saveData(values);
  },
});
```

### useLocalStorage

Sync state to localStorage.

```tsx
import { useLocalStorage } from '@/hooks/useLocalStorage';

const [theme, setTheme] = useLocalStorage('theme', 'light');
```

### useDebounce

Debounce value changes.

```tsx
import { useDebounce } from '@/hooks/useDebounce';

const debouncedSearch = useDebounce(searchTerm, 500);
```

### useKeyboardShortcuts

Register keyboard shortcuts.

```tsx
import { useKeyboardShortcut } from '@/hooks/useKeyboardShortcuts';

useKeyboardShortcut('k', () => setOpen(true), {
  ctrlKey: true,
});
```

---

## Utility Functions

### Validation

```tsx
import { validateEmail, validateUrl } from '@/utils/validation';

validateEmail('test@example.com'); // true
validateUrl('https://example.com'); // true
```

### Format

```tsx
import { formatDate, formatCurrency } from '@/utils/format';

formatDate(new Date()); // "Jan 1, 2025"
formatCurrency(1234.56); // "$1,234.56"
```

### Storage

```tsx
import { storageGet, storageSet } from '@/utils/storage';

storageSet('key', { data: 'value' });
const value = storageGet('key', {});
```

---

## Dark Mode

All components support dark mode automatically via Tailwind's `dark:` prefix.

```tsx
<html class="dark">
  <!-- Components automatically use dark mode styles -->
</html>
```

---

## Accessibility

All components follow WCAG 2.1 AA guidelines:
- Proper ARIA labels
- Keyboard navigation
- Focus indicators
- Screen reader support
- Color contrast ratios

---

## TypeScript Support

All components are fully typed with TypeScript:

```tsx
import type { ButtonProps } from '@/components/common/Button';

const buttonProps: ButtonProps = {
  variant: 'primary',
  size: 'md',
  onClick: () => {},
};
```

---

## Contributing

When adding new components:

1. Create component file in `components/common/`
2. Add TypeScript interfaces for props
3. Include JSDoc comments
4. Create test file (ComponentName.test.tsx)
5. Add to this documentation
6. Export from `components/common/index.ts`

---

**For more examples, see the storybook or source files.**
