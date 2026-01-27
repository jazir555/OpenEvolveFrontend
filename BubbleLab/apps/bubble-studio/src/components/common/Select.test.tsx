/**
 * Select Component Tests
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { Select } from './Select';

describe('Select', () => {
  const options = [
    { value: 'option1', label: 'Option 1' },
    { value: 'option2', label: 'Option 2' },
    { value: 'option3', label: 'Option 3' },
  ];

  it('renders with options', () => {
    render(
      <Select
        value="option1"
        onChange={() => {}}
        options={options}
      />
    );
    expect(screen.getByText('Option 1')).toBeInTheDocument();
  });

  it('displays selected value', () => {
    render(
      <Select
        value="option2"
        onChange={() => {}}
        options={options}
      />
    );
    expect(screen.getByText('Option 2')).toBeInTheDocument();
  });

  it('calls onChange when option is selected', () => {
    const handleChange = vi.fn();
    render(
      <Select
        value="option1"
        onChange={handleChange}
        options={options}
      />
    );

    fireEvent.click(screen.getByRole('button'));
    fireEvent.click(screen.getByText('Option 2'));

    expect(handleChange).toHaveBeenCalledWith('option2');
  });

  it('disables when disabled prop is true', () => {
    render(
      <Select
        value="option1"
        onChange={() => {}}
        options={options}
        disabled
      />
    );
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('shows label', () => {
    render(
      <Select
        value="option1"
        onChange={() => {}}
        options={options}
        label="Choose an option"
      />
    );
    expect(screen.getByText('Choose an option')).toBeInTheDocument();
  });

  it('shows placeholder when no value', () => {
    render(
      <Select
        value=""
        onChange={() => {}}
        options={options}
        placeholder="Select..."
      />
    );
    expect(screen.getByText('Select...')).toBeInTheDocument();
  });
});
