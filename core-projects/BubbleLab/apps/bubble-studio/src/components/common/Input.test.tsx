/**
 * Input Component Tests
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Input } from './Input';

describe('Input', () => {
  it('renders with label', () => {
    render(<Input label="Email" value="" onChange={() => {}} />);
    expect(screen.getByLabelText('Email')).toBeInTheDocument();
  });

  it('renders without label', () => {
    render(<Input value="" onChange={() => {}} />);
    const input = screen.getByRole('textbox');
    expect(input).toBeInTheDocument();
  });

  it('displays value correctly', () => {
    render(<Input value="test@example.com" onChange={() => {}} />);
    expect(screen.getByDisplayValue('test@example.com')).toBeInTheDocument();
  });

  it('calls onChange when value changes', () => {
    const handleChange = vi.fn();
    render(<Input value="" onChange={handleChange} />);

    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'new value' } });

    expect(handleChange).toHaveBeenCalledTimes(1);
  });

  it('shows placeholder text', () => {
    render(
      <Input value="" onChange={() => {}} placeholder="Enter email" />
    );
    expect(screen.getByPlaceholderText('Enter email')).toBeInTheDocument();
  });

  it('disables input when disabled', () => {
    render(<Input value="" onChange={() => {}} disabled />);
    expect(screen.getByRole('textbox')).toBeDisabled();
  });

  it('shows error message', () => {
    render(
      <Input value="" onChange={() => {}} error="Invalid email" />
    );
    expect(screen.getByText('Invalid email')).toBeInTheDocument();
  });

  it('shows helper text', () => {
    render(
      <Input value="" onChange={() => {}} helperText="Enter a valid email" />
    );
    expect(screen.getByText('Enter a valid email')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<Input value="" onChange={() => {}} className="custom-class" />);
    expect(screen.getByRole('textbox')).toHaveClass('custom-class');
  });
});
