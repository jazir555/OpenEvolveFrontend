/**
 * FormField Component Tests
 */

import { render, screen } from '@testing-library/react';
import { FormField } from './FormField';
import { Input } from './Input';

describe('FormField', () => {
  it('renders label', () => {
    render(
      <FormField label="Email" required>
        <Input value="" onChange={() => {}} />
      </FormField>
    );
    expect(screen.getByText('Email')).toBeInTheDocument();
    expect(screen.getByText('*')).toBeInTheDocument();
  });

  it('renders description', () => {
    render(
      <FormField label="Email" description="Enter your email">
        <Input value="" onChange={() => {}} />
      </FormField>
    );
    expect(screen.getByText('Enter your email')).toBeInTheDocument();
  });

  it('renders error message', () => {
    render(
      <FormField label="Email" error="Invalid email">
        <Input value="" onChange={() => {}} />
      </FormField>
    );
    expect(screen.getByText('Invalid email')).toBeInTheDocument();
  });

  it('renders children', () => {
    render(
      <FormField label="Email">
        <Input value="" onChange={() => {}} />
      </FormField>
    );
    expect(screen.getByRole('textbox')).toBeInTheDocument();
  });
});
