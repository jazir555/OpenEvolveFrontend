/**
 * ToggleSwitch Component Tests
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { ToggleSwitch } from './ToggleSwitch';

describe('ToggleSwitch', () => {
  it('renders label and description', () => {
    render(
      <ToggleSwitch
        checked={false}
        onChange={() => {}}
        label="Enable notifications"
        description="Receive email updates"
      />
    );
    expect(screen.getByText('Enable notifications')).toBeInTheDocument();
    expect(screen.getByText('Receive email updates')).toBeInTheDocument();
  });

  it('calls onChange when clicked', () => {
    const handleChange = vi.fn();
    render(
      <ToggleSwitch checked={false} onChange={handleChange} label="Toggle" />
    );

    fireEvent.click(screen.getByRole('switch'));
    expect(handleChange).toHaveBeenCalledWith(true);
  });

  it('is disabled when disabled prop is true', () => {
    render(
      <ToggleSwitch checked={false} onChange={() => {}} disabled label="Toggle" />
    );
    expect(screen.getByRole('switch')).toHaveClass('cursor-not-allowed');
  });

  it('shows correct checked state', () => {
    const { rerender } = render(
      <ToggleSwitch checked={true} onChange={() => {}} label="Toggle" />
    );

    const switchEl = screen.getByRole('switch');
    expect(switchEl).toHaveClass('bg-blue-600');

    rerender(
      <ToggleSwitch checked={false} onChange={() => {}} label="Toggle" />
    );

    expect(switchEl).toHaveClass('bg-gray-200');
  });
});
