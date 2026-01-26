/**
 * DOM Utilities
 * Helper functions for DOM manipulation
 */

/**
 * Get element by id
 */
export function getById<T extends HTMLElement = HTMLElement>(id: string): T | null {
  return document.getElementById(id) as T | null;
}

/**
 * Query selector
 */
export function query<T extends HTMLElement = HTMLElement>(
  selector: string,
  parent: HTMLElement | Document = document
): T | null {
  return parent.querySelector<T>(selector);
}

/**
 * Query selector all
 */
export function queryAll<T extends HTMLElement = HTMLElement>(
  selector: string,
  parent: HTMLElement | Document = document
): T[] {
  return Array.from(parent.querySelectorAll<T>(selector));
}

/**
 * Add class to element
 */
export function addClass(element: HTMLElement, className: string): void {
  element.classList.add(className);
}

/**
 * Remove class from element
 */
export function removeClass(element: HTMLElement, className: string): void {
  element.classList.remove(className);
}

/**
 * Toggle class on element
 */
export function toggleClass(element: HTMLElement, className: string): void {
  element.classList.toggle(className);
}

/**
 * Check if element has class
 */
export function hasClass(element: HTMLElement, className: string): boolean {
  return element.classList.contains(className);
}

/**
 * Set style on element
 */
export function setStyle(element: HTMLElement, styles: Partial<CSSStyleDeclaration>): void {
  Object.assign(element.style, styles);
}

/**
 * Get element's offset relative to document
 */
export function getOffset(element: HTMLElement): { top: number; left: number } {
  const rect = element.getBoundingClientRect();
  return {
    top: rect.top + window.scrollY,
    left: rect.left + window.scrollX,
  };
}

/**
 * Get element's dimensions
 */
export function getDimensions(element: HTMLElement): { width: number; height: number } {
  return {
    width: element.offsetWidth,
    height: element.offsetHeight,
  };
}

/**
 * Check if element is in viewport
 */
export function isInViewport(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

/**
 * Scroll element into view
 */
export function scrollIntoView(element: HTMLElement, behavior: ScrollBehavior = 'smooth'): void {
  element.scrollIntoView({ behavior, block: 'nearest' });
}

/**
 * Add event listener with automatic cleanup
 */
export function addEventListener<K extends keyof HTMLElementEventMap>(
  element: HTMLElement,
  type: K,
  listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => unknown,
  options?: boolean | AddEventListenerOptions
): () => void {
  element.addEventListener(type, listener, options);
  return () => element.removeEventListener(type, listener, options);
}

/**
 * Prevent default event
 */
export function preventDefault(event: Event): void {
  event.preventDefault();
}

/**
 * Stop event propagation
 */
export function stopPropagation(event: Event): void {
  event.stopPropagation();
}

/**
 * Focus element
 */
export function focus(element: HTMLElement): void {
  element.focus();
}

/**
 * Blur element
 */
export function blur(element: HTMLElement): void {
  element.blur();
}

/**
 * Create element with attributes
 */
export function createElement<K extends keyof HTMLElementTagNameMap>(
  tagName: K,
  attributes?: Partial<HTMLElementTagNameMap[K]> & {
    className?: string;
    innerHTML?: string;
  }
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tagName);

  if (attributes) {
    const { className, innerHTML, ...attrs } = attributes;
    if (className) element.className = className;
    if (innerHTML) element.innerHTML = innerHTML;

    Object.entries(attrs).forEach(([key, value]) => {
      if (key in element) {
        (element as unknown as Record<string, unknown>)[key] = value;
      } else {
        element.setAttribute(key, String(value));
      }
    });
  }

  return element;
}

/**
 * Remove element from DOM
 */
export function removeElement(element: HTMLElement): void {
  element.remove();
}

/**
 * Insert element after reference
 */
export function insertAfter(newElement: HTMLElement, referenceElement: HTMLElement): void {
  referenceElement.parentNode?.insertBefore(newElement, referenceElement.nextSibling);
}

/**
 * Insert element before reference
 */
export function insertBefore(newElement: HTMLElement, referenceElement: HTMLElement): void {
  referenceElement.parentNode?.insertBefore(newElement, referenceElement);
}
