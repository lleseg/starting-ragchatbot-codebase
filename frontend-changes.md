# Frontend Changes: Theme Toggle Button

Adds a light/dark theme toggle button fixed to the top-right corner of the UI.

## `frontend/index.html`
- Added `#themeToggle` `<button>` as the first element in `<body>`, with inline sun and moon SVG icons (Feather style, matching the existing send button icon). Icons are `aria-hidden`; the button carries an `aria-label` and `title`.
- Added a small inline script in `<head>` that applies the saved theme (`localStorage.theme`) to `<html data-theme>` before first paint, to avoid a flash of the wrong theme.
- Bumped cache-busting query strings on `style.css` and `script.js` from `v=9` to `v=10`.

## `frontend/style.css`
- Added a `:root[data-theme="light"]` block that overrides the existing color variables (background, surface, text, border, shadow, focus ring). Dark stays the default.
- Added `background-color`/`color`/`border-color` transitions (0.3s) on the main surfaces so the theme switch fades smoothly.
- Added `.theme-toggle` styles: 44px circular button, fixed at top-right, using `--surface`, `--border-color` and `--primary-color` on hover, like the other controls. Focus ring uses `--focus-ring` on `:focus-visible`, so keyboard users get a visible indicator.
- Icon animation: sun and moon are stacked; switching themes cross-fades them with a rotate and scale transition. In dark theme the button shows the sun (switch to light); in light theme it shows the moon.
- `prefers-reduced-motion: reduce` disables the toggle animations.
- On screens 768px wide or less, the button is smaller (40px) and sits closer to the corner.

## `frontend/script.js`
- Added `setupThemeToggle()` (called on `DOMContentLoaded`): flips `data-theme` between `light` and `dark`, saves the choice in `localStorage` (wrapped in try/catch), and updates the button's `aria-label`/`title`.
- Added `updateThemeToggleLabel()` to keep the label in sync ("Switch to light theme" / "Switch to dark theme").

## Accessibility
- The control is a native `<button type="button">`, so it is in the tab order and responds to Enter and Space without extra key handlers.
- Its accessible name describes the action it will perform and updates after each toggle.
- It has a visible focus ring and respects reduced-motion preferences.

---

# Frontend Changes: Light Theme CSS Variables

Completes the light theme started with the toggle button, and moves the remaining hardcoded colors into CSS variables so both themes control them.

## `frontend/style.css`

### New shared variables (dark defaults in `:root`, same values as before)
- `--code-bg`: inline code and `<pre>` background
- `--elevated-shadow`: welcome message shadow
- `--primary-glow`: send button hover shadow
- `--error-text`, `--error-bg`, `--error-border`: `.error-message`
- `--success-text`, `--success-bg`, `--success-border`: `.success-message`

### `:root[data-theme="light"]` palette
| Variable | Value | Note |
|---|---|---|
| `--background` | `#f8fafc` | light page background |
| `--surface` | `#ffffff` | sidebar, assistant bubbles, input |
| `--surface-hover` | `#e2e8f0` | hover state |
| `--text-primary` | `#0f172a` | 17:1 contrast on background |
| `--text-secondary` | `#475569` | 7.2:1 on background (was `#64748b`, about 4.5:1) |
| `--border-color` | `#cbd5e1` | visible on white (was `#e2e8f0`, too faint) |
| `--primary-color` / `--user-message` / `--welcome-border` | `#1d4ed8` | 6.7:1 on white; white text on user bubbles is 6.7:1 |
| `--primary-hover` | `#1e40af` | darker hover |
| `--focus-ring` | `rgba(29, 78, 216, 0.35)` | stronger ring so focus shows on light surfaces |
| `--code-bg` | `rgba(15, 23, 42, 0.06)` | subtle code shading |
| `--elevated-shadow`, `--shadow`, `--primary-glow` | low-opacity slate/blue | softer shadows |
| `--error-text` / `--error-bg` / `--error-border` | `#b91c1c` / `#fef2f2` / `#fecaca` | 5.9:1 |
| `--success-text` / `--success-bg` / `--success-border` | `#15803d` / `#f0fdf4` / `#bbf7d0` | 4.8:1 |

All text and background pairs meet WCAG 2.1 AA (4.5:1 for normal text).

### Hardcoded colors replaced with variables
- `.message-content code` and `.message-content pre` now use `var(--code-bg)`.
- The welcome message shadow now uses `var(--elevated-shadow)`.
- The `#sendButton:hover` shadow now uses `var(--primary-glow)`.
- `.error-message` and `.success-message` now use the new error and success variables.

### Bug fix
- `.message-content blockquote` referenced an undefined `var(--primary)`, so its border had no color. It now uses `var(--primary-color)`.

---

# Frontend Changes: Theme Toggle JavaScript and Smooth Transitions

Clicking the button already switched themes. This change makes the color transition cover the whole page and keeps it from affecting anything else.

## `frontend/script.js`
- Added `applyTheme(theme)`. It sets `data-theme` on `<html>`, saves the theme in `localStorage` (wrapped in try/catch), and adds the `theme-transition` class to `<html>` for 350ms.
- When the user has `prefers-reduced-motion: reduce` set, `applyTheme` skips the transition class, so the theme switches instantly.
- When the user clicks again during a transition, the timer resets so the class is not removed early.
- The click handler in `setupThemeToggle()` now calls `applyTheme()` and then updates the button label.

## `frontend/style.css`
- Replaced the fixed list of transitioning elements with a `.theme-transition` rule. While the class is on `<html>`, every element and its `::before`/`::after` transitions `background-color`, `color`, `border-color` and `box-shadow` over 0.3s.
  - Before this change, only some elements faded. Others, such as stat values, sidebar headers, sources and messages, changed instantly. Some listed elements (`#chatInput`, `.suggested-item`) had their transition overridden by their own later `transition: all 0.2s` rules.
  - The rule applies only during a switch, so hover effects keep their normal speed and the page does not fade on first load.
  - The rule uses `!important` to override per-element transitions. It excludes `.theme-toggle` and `.theme-icon`, so the button keeps its own icon rotate and fade animation.

## `frontend/index.html`
- Bumped cache-busting query strings to `v=11`.

---

# Frontend Changes: Theme Switching Cleanup

The theme toggle, light palette and transitions were already in place. This change fills the remaining gaps so that CSS variables and the `data-theme` attribute fully control both themes.

## `frontend/index.html`
- `<html>` now starts with `data-theme="dark"`. Before this change, the attribute was missing until the user toggled the theme or a saved theme loaded. Now the attribute always shows the active theme. The inline head script still replaces it with the saved theme before first paint.
- Bumped cache-busting query strings to `v=12`.

## `frontend/style.css`
- Added `color-scheme: dark` to `:root` and `color-scheme: light` to `:root[data-theme="light"]`. Native browser UI, such as scrollbars, input carets, autofill and form controls, now matches the active theme.
- Added an `--on-primary` variable (`#ffffff` in both themes) for text on primary-colored surfaces.
- Replaced the two hardcoded `color: white` rules (`.message.user .message-content` and `#sendButton`) with `var(--on-primary)`. Every color in the theme now comes from a CSS variable.

## Verification
- `script.js` checks `data-theme === 'light'`, so the explicit `dark` value does not change toggle behavior.
- The only remaining literal colors are the gradient on `header h1`, which is hidden (`header { display: none; }`), and `transparent` values, which work in both themes.
