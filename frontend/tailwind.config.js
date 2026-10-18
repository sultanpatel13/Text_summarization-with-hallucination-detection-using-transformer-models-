/** @type {import('tailwindcss').Config} */
function withOpacity(variableName) {
    return ({ opacityValue }) => {
        if (opacityValue !== undefined) {
            return `rgb(var(${variableName}) / ${opacityValue})`;
        }
        return `rgb(var(${variableName}))`;
    };
}

export default {
    darkMode: "class",
    content: ["./src/**/*.{js,jsx,ts,tsx}"],
    theme: {
        extend: {
            fontFamily: {
                sans: ["Geist", "Inter", "system-ui", "-apple-system", "sans-serif"],
                mono: ["Geist Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "monospace"],
            },
            colors: {
                ink: withOpacity("--color-ink-rgb"),
                "ink-muted": withOpacity("--color-mute-rgb"),
                "on-primary": withOpacity("--color-on-primary-rgb"),
                body: withOpacity("--color-body-rgb"),
                mute: withOpacity("--color-mute-rgb"),
                hairline: withOpacity("--color-hairline-rgb"),
                "hairline-strong": withOpacity("--color-hairline-strong-rgb"),
                canvas: withOpacity("--color-canvas-rgb"),
                "canvas-soft": withOpacity("--color-canvas-soft-rgb"),
                "canvas-soft-2": withOpacity("--color-canvas-soft-2-rgb"),
                link: withOpacity("--color-link-rgb"),
                "link-deep": withOpacity("--color-link-deep-rgb"),
                "link-bg-soft": withOpacity("--color-link-bg-soft-rgb"),
                success: "#0070f3",
                error: "#ee0000",
                "error-soft": withOpacity("--color-error-soft-rgb"),
                "error-subtle": withOpacity("--color-error-soft-rgb"),
                "error-deep": "#c50000",
                warning: "#f5a623",
                "warning-soft": withOpacity("--color-warning-soft-rgb"),
                "warning-deep": "#ab570a",
                violet: "#7928ca",
                "violet-soft": withOpacity("--color-violet-soft-rgb"),
                "violet-deep": "#4c2889",
                cyan: "#50e3c2",
                "cyan-soft": withOpacity("--color-cyan-soft-rgb"),
                "cyan-deep": "#29bc9b",
                "highlight-pink": "#ff0080",
                "highlight-magenta": "#eb367f",
                "gradient-develop-start": "#007cf0",
                "gradient-develop-end": "#00dfd8",
                "gradient-preview-start": "#7928ca",
                "gradient-preview-end": "#ff0080",
                "gradient-ship-start": "#ff4d4d",
                "gradient-ship-end": "#f9cb28",
            },
            fontSize: {
                "display-xl": ["48px", { lineHeight: "48px", letterSpacing: "-2.4px", fontWeight: "600" }],
                "display-lg": ["32px", { lineHeight: "40px", letterSpacing: "-1.28px", fontWeight: "600" }],
                "display-md": ["24px", { lineHeight: "32px", letterSpacing: "-0.96px", fontWeight: "600" }],
                "display-sm": ["20px", { lineHeight: "28px", letterSpacing: "-0.6px", fontWeight: "600" }],
                "body-lg": ["18px", { lineHeight: "28px", fontWeight: "400" }],
                "body-md": ["16px", { lineHeight: "24px", fontWeight: "400" }],
                "body-sm": ["14px", { lineHeight: "20px", letterSpacing: "-0.28px", fontWeight: "400" }],
            },
            borderRadius: {
                xs: "4px",
                sm: "6px",
                md: "8px",
                lg: "12px",
                xl: "16px",
                "pill-sm": "64px",
                pill: "100px",
            },
            maxWidth: {
                page: "1400px",
            },
            spacing: {
                section: "192px",
            },
            boxShadow: {
                "elevation-1": "var(--elevation-1)",
                "elevation-2": "var(--elevation-2)",
                "elevation-3": "var(--elevation-3)",
                "elevation-4": "var(--elevation-4)",
                "elevation-5": "var(--elevation-5)",
            },
            animation: {
                blink: "blink 1s step-end infinite",
                "mesh-drift": "meshDrift 20s ease-in-out infinite alternate",
            },
            keyframes: {
                blink: {
                    "50%": { opacity: "0" },
                },
                meshDrift: {
                    "0%": { transform: "translate(-2%, -1%) scale(1.02)" },
                    "100%": { transform: "translate(2%, 1%) scale(1.05)" },
                },
            },
        },
    },
    plugins: [],
};
