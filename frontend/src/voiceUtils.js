/**
 * Utility functions for voice recognition transcription formatting,
 * spoken punctuation parsing, capitalization, and text hygiene.
 */

const SPOKEN_PUNCTUATION_MAP = [
    { pattern: /\b(period|full stop)\b/gi, replacement: "." },
    { pattern: /\b(comma)\b/gi, replacement: "," },
    { pattern: /\b(question mark)\b/gi, replacement: "?" },
    { pattern: /\b(exclamation mark|exclamation point)\b/gi, replacement: "!" },
    { pattern: /\b(semicolon)\b/gi, replacement: ";" },
    { pattern: /\b(colon)\b/gi, replacement: ":" },
    { pattern: /\b(new paragraph)\b/gi, replacement: "\n\n" },
    { pattern: /\b(new line)\b/gi, replacement: "\n" },
];

/**
 * Format raw transcript:
 * 1. Translates spoken punctuation (e.g. "period" -> ".").
 * 2. Cleans up whitespace around punctuation.
 * 3. Auto-capitalizes sentences and pronouns.
 *
 * @param {string} rawText
 * @returns {string}
 */
export function formatTranscript(rawText) {
    if (!rawText) return "";

    let text = rawText;

    // 1. Replace spoken punctuation phrases
    for (const { pattern, replacement } of SPOKEN_PUNCTUATION_MAP) {
        text = text.replace(pattern, replacement);
    }

    // 2. Fix spaces before punctuation (e.g., "hello ." -> "hello.")
    text = text.replace(/\s+([.,!?:;])/g, "$1");

    // 3. Ensure a space after punctuation if followed immediately by a letter/digit
    text = text.replace(/([.,!?:;])([a-zA-Z0-9])/g, "$1 $2");

    // 4. Auto-capitalize the first character of the entire text
    text = text.replace(/^(\s*[a-z])/g, (match) => match.toUpperCase());

    // 5. Auto-capitalize letters following terminal punctuation (. ! ? \n)
    text = text.replace(/([.!?]\s+|\n+\s*)([a-z])/g, (_, boundary, letter) => {
        return boundary + letter.toUpperCase();
    });

    // 6. Capitalize standalone pronoun "I"
    text = text.replace(/\b(i)\b/g, "I");
    text = text.replace(/\b(i)('m|'ve|'ll|'d)\b/gi, (_, first, rest) => "I" + rest);

    return text;
}

/**
 * Ensures text ends with valid terminal punctuation (. ! ?),
 * keeping downstream NLP chunking and abstractive summarizers clean.
 *
 * @param {string} text
 * @returns {string}
 */
export function ensureTerminalPunctuation(text) {
    if (!text || !text.trim()) return "";
    const trimmed = text.trim();
    if (/[.!?]$/.test(trimmed)) {
        return trimmed;
    }
    return `${trimmed}.`;
}
