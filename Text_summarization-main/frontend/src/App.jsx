import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
    AudioLines,
    Copy,
    Download,
    FileUp,
    History,
    Menu,
    Mic,
    Sparkles,
    Trash2,
    WandSparkles,
    X,
} from "lucide-react";

const SUMMARY_PRESETS = { short: 2, medium: 3, long: 5 };
const MODEL_OPTIONS = {
    extractive: "Extractive AI",
    pegasus: "PEGASUS",
    bart: "BART",
};
const DEFAULT_PLACEHOLDER = "Paste or type your text here...";
const DEFAULT_OUTPUT = "Your summary will appear here with highlighted key sentences and analytics.";
const HISTORY_KEY = "summarizer-history-v2";
const MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024;

const VOICE_CONFIG = {
    INITIAL_TIMEOUT: 5000,
    NORMAL_SILENCE: 2200,
    FINAL_RESULT_SILENCE: 1400,
    INTERIM_GRACE_PERIOD: 3000,
    MAX_LISTENING_TIME: 60000,
    RESTART_DELAY: 300,
    MAX_RESTARTS: 3,
};

const VOICE_STATES = {
    IDLE: "idle",
    STARTING: "starting",
    LISTENING: "listening",
    SPEAKING: "speaking",
    POSSIBLE_END: "possible_end",
    STOPPING: "stopping",
};

const DEFAULT_LANGUAGE = "en-US";

// Vite exposes client-side environment values through import.meta.env.
// Referencing Node's `process` object here crashes the browser before React
// gets a chance to mount.
const isDev = import.meta.env.DEV;
const logVoice = (...args) => {
    if (isDev) console.log("[Voice]", new Date().toISOString(), ...args);
};

const countWords = (text) => {
    const trimmed = text.trim();
    return trimmed ? trimmed.split(/\s+/).length : 0;
};

const splitSentences = (text) => text.split(/(?<=[.!?])\s+/).filter(Boolean);

function App() {
    const [text, setText] = useState("");
    const [draftTranscript, setDraftTranscript] = useState("");
    const [uploadedFile, setUploadedFile] = useState(null);
    const [summaryPreset, setSummaryPreset] = useState("medium");
    const [summaryModel, setSummaryModel] = useState("extractive");
    const [overview, setOverview] = useState("");
    const [summary, setSummary] = useState("");
    const [displayedSummary, setDisplayedSummary] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [analysis, setAnalysis] = useState(null);
    const [hallucinationDetection, setHallucinationDetection] = useState(null);
    const [historyOpen, setHistoryOpen] = useState(false);
    const [resultsOpen, setResultsOpen] = useState(false);
    const [historyItems, setHistoryItems] = useState([]);
    const [voiceState, setVoiceState] = useState(VOICE_STATES.IDLE);
    const [copied, setCopied] = useState(false);
    const [mobileNavOpen, setMobileNavOpen] = useState(false);
    const [isDragging, setIsDragging] = useState(false);

    const recognitionRef = useRef(null);
    const baseTranscriptRef = useRef("");
    const finalTranscriptRef = useRef("");
    const interimTranscriptRef = useRef("");
    const recognitionActiveRef = useRef(false);
    const isSpeakingRef = useRef(false);
    const hasReceivedSpeechRef = useRef(false);
    const lastSpeechTimeRef = useRef(0);
    const lastResultTimeRef = useRef(0);
    const lastFinalResultTimeRef = useRef(0);
    const silenceTimeoutRef = useRef(null);
    const maxDurationTimeoutRef = useRef(null);
    const recognitionSessionIdRef = useRef(0);
    const restartCountRef = useRef(0);
    const userStoppedRef = useRef(false);
    const processedResultIndexRef = useRef(-1);

    useEffect(() => {
        try {
            const savedHistory = window.localStorage.getItem(HISTORY_KEY);
            if (savedHistory) {
                const parsed = JSON.parse(savedHistory);
                setHistoryItems(Array.isArray(parsed) ? parsed : []);
            }
        } catch {
            window.localStorage.removeItem(HISTORY_KEY);
        }
    }, []);

    useEffect(() => {
        try {
            window.localStorage.setItem(HISTORY_KEY, JSON.stringify(historyItems));
        } catch {
            // History is optional; a quota or privacy setting must not break the app.
        }
    }, [historyItems]);

useEffect(() => () => {
        clearSilenceTimeout();
        if (maxDurationTimeoutRef.current) {
            window.clearTimeout(maxDurationTimeoutRef.current);
            maxDurationTimeoutRef.current = null;
        }
        if (recognitionRef.current) {
            try {
                recognitionRef.current.onend = null;
                recognitionRef.current.onerror = null;
                recognitionRef.current.abort?.();
            } catch {
                // ignore cleanup errors
            }
            recognitionRef.current = null;
        }
    }, []);

    useEffect(() => {
        if (!summary) {
            setDisplayedSummary("");
            return undefined;
        }
        let index = 0;
        const interval = window.setInterval(() => {
            index += 1;
            setDisplayedSummary(summary.slice(0, index));
            if (index >= summary.length) {
                window.clearInterval(interval);
            }
        }, 8);
        return () => window.clearInterval(interval);
    }, [summary]);

    const inputWordCount = useMemo(() => countWords(text), [text]);
    const outputSentences = useMemo(() => splitSentences(displayedSummary), [displayedSummary]);
    const keyTerms = useMemo(() => new Set(analysis?.top_keywords ?? []), [analysis]);
    const confidence = analysis?.topic_overlap_score ?? 0;
    const aiSignal = analysis?.ai_writing_signals?.score ?? 0;

const handleUpload = (file) => {
        if (file && file.size > MAX_UPLOAD_SIZE_BYTES) {
            setUploadedFile(null);
            setError("Choose a file smaller than 5 MB.");
            return;
        }
        setUploadedFile(file || null);
        setError("");
    };

    const handleClear = () => {
        setText("");
        setDraftTranscript("");
        baseTranscriptRef.current = "";
        finalTranscriptRef.current = "";
        interimTranscriptRef.current = "";
        setUploadedFile(null);
        setIsDragging(false);
        setOverview("");
        setSummary("");
        setDisplayedSummary("");
        setAnalysis(null);
        setHallucinationDetection(null);
        setError("");
        setCopied(false);
        setResultsOpen(false);
        clearSilenceTimeout();
        clearMaxDurationTimeout();
        if (recognitionActiveRef.current) {
            userStoppedRef.current = true;
            recognitionActiveRef.current = false;
            try {
                recognitionRef.current?.stop();
            } catch {
                recognitionRef.current?.abort?.();
            }
        }
        setVoiceState(VOICE_STATES.IDLE);
    };

const clearSilenceTimeout = () => {
        if (silenceTimeoutRef.current) {
            window.clearTimeout(silenceTimeoutRef.current);
            silenceTimeoutRef.current = null;
        }
    };

    const clearMaxDurationTimeout = () => {
        if (maxDurationTimeoutRef.current) {
            window.clearTimeout(maxDurationTimeoutRef.current);
            maxDurationTimeoutRef.current = null;
        }
    };

    const setVoiceStateSafe = (newState) => {
        logVoice("state:", voiceState, "->", newState);
        setVoiceState(newState);
    };

    const scheduleSilenceTimeout = (timeoutMs) => {
        clearSilenceTimeout();
        silenceTimeoutRef.current = window.setTimeout(() => {
            logVoice("silence timeout fired, current state:", voiceState);
            if (recognitionActiveRef.current && voiceState !== VOICE_STATES.STOPPING) {
                handlePossibleEnd();
            }
        }, timeoutMs);
    };

    const startMaxDurationTimer = () => {
        clearMaxDurationTimeout();
        maxDurationTimeoutRef.current = window.setTimeout(() => {
            logVoice("max duration reached");
            setError("Maximum recording duration reached (60 seconds).");
            stopVoiceInput();
        }, VOICE_CONFIG.MAX_LISTENING_TIME);
    };

    const handlePossibleEnd = () => {
        if (!recognitionActiveRef.current || voiceState === VOICE_STATES.STOPPING) {
            return;
        }
        const now = Date.now();
        const timeSinceLastSpeech = now - lastSpeechTimeRef.current;
        const timeSinceLastResult = now - lastResultTimeRef.current;
        const timeSinceLastFinalResult = now - lastFinalResultTimeRef.current;
        const hasInterim = interimTranscriptRef.current.trim().length > 0;
        const hasFinal = finalTranscriptRef.current.trim().length > 0;

        let silenceTimeout;
        if (!hasReceivedSpeechRef.current) {
            silenceTimeout = VOICE_CONFIG.INITIAL_TIMEOUT;
            logVoice("no speech received yet, using INITIAL_TIMEOUT");
        } else if (hasInterim && timeSinceLastFinalResult < VOICE_CONFIG.INTERIM_GRACE_PERIOD) {
            silenceTimeout = VOICE_CONFIG.INTERIM_GRACE_PERIOD;
            logVoice("interim exists, using INTERIM_GRACE_PERIOD");
        } else if (hasFinal && timeSinceLastFinalResult < VOICE_CONFIG.FINAL_RESULT_SILENCE) {
            silenceTimeout = VOICE_CONFIG.FINAL_RESULT_SILENCE;
            logVoice("recent final result, using FINAL_RESULT_SILENCE");
        } else {
            silenceTimeout = VOICE_CONFIG.NORMAL_SILENCE;
            logVoice("using NORMAL_SILENCE");
        }

        logVoice("possible_end, scheduling stop in", silenceTimeout, "ms");
        setVoiceStateSafe(VOICE_STATES.POSSIBLE_END);
        scheduleSilenceTimeout(silenceTimeout);
    };

    const finalizeTranscript = () => {
        const interim = interimTranscriptRef.current.trim();
        if (interim && !finalTranscriptRef.current.endsWith(interim)) {
            const words = interim.split(/\s+/);
            const lastFewWords = words.slice(-3).join(" ");
            if (finalTranscriptRef.current.includes(lastFewWords)) {
                logVoice("interim appears to be duplicate of final, skipping");
            } else {
                finalTranscriptRef.current = (finalTranscriptRef.current + " " + interim).trim();
                logVoice("appended interim to final transcript");
            }
        }
        const complete = finalTranscriptRef.current.trim();
        baseTranscriptRef.current = complete;
        setText(complete);
        logVoice("finalized transcript length:", complete.length);
    };

    const stopVoiceInput = () => {
        logVoice("stopVoiceInput called, state:", voiceState);
        if (!recognitionActiveRef.current && voiceState === VOICE_STATES.IDLE) {
            return;
        }
        userStoppedRef.current = true;
        recognitionActiveRef.current = false;
        clearSilenceTimeout();
        clearMaxDurationTimeout();

        if (recognitionRef.current) {
            setVoiceStateSafe(VOICE_STATES.STOPPING);
            try {
                recognitionRef.current.stop();
            } catch {
                recognitionRef.current?.abort?.();
            }
        } else {
            finalizeTranscript();
            setVoiceStateSafe(VOICE_STATES.IDLE);
            setDraftTranscript("");
            interimTranscriptRef.current = "";
        }
    };

    const processResults = (event) => {
        const results = event.results;
        const resultIndex = event.resultIndex;
        let newFinal = "";
        let newInterim = "";

        for (let i = resultIndex; i < results.length; i += 1) {
            const result = results[i];
            if (!result || result.length === 0) continue;
            const transcript = result[0]?.transcript?.trim() || "";
            if (!transcript) continue;

            if (result.isFinal) {
                newFinal += transcript + " ";
                lastFinalResultTimeRef.current = Date.now();
            } else {
                newInterim += transcript + " ";
            }
        }

        lastResultTimeRef.current = Date.now();

        if (newFinal) {
            const cleanedFinal = newFinal.trim();
            if (!finalTranscriptRef.current.endsWith(cleanedFinal)) {
                finalTranscriptRef.current = (finalTranscriptRef.current + " " + cleanedFinal).trim();
                logVoice("added final result:", cleanedFinal);
            } else {
                logVoice("skipped duplicate final result");
            }
        }

        if (newInterim) {
            interimTranscriptRef.current = newInterim.trim();
        } else if (newFinal) {
            interimTranscriptRef.current = "";
        }

        const displayText = (finalTranscriptRef.current + " " + interimTranscriptRef.current).trim();
        setDraftTranscript(displayText);
    };

    const startVoiceInput = () => {
        logVoice("startVoiceInput called, current state:", voiceState);

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setError("Voice input is not supported in this browser. Try Chrome or Edge.");
            return;
        }

        if (recognitionActiveRef.current) {
            logVoice("already active, ignoring");
            return;
        }

        if (recognitionRef.current) {
            try {
                recognitionRef.current.onend = null;
                recognitionRef.current.onerror = null;
                recognitionRef.current.abort?.();
            } catch {
                // ignore
            }
        }

        recognitionSessionIdRef.current += 1;
        const currentSessionId = recognitionSessionIdRef.current;
        restartCountRef.current = 0;
        userStoppedRef.current = false;
        processedResultIndexRef.current = -1;

        const recognition = new SpeechRecognition();
        recognition.lang = DEFAULT_LANGUAGE;
        recognition.interimResults = true;
        recognition.continuous = true;
        recognition.maxAlternatives = 1;

        recognition.onstart = () => {
            if (recognitionSessionIdRef.current !== currentSessionId) {
                logVoice("stale session, ignoring onstart");
                return;
            }
            logVoice("onstart");
            recognitionActiveRef.current = true;
            hasReceivedSpeechRef.current = false;
            isSpeakingRef.current = false;
            finalTranscriptRef.current = "";
            interimTranscriptRef.current = "";
            baseTranscriptRef.current = text.trim();
            lastSpeechTimeRef.current = Date.now();
            lastResultTimeRef.current = Date.now();
            lastFinalResultTimeRef.current = 0;
            setVoiceStateSafe(VOICE_STATES.LISTENING);
            setError("");
            scheduleSilenceTimeout(VOICE_CONFIG.INITIAL_TIMEOUT);
            startMaxDurationTimer();
        };

        recognition.onresult = (event) => {
            if (recognitionSessionIdRef.current !== currentSessionId) {
                logVoice("stale session, ignoring onresult");
                return;
            }
            logVoice("onresult, resultIndex:", event.resultIndex, "results:", event.results.length);
            processResults(event);
            hasReceivedSpeechRef.current = true;

            if (voiceState === VOICE_STATES.LISTENING || voiceState === VOICE_STATES.POSSIBLE_END) {
                setVoiceStateSafe(VOICE_STATES.SPEAKING);
            }
            clearSilenceTimeout();
            scheduleSilenceTimeout(VOICE_CONFIG.NORMAL_SILENCE);
        };

        recognition.onspeechstart = () => {
            if (recognitionSessionIdRef.current !== currentSessionId) {
                return;
            }
            logVoice("onspeechstart");
            isSpeakingRef.current = true;
            hasReceivedSpeechRef.current = true;
            lastSpeechTimeRef.current = Date.now();
            clearSilenceTimeout();
            if (voiceState === VOICE_STATES.LISTENING || voiceState === VOICE_STATES.POSSIBLE_END) {
                setVoiceStateSafe(VOICE_STATES.SPEAKING);
            }
        };

        recognition.onspeechend = () => {
            if (recognitionSessionIdRef.current !== currentSessionId) {
                return;
            }
            logVoice("onspeechend");
            isSpeakingRef.current = false;
            lastSpeechTimeRef.current = Date.now();
            handlePossibleEnd();
        };

        recognition.onerror = (event) => {
            if (recognitionSessionIdRef.current !== currentSessionId) {
                return;
            }
            logVoice("onerror:", event.error);
            const errorMap = {
                "not-allowed": "Microphone permission was denied. Please allow microphone access and try again.",
                "service-not-allowed": "Speech recognition service is not allowed in this browser.",
                network: "Speech recognition encountered a network problem. Please check your connection.",
                "no-speech": "No speech was detected. Try speaking again.",
                "audio-capture": "No microphone was found. Please check your audio input device.",
                aborted: "Voice input was stopped.",
            };
            const msg = errorMap[event.error] || "Voice capture ran into an issue. Please try again.";
            setError(msg);
            recognitionActiveRef.current = false;
            clearSilenceTimeout();
            clearMaxDurationTimeout();
            finalizeTranscript();
            setVoiceStateSafe(VOICE_STATES.IDLE);
            setDraftTranscript("");
            interimTranscriptRef.current = "";
        };

        recognition.onend = () => {
            if (recognitionSessionIdRef.current !== currentSessionId) {
                logVoice("stale session, ignoring onend");
                return;
            }
            logVoice("onend, active:", recognitionActiveRef.current, "userStopped:", userStoppedRef.current, "state:", voiceState);

            clearSilenceTimeout();
            clearMaxDurationTimeout();

            if (userStoppedRef.current) {
                logVoice("user stopped, finalizing");
                finalizeTranscript();
                recognitionActiveRef.current = false;
                setVoiceStateSafe(VOICE_STATES.IDLE);
                setDraftTranscript("");
                interimTranscriptRef.current = "";
                return;
            }

            if (!recognitionActiveRef.current) {
                logVoice("already inactive, finalizing");
                finalizeTranscript();
                setVoiceStateSafe(VOICE_STATES.IDLE);
                setDraftTranscript("");
                interimTranscriptRef.current = "";
                return;
            }

            const shouldRestart = isSpeakingRef.current || hasReceivedSpeechRef.current;
            if (shouldRestart && restartCountRef.current < VOICE_CONFIG.MAX_RESTARTS) {
                restartCountRef.current += 1;
                logVoice("restarting recognition, attempt:", restartCountRef.current);
                setTimeout(() => {
                    if (recognitionSessionIdRef.current === currentSessionId && recognitionActiveRef.current) {
                        try {
                            recognitionRef.current?.start();
                        } catch (e) {
                            logVoice("restart failed:", e);
                            finalizeTranscript();
                            recognitionActiveRef.current = false;
                            setVoiceStateSafe(VOICE_STATES.IDLE);
                            setDraftTranscript("");
                            interimTranscriptRef.current = "";
                        }
                    }
                }, VOICE_CONFIG.RESTART_DELAY);
                return;
            }

            logVoice("ending session normally");
            finalizeTranscript();
            recognitionActiveRef.current = false;
            setVoiceStateSafe(VOICE_STATES.IDLE);
            setDraftTranscript("");
            interimTranscriptRef.current = "";
        };

        recognitionRef.current = recognition;
        setVoiceStateSafe(VOICE_STATES.STARTING);
        logVoice("starting recognition...");
        try {
            recognition.start();
        } catch (e) {
            logVoice("start failed:", e);
            recognitionActiveRef.current = false;
            setVoiceStateSafe(VOICE_STATES.IDLE);
            setError("Voice input could not start. If the microphone is already active, stop it and try again.");
            clearSilenceTimeout();
        }
    };

    const handleVoiceToggle = () => {
        if (voiceState === VOICE_STATES.IDLE || voiceState === VOICE_STATES.STARTING) {
            startVoiceInput();
        } else {
            stopVoiceInput();
        }
    };

    const handleCopy = async () => {
        if (!summary) return;
        try {
            await navigator.clipboard.writeText(summary);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
        } catch {
            setError("Copying is unavailable in this browser. Select the summary and copy it manually.");
        }
    };

    const handleDownloadPdf = async () => {
        if (!summary) return;
        const { jsPDF } = await import("jspdf");
        const pdf = new jsPDF({ orientation: "p", unit: "pt", format: "a4" });
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(18);
        pdf.text("AI Summary", 40, 50);
        pdf.setFont("helvetica", "normal");
        pdf.setFontSize(11);
        pdf.text(pdf.splitTextToSize(summary, 520), 40, 80);
        pdf.save("summary.pdf");
    };

    const loadHistoryItem = (item) => {
        setOverview(item.overview || "");
        setSummary(item.summary);
        setDisplayedSummary(item.summary);
        setAnalysis(item.analysis);
        setHallucinationDetection(null);
        setSummaryPreset(item.preset);
        setSummaryModel(item.model || "extractive");
        setUploadedFile(null);
        setText(item.sourceText || item.inputPreview || "");
        setHistoryOpen(false);
        setResultsOpen(true);
    };

    const handleSummarize = async () => {
        if (!text.trim() && !uploadedFile) {
            setError("Add some text or upload a file before summarizing.");
            return;
        }
        setLoading(true);
        setError("");
        setCopied(false);
        try {
            const formData = new FormData();
            formData.append("text", text);
            formData.append("summary_length", String(SUMMARY_PRESETS[summaryPreset]));
            formData.append("model_type", summaryModel);
            if (uploadedFile) {
                formData.append("file", uploadedFile);
            }
            const response = await fetch("/summarize", { method: "POST", body: formData });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Unable to summarize this content.");
            }
            setOverview(data.overview || "");
            setSummary(data.summary || "");
            setAnalysis(data.analysis || null);
            setHallucinationDetection(data.hallucination_detection || null);
            setResultsOpen(true);
            persistHistory({
                sourceText: uploadedFile ? `${uploadedFile.name} upload` : text,
                overview: data.overview || "",
                summary: data.summary || "",
                analysis: data.analysis || null,
                sourceFile: data.source_file || null,
                preset: summaryPreset,
                model: summaryModel,
            });
        } catch (requestError) {
            setError(requestError.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="relative min-h-screen bg-canvas-soft text-ink">
            <TopNav
                setHistoryOpen={setHistoryOpen}
                mobileNavOpen={mobileNavOpen}
                setMobileNavOpen={setMobileNavOpen}
            />

            <HeroBand onSummarize={handleSummarize} loading={loading} />

            <main className="relative z-10 mx-auto max-w-page px-4 pb-24 pt-8 sm:px-6 lg:px-8">
<InputPanel
                    draftTranscript={draftTranscript}
                    text={text}
                    setText={setText}
                    inputWordCount={inputWordCount}
                    inputCharacterCount={text.length}
                    uploadedFile={uploadedFile}
                    handleUpload={handleUpload}
                    summaryPreset={summaryPreset}
                    setSummaryPreset={setSummaryPreset}
                    summaryModel={summaryModel}
                    setSummaryModel={setSummaryModel}
                    handleClear={handleClear}
                    loading={loading}
                    voiceState={voiceState}
                    handleVoiceToggle={handleVoiceToggle}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    handleSummarize={handleSummarize}
                />

                <FeatureGrid />
            </main>

            <Footer />

            <HistoryDrawer
                historyOpen={historyOpen}
                setHistoryOpen={setHistoryOpen}
                historyItems={historyItems}
                loadHistoryItem={loadHistoryItem}
                clearHistory={() => setHistoryItems([])}
            />

            <ResultsWindow
                resultsOpen={resultsOpen}
                setResultsOpen={setResultsOpen}
                loading={loading}
                analysis={analysis}
                summary={summary}
                hallucinationDetection={hallucinationDetection}
                confidence={confidence}
                aiSignal={aiSignal}
                overview={overview}
                displayedSummary={displayedSummary}
                outputSentences={outputSentences}
                keyTerms={keyTerms}
                copied={copied}
                handleCopy={handleCopy}
                handleDownloadPdf={handleDownloadPdf}
                uploadedFile={uploadedFile}
            />

            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: 18 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 18 }}
                        className="fixed bottom-6 left-6 z-50 max-w-md toast-error"
                    >
                        {error}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function TopNav({ setHistoryOpen, mobileNavOpen, setMobileNavOpen }) {
    return (
        <nav className="nav-bar">
            <div className="mx-auto flex w-full max-w-page items-center justify-between px-4 sm:px-6 lg:px-8">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <div className="flex h-6 w-6 items-center justify-center rounded-sm bg-ink">
                            <Sparkles size={14} className="text-on-primary" />
                        </div>
                        <span className="text-body-sm font-medium text-ink">Summarize</span>
                    </div>
                    <div className="hidden items-center gap-1 md:flex">
                        <button type="button" onClick={() => document.getElementById("features")?.scrollIntoView({ behavior: "smooth" })} className="btn-ghost">
                            Features
                        </button>
                        <button type="button" onClick={() => setHistoryOpen(true)} className="btn-ghost">
                            History
                        </button>
                        <a href="#documentation" className="btn-ghost">
                            Docs
                        </a>
                    </div>
                </div>

                <div className="hidden items-center gap-2 sm:flex">
                    <button type="button" onClick={() => setHistoryOpen(true)} className="btn-secondary-sm">
                        History
                    </button>
                    <button type="button" onClick={() => document.getElementById("input")?.scrollIntoView({ behavior: "smooth" })} className="btn-primary-sm">
                        Open App
                    </button>
                </div>

                <button
                    type="button"
                    onClick={() => setMobileNavOpen(!mobileNavOpen)}
                    className="icon-btn sm:hidden"
                    aria-label="Toggle menu"
                >
                    {mobileNavOpen ? <X size={18} /> : <Menu size={18} />}
                </button>
            </div>

            <AnimatePresence>
                {mobileNavOpen && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="absolute left-0 right-0 top-16 overflow-y-auto border-b border-hairline bg-canvas sm:hidden"
                    >
                        <div className="flex flex-col gap-1 px-4 py-4">
                            <button type="button" className="btn-ghost justify-start">
                                Features
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setHistoryOpen(true);
                                    setMobileNavOpen(false);
                                }}
                                className="btn-ghost justify-start"
                            >
                                History
                            </button>
                            <button type="button" className="btn-ghost justify-start">
                                Docs
                            </button>
                            <div className="mt-2 flex gap-2">
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </nav>
    );
}

function HeroBand({ onSummarize, loading }) {
    return (
        <section className="relative overflow-y-auto bg-canvas">
            <div className="pointer-events-none absolute inset-0 overflow-y-auto">
                <div className="mesh-gradient absolute -left-1/4 -top-1/4 h-[600px] w-[800px] animate-mesh-drift" />
                <div className="mesh-gradient absolute -right-1/4 top-0 h-[500px] w-[700px] animate-mesh-drift [animation-delay:3s]" />
            </div>

            <div className="relative z-10 mx-auto max-w-page px-4 py-16 sm:px-6 sm:py-24 lg:px-8 lg:py-32">
                <div className="mx-auto max-w-3xl text-center">
                    <div className="mb-6 flex justify-center">
                        <span className="banner-marketing">
                            <span className="caption-mono mr-2 text-mute">New</span>
                            PEGASUS model now available
                        </span>
                    </div>
                    <h1 className="text-display-xl text-ink sm:text-[48px]">
                        Summarize any text in seconds.
                    </h1>
                    <p className="mx-auto mt-6 max-w-xl text-body-lg text-body">
                        Paste, upload, or speak your content. Get concise summaries with keyword highlights,
                        confidence scores, and AI-writing analysis.
                    </p>
                    <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                        <button type="button" onClick={onSummarize} disabled={loading} className="btn-primary">
                            <Sparkles size={16} />
                            {loading ? "Summarizing..." : "Start Summarizing"}
                        </button>
                        <a href="#documentation" className="btn-secondary">
                            View Documentation
                        </a>
                    </div>
                </div>
            </div>
        </section>
    );
}

function InputPanel(props) {
    const {
        draftTranscript,
        text,
        setText,
        inputWordCount,
        inputCharacterCount,
        uploadedFile,
        handleUpload,
        summaryPreset,
        setSummaryPreset,
        summaryModel,
        setSummaryModel,
        handleClear,
        loading,
        voiceState,
        handleVoiceToggle,
        isDragging,
        setIsDragging,
        handleSummarize,
    } = props;

    const isActive = voiceState !== VOICE_STATES.IDLE && voiceState !== VOICE_STATES.STARTING;
    const isSpeaking = voiceState === VOICE_STATES.SPEAKING;
    const isPossibleEnd = voiceState === VOICE_STATES.POSSIBLE_END;
    const isStopping = voiceState === VOICE_STATES.STOPPING;
    const isStarting = voiceState === VOICE_STATES.STARTING;

    const getVoiceButtonText = () => {
        if (isStarting) return "Starting...";
        if (isSpeaking) return "Listening — speech detected";
        if (isPossibleEnd) return "Finishing...";
        if (isStopping) return "Processing...";
        if (isActive) return "Listening...";
        return "Voice Input";
    };

    const getVoiceButtonClass = () => {
        if (isStopping) return "btn-outline border-amber bg-amber-soft text-amber-deep";
        if (isPossibleEnd) return "btn-outline border-blue bg-blue-soft text-blue-deep";
        if (isSpeaking) return "btn-outline border-green bg-green-soft text-green-deep animate-pulse-subtle";
        if (isActive) return "btn-outline border-violet bg-violet-soft text-violet-deep";
        return "btn-outline";
    };

    return (
            <motion.section id="input"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="card-marketing-lg mb-16"
        >
            <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
                <div>
                    <p className="caption-mono mb-2">Input Engine</p>
                    <h2 className="text-display-md text-ink">Context ingestion.</h2>
                    <p className="mt-2 text-body-sm text-body">
                        Upload a document or paste text to begin summarization.
                    </p>
                </div>
                <div className="flex gap-2">
                    <span className="badge-secondary">{inputWordCount} words</span>
                    <span className="badge-secondary">{inputCharacterCount} chars</span>
                </div>
            </div>

            <motion.div
                onDragOver={(event) => {
                    event.preventDefault();
                    setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(event) => {
                    event.preventDefault();
                    setIsDragging(false);
                    const [file] = event.dataTransfer.files;
                    if (file) handleUpload(file);
                }}
                className={`mb-4 rounded-md border border-dashed p-4 transition ${
                    isDragging
                        ? "border-link bg-link-bg-soft/30"
                        : "border-hairline bg-canvas-soft"
                }`}
            >
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-md bg-canvas-soft-2 text-ink">
                            <FileUp size={18} />
                        </div>
                        <div>
                            <p className="text-body-sm font-medium text-ink">Drag, drop, or upload your source</p>
                            <p className="text-xs text-mute">
                                TXT, PDF, DOCX, PPTX, HTML, CSV, JSON, code, logs, and more.
                            </p>
                        </div>
                    </div>
                    <label className="btn-outline cursor-pointer">
                        Choose File
                        <input type="file" className="hidden" onChange={(event) => handleUpload(event.target.files?.[0] ?? null)} />
                    </label>
                </div>
                <p className="mt-3 text-xs text-mute">
                    {uploadedFile ? `Loaded file: ${uploadedFile.name}` : "No file selected. Manual text still works perfectly."}
                </p>
            </motion.div>

            <div className="relative mb-6">
                <textarea
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    placeholder={DEFAULT_PLACEHOLDER}
                    className="form-textarea min-h-[320px]"
                />
                {draftTranscript && (
                    <div className="pointer-events-none absolute bottom-3 left-3 right-3 rounded-sm border border-hairline bg-canvas-soft px-3 py-2 text-body-sm italic text-body">
                        {draftTranscript}
                    </div>
                )}
            </div>

            <div className="mb-6 flex flex-col gap-4">
                <div>
                    <p className="caption-mono mb-2">Summary Length</p>
                    <div className="flex flex-wrap gap-2">
                        {Object.keys(SUMMARY_PRESETS).map((preset) => (
                            <button
                                key={preset}
                                type="button"
                                onClick={() => setSummaryPreset(preset)}
                                className={summaryPreset === preset ? "tab-ghost-active" : "tab-ghost"}
                            >
                                {preset.charAt(0).toUpperCase() + preset.slice(1)}
                            </button>
                        ))}
                    </div>
                </div>
                <div>
                    <p className="caption-mono mb-2">Model</p>
                    <div className="flex flex-wrap gap-2">
                        {Object.entries(MODEL_OPTIONS).map(([modelKey, label]) => (
                            <button
                                key={modelKey}
                                type="button"
                                onClick={() => setSummaryModel(modelKey)}
                                className={summaryModel === modelKey ? "tab-ghost-active" : "tab-ghost"}
                            >
                                {label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 border-t border-hairline pt-6">
                <div className="flex flex-wrap items-center gap-2">
                    <button
                        type="button"
                        onClick={handleVoiceToggle}
                        disabled={isStarting || isStopping}
                        className={getVoiceButtonClass()}
                        aria-label={isActive ? "Stop voice input" : "Start voice input"}
                    >
                        <Mic size={14} />
                        {getVoiceButtonText()}
                    </button>
                    <button type="button" onClick={handleClear} className="btn-outline">
                        <Trash2 size={14} />
                        Clear
                    </button>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                    <span className="text-body-sm text-mute">
                        {loading ? "Preparing summary..." : "Results open in a focused window"}
                    </span>
                    <button type="button" onClick={handleSummarize} disabled={loading} className="btn-primary">
                        <Sparkles size={16} />
                        {loading ? "Summarizing..." : "Summarize"}
                    </button>
                </div>
            </div>
        </motion.section>
    );
}

function FeatureGrid() {
    const features = [
        {
            title: "Extractive AI.",
            body: "TextRank-style sentence ranking picks the most important ideas from your source text.",
            mono: "extractive",
        },
        {
            title: "PEGASUS model.",
            body: "Abstractive summarization for fluent, human-like output on longer documents.",
            mono: "pegasus",
        },
        {
            title: "Rich analytics.",
            body: "Confidence scores, keyword highlights, reading time, and AI-writing signal detection.",
            mono: "analytics",
        },
    ];

    return (
        <section id="features" className="mb-16">
            <div className="mb-8 text-center">
                <p className="caption-mono mb-2">Capabilities</p>
                <h2 className="text-display-lg text-ink">A compute model for all workloads.</h2>
                <p className="mx-auto mt-4 max-w-xl text-body-md text-body">
                    Three summarization engines, one interface. Choose the model that fits your content.
                </p>
            </div>
            <div className="grid gap-6 md:grid-cols-3">
                {features.map((feature) => (
                    <div key={feature.mono} className="card-marketing">
                        <p className="caption-mono mb-3 text-mute">{feature.mono}</p>
                        <h3 className="text-display-sm text-ink">{feature.title}</h3>
                        <p className="mt-2 text-body-sm text-body">{feature.body}</p>
                    </div>
                ))}
            </div>
        </section>
    );
}

function ResultsWindow(props) {
    const {
        resultsOpen,
        setResultsOpen,
        loading,
        analysis,
        hallucinationDetection,
        confidence,
        aiSignal,
        overview,
        displayedSummary,
        summary,
        outputSentences,
        keyTerms,
        copied,
        handleCopy,
        handleDownloadPdf,
        uploadedFile,
    } = props;

    return (
        <AnimatePresence>
            {resultsOpen && (
                <>
                    <motion.button
                        type="button"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setResultsOpen(false)}
                        className="fixed inset-0 z-40 bg-ink/20 backdrop-blur-sm"
                    />
                    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5">
                        <motion.section
                            initial={{ opacity: 0, scale: 0.96, y: 24 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.96, y: 24 }}
                            transition={{ duration: 0.22, ease: "easeOut" }}
                            className="flex max-h-[92vh] min-h-0 w-[min(1240px,96vw)] flex-col overflow-y-auto rounded-lg bg-canvas p-4 shadow-elevation-5 sm:p-6"
                        >
                            <div className="mb-5 flex items-start justify-between gap-4">
                                <div>
                                    <p className="caption-mono mb-2">Interactive Results</p>
                                    <h3 className="text-display-md text-ink">Summary window.</h3>
                                    <p className="mt-1 text-body-sm text-mute">
                                        {uploadedFile?.name || "Manual text input"}
                                    </p>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button type="button" onClick={handleCopy} className="icon-btn" aria-label="Copy">
                                        <Copy size={16} />
                                    </button>
                                    <button type="button" onClick={handleDownloadPdf} className="icon-btn" aria-label="Download PDF">
                                        <Download size={16} />
                                    </button>
                                    <button type="button" onClick={() => setResultsOpen(false)} className="icon-btn" aria-label="Close">
                                        <X size={16} />
                                    </button>
                                </div>
                            </div>

                            <div className="mb-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                                <MetricCard label="Topic Overlap" value={`${confidence}%`} />
                                <MetricCard label="Writing Pattern" value={`${aiSignal}%`} />
                                <MetricCard label="Summary Words" value={`${analysis?.summary_word_count ?? 0}`} />
                                <MetricCard label="Reading Time" value={`${analysis?.summary_reading_time_minutes ?? 0} min`} />
                            </div>

                            <div className="grid min-h-0 flex-1 gap-5 overflow-y-auto lg:grid-cols-[0.9fr_1.1fr]">
                                <div className="scrollbar-thin flex min-h-0 flex-col gap-4 overflow-y-auto pr-1">
                                    <div className="rounded-md border border-hairline bg-link-bg-soft/40 px-5 py-4">
                                        <p className="caption-mono mb-2 text-link">What it&apos;s about</p>
                                        <p className="text-body-sm leading-6 text-ink">
                                            {overview || "A clear one-line explanation of the text will appear here."}
                                        </p>
                                    </div>

                                    <div className="rounded-md border border-hairline bg-canvas-soft p-5">
                                        <p className="text-display-sm mb-3 text-ink">Summary insights.</p>
                                        <div className="grid gap-3 sm:grid-cols-2">
                                            <InsightCard
                                                icon={<AudioLines size={16} />}
                                                title="Source Stats"
                                                body={
                                                    analysis
                                                        ? `${analysis.input_word_count} words, ${analysis.input_sentence_count} sentences, ${analysis.input_character_count} characters`
                                                        : "No analytics yet."
                                                }
                                            />
                                            <InsightCard
                                                icon={<History size={16} />}
                                                title="Source File"
                                                body={uploadedFile?.name || "Manual text input"}
                                            />
                                            <InsightCard
                                                icon={<WandSparkles size={16} />}
                                                title="Writing Pattern (heuristic)"
                                                body={analysis?.ai_writing_signals?.details || "Heuristic guidance appears here after processing."}
                                            />
                                            <InsightCard
                                                icon={<Sparkles size={16} />}
                                                title="Top Keywords"
                                                body={analysis?.top_keywords?.join(", ") || "Keywords will surface after summarization."}
                                            />
                                        </div>
                                    </div>
                                </div>                                    <div className="mb-5 rounded-md border border-hairline bg-canvas-soft p-5">
  <div className="mb-3">
    <p className="caption-mono mb-1 text-link">GENERATED OUTPUT</p>
    <h3 className="text-display-sm text-ink">Generated Summary</h3>
  </div>

  <div className="rounded-md border border-hairline bg-canvas p-4">
    <p className="text-body-md leading-7 text-ink whitespace-pre-wrap">
      {summary}
    </p>
  </div>
</div>
                                    {hallucinationDetection && !hallucinationDetection.error && (
                                        <div className="mb-5 rounded-md border border-hairline bg-canvas-soft p-5">
                                            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <p className="caption-mono mb-1">AI Verification</p>
                                                    <h3 className="text-display-sm text-ink">
                                                        Source-consistency check
                                                    </h3>
                                                </div>

                                                <div className="flex flex-wrap gap-2">
                                                    <span className="badge-secondary">
                                                        {hallucinationDetection.total_sentences} sentences
                                                    </span>
                                                    <span className="badge-secondary">
                                                        {hallucinationDetection.hallucinated_sentences} hallucinated
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="mb-4 rounded-md border border-hairline bg-canvas p-4">
                                                <div className="flex items-center justify-between">
                                                    <span className="text-body-sm font-medium text-ink">
                                                        Overall Hallucination Score
                                                    </span>
                                                    <span className="text-display-sm text-ink">
                                                        {hallucinationDetection.hallucination_score}%
                                                    </span>
                                                </div>

                                                <div className="mt-3 h-2 overflow-y-auto rounded-full bg-canvas-soft-2">
                                                    <div
                                                        className="h-full rounded-full bg-ink transition-all"
                                                        style={{
                                                            width: `${Math.min(
                                                                hallucinationDetection.hallucination_score,
                                                                100
                                                            )}%`
                                                        }}
                                                    />
                                                </div>
                                            </div>

                                            <div className="space-y-3">
                                                {hallucinationDetection.results?.map((item, index) => {
                                                    const isHallucinated =
                                                        item.label === "Hallucinated";

                                                    return (
                                                        <div
                                                            key={`${item.sentence}-${index}`}
                                                            className="rounded-md border border-hairline bg-canvas p-4"
                                                        >
                                                            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                                                                <span className="caption-mono text-mute">
                                                                    Sentence {index + 1}
                                                                </span>

                                                                <span
                                                                    className={
                                                                        isHallucinated
                                                                            ? "badge-secondary text-red-600"
                                                                            : "badge-secondary"
                                                                    }
                                                                >
                                                                    {isHallucinated
                                                                        ? "HALLUCINATED"
                                                                        : "FACTUAL"}
                                                                </span>
                                                            </div>

                                                            <p className="text-body-sm leading-6 text-ink">
                                                                {item.sentence}
                                                            </p>

                                                            <div className="mt-3 grid gap-3 sm:grid-cols-2">
                                                                <div>
                                                                    <p className="caption-mono mb-1 text-mute">
                                                                        Confidence
                                                                    </p>
                                                                    <p className="text-body-sm font-medium text-ink">
                                                                        {item.probability}%
                                                                    </p>
                                                                </div>

                                                                <div>
                                                                    <p className="caption-mono mb-1 text-mute">
                                                                        Evidence Similarity
                                                                    </p>
                                                                    <p className="text-body-sm font-medium text-ink">
                                                                        {item.evidence_similarity}
                                                                    </p>
                                                                </div>
                                                            </div>

                                                            {item.evidence && (
                                                                <div className="mt-3 rounded-md bg-link-bg-soft/40 p-3">
                                                                    <p className="caption-mono mb-1 text-link">
                                                                        Source Evidence
                                                                    </p>
                                                                    <p className="text-xs leading-5 text-ink">
                                                                        "{item.evidence}"
                                                                    </p>
                                                                </div>
                                                            )}
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}


                                <div className="min-h-0 overflow-y-auto rounded-md border border-hairline bg-canvas p-4 sm:p-5">
                                    <div className="mb-4 flex items-center justify-between gap-4">
                                        <div className="text-body-sm text-mute">
                                            {analysis ? `${analysis.summary_word_count} words in summary` : "Ready for generation"}
                                        </div>
                                        {copied && <div className="text-xs text-link">Copied to clipboard</div>}
                                    </div>

                                    <div className="scrollbar-thin h-[calc(100%-32px)] min-h-[280px] overflow-y-auto pr-2">
                                        {loading ? (
                                            <div className="flex h-full flex-col items-center justify-center gap-4">
                                                <motion.span
                                                    animate={{ rotate: 360 }}
                                                    transition={{ repeat: Infinity, duration: 2.4, ease: "linear" }}
                                                    className="flex h-12 w-12 items-center justify-center rounded-full bg-ink text-on-primary"
                                                >
                                                    <Sparkles size={20} />
                                                </motion.span>
                                                <p className="text-body-sm text-mute">Generating interactive summary view...</p>
                                            </div>
                                        ) : displayedSummary ? (
                                            <div className="space-y-3">
                                                {outputSentences.map((sentence, index) => {
                                                    const isKeySentence = Array.from(keyTerms).some((term) =>
                                                        sentence.toLowerCase().includes(term.toLowerCase())
                                                    );
                                                    return (
                                                        <motion.p
                                                            key={`${sentence}-${index}-window`}
                                                            initial={{ opacity: 0, x: 8 }}
                                                            animate={{ opacity: 1, x: 0 }}
                                                            transition={{ delay: index * 0.05 }}
                                                            className={isKeySentence ? "sentence-key" : "sentence-default"}
                                                        >
                                                            {sentence}
                                                        </motion.p>
                                                    );
                                                })}
                                                {displayedSummary.length < summary.length && (
                                                    <span className="ml-2 inline-block h-5 w-0.5 animate-blink bg-ink align-middle" />
                                                )}
                                            </div>
                                        ) : (
                                            <p className="text-body-sm leading-6 text-mute">{DEFAULT_OUTPUT}</p>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </motion.section>
                    </div>
                </>
            )}
        </AnimatePresence>
    );
}

function HistoryDrawer({ historyOpen, setHistoryOpen, historyItems, loadHistoryItem, clearHistory }) {
    return (
        <AnimatePresence>
            {historyOpen && (
                <>
                    <motion.button
                        type="button"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setHistoryOpen(false)}
                        className="fixed inset-0 z-30 bg-ink/15 backdrop-blur-sm"
                    />
                    <motion.aside
                        initial={{ x: -420 }}
                        animate={{ x: 0 }}
                        exit={{ x: -420 }}
                        transition={{ type: "spring", damping: 25, stiffness: 220 }}
                        className="fixed left-0 top-0 z-40 flex h-full w-full max-w-md flex-col border-r border-hairline bg-canvas shadow-elevation-5"
                    >
                        <div className="flex items-center justify-between border-b border-hairline px-5 py-4">
                            <div>
                                <p className="caption-mono mb-1">Recent Runs</p>
                                <h3 className="text-display-sm text-ink">Summary history.</h3>
                            </div>
                            <button type="button" onClick={() => setHistoryOpen(false)} className="icon-btn">
                                <X size={16} />
                            </button>
                        </div>
                        <div className="scrollbar-thin flex flex-1 flex-col gap-1 overflow-y-auto p-3">
                            {historyItems.length === 0 ? (
                                <div className="rounded-lg bg-canvas-soft p-8 text-center">
                                    <p className="text-body-md text-body">
                                        Your recent summaries will appear here for quick recall.
                                    </p>
                                </div>
                            ) : (
                                historyItems.map((item) => (
                                    <button
                                        key={item.id}
                                        type="button"
                                        onClick={() => loadHistoryItem(item)}
                                        className="history-row group w-full"
                                    >
                                        <div className="mb-1 flex items-center justify-between gap-2">
                                            <span className="badge-secondary">{item.preset}</span>
                                            <span className="text-xs text-mute">{item.createdAt}</span>
                                        </div>
                                        <p className="line-clamp-2 text-body-sm text-ink">{item.inputPreview || "Uploaded file summary"}</p>
                                        <p className="mt-1 font-mono text-xs text-mute">{item.sourceFile || "Manual input"}</p>
                                    </button>
                                ))
                            )}
                        </div>
                        {historyItems.length > 0 && (
                            <div className="border-t border-hairline p-3">
                                <button type="button" onClick={clearHistory} className="btn-outline w-full">
                                    Clear local history
                                </button>
                            </div>
                        )}
                    </motion.aside>
                </>
            )}
        </AnimatePresence>
    );
}

function Footer() {
    const columns = [
        {
            title: "Product",
            links: ["Summarize", "Analytics", "History", "API"],
        },
        {
            title: "Models",
            links: ["Extractive AI", "PEGASUS", "Benchmarks", "Training"],
        },
        {
            title: "Resources",
            links: ["Documentation", "Guides", "Support", "Changelog"],
        },
        {
            title: "Company",
            links: ["About", "Blog", "Privacy", "Terms"],
        },
    ];

    return (
        <footer id="documentation" className="border-t border-hairline bg-canvas px-4 py-16 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-page">
                <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-4">
                    {columns.map((column) => (
                        <div key={column.title}>
                            <p className="caption-mono mb-4 text-mute">{column.title}</p>
                            <ul className="space-y-2">
                                {column.links.map((link) => (
                                    <li key={link}>
                                        <button type="button" className="text-body-sm text-body transition hover:text-ink">
                                            {link}
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
                <div className="mt-12 flex flex-wrap items-center justify-between gap-4 border-t border-hairline pt-8">
                    <div className="flex items-center gap-2">
                        <div className="flex h-5 w-5 items-center justify-center rounded-sm bg-ink">
                            <Sparkles size={12} className="text-on-primary" />
                        </div>
                        <span className="text-body-sm text-mute">Text Summarization</span>
                    </div>
                    <p className="text-xs text-mute">Â© 2026 Text Summarization. All rights reserved.</p>
                </div>
            </div>
        </footer>
    );
}

function MetricCard({ label, value }) {
    return (
        <div className="metric-card">
            <p className="caption-mono mb-2 text-mute">{label}</p>
            <p className="text-display-sm text-ink">{value}</p>
        </div>
    );
}

function InsightCard({ icon, title, body }) {
    return (
        <div className="insight-card">
            <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-md bg-canvas-soft-2 text-ink">
                {icon}
            </div>
            <h4 className="text-body-sm font-medium text-ink">{title}</h4>
            <p className="mt-1 text-xs leading-5 text-body">{body}</p>
        </div>
    );
}

export default App;








