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
    Moon,
    Sparkles,
    Sun,
    Trash2,
    WandSparkles,
    X,
    Info,
} from "lucide-react";
import { exportVerificationReportPdf } from "./pdfExport";
import { formatTranscript, ensureTerminalPunctuation } from "./voiceUtils";

const SUMMARY_PRESETS = { short: 6, medium: 10, long: 15 };
const MODEL_OPTIONS = {
    bart: "BART",
    pegasus: "PEGASUS",
};

const DEFAULT_PLACEHOLDER = "Paste or type your text here...";
const DEFAULT_OUTPUT = "Your summary will appear here with highlighted key sentences and analytics.";
const HISTORY_KEY = "summarizer-history-v2";
const VOICE_INITIAL_TIMEOUT_MS = 5000;
const VOICE_PAUSE_TIMEOUT_MS = 2400;
const VOICE_MAX_SESSION_MS = 60000;

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
    const [summaryModel, setSummaryModel] = useState("bart");
    const [overview, setOverview] = useState("");
    const [summary, setSummary] = useState("");
    const [displayedSummary, setDisplayedSummary] = useState("");
    const [loading, setLoading] = useState(false);
    const [elapsedTime, setElapsedTime] = useState(0);
    const [error, setError] = useState("");
    const [analysis, setAnalysis] = useState(null);
    
    // --- STEP 2: Add the new state ---
    const [hallucinationAnalysis, setHallucinationAnalysis] = useState(null);
    const [hallucinationLoading, setHallucinationLoading] = useState(false);
    
    const [historyOpen, setHistoryOpen] = useState(false);
    const [resultsOpen, setResultsOpen] = useState(false);

    useEffect(() => {
        if (!loading) {
            setElapsedTime(0);
            return undefined;
        }
        const startTime = Date.now();
        const timer = window.setInterval(() => {
            setElapsedTime((Date.now() - startTime) / 1000);
        }, 100);
        return () => window.clearInterval(timer);
    }, [loading]);
    const [historyItems, setHistoryItems] = useState([]);
    const [isListening, setIsListening] = useState(false);
    const [voiceStatus, setVoiceStatus] = useState("idle");
    const [audioLevel, setAudioLevel] = useState(0);
    const [copied, setCopied] = useState(false);
    const [mobileNavOpen, setMobileNavOpen] = useState(false);
    const [isDragging, setIsDragging] = useState(false);

    const [theme, setTheme] = useState(() => {
        if (typeof window !== "undefined") {
            const stored = localStorage.getItem("app-theme");
            if (stored === "dark" || stored === "light") return stored;
            return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        }
        return "light";
    });

    useEffect(() => {
        const root = document.documentElement;
        if (theme === "dark") {
            root.classList.add("dark");
        } else {
            root.classList.remove("dark");
        }
        try {
            localStorage.setItem("app-theme", theme);
        } catch {
            // ignore in private browsing modes
        }
    }, [theme]);

    const toggleTheme = () => {
        setTheme((prev) => (prev === "dark" ? "light" : "dark"));
    };
    const recognitionRef = useRef(null);
    const baseTranscriptRef = useRef("");
    const recognitionActiveRef = useRef(false);
    const hasSpokenRef = useRef(false);
    const silenceTimeoutRef = useRef(null);
    const maxSessionTimeoutRef = useRef(null);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const mediaStreamRef = useRef(null);
    const animationFrameRef = useRef(null);

    useEffect(() => {
        const savedHistory = window.localStorage.getItem(HISTORY_KEY);
        if (savedHistory) {
            setHistoryItems(JSON.parse(savedHistory));
        }
    }, []);

    useEffect(() => {
        window.localStorage.setItem(HISTORY_KEY, JSON.stringify(historyItems));
    }, [historyItems]);

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
    const confidence = analysis?.content_accuracy_score ?? 0;
    const aiSignal = analysis?.ai_writing_signals?.score ?? 0;

    const handleUpload = (file) => {
        setUploadedFile(file || null);
        setError("");
    };

    const stopAudioAnalyser = () => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach((track) => track.stop());
            mediaStreamRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== "closed") {
            try {
                audioContextRef.current.close();
            } catch {
                // ignore close errors
            }
            audioContextRef.current = null;
        }
        setAudioLevel(0);
    };

    const startAudioAnalyser = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaStreamRef.current = stream;

            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) return;

            const audioCtx = new AudioContextClass();
            audioContextRef.current = audioCtx;

            const source = audioCtx.createMediaStreamSource(stream);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            analyserRef.current = analyser;

            const dataArray = new Uint8Array(analyser.frequencyBinCount);

            const updateMeter = () => {
                if (!recognitionActiveRef.current) return;
                analyser.getByteFrequencyData(dataArray);

                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) {
                    sum += dataArray[i];
                }
                const average = sum / dataArray.length;
                const normalized = Math.min(1, average / 64);
                setAudioLevel(normalized);

                // If user is actively speaking, reset pause timer
                if (normalized > 0.08 && hasSpokenRef.current) {
                    scheduleVoiceSilenceTimer();
                }

                animationFrameRef.current = requestAnimationFrame(updateMeter);
            };

            updateMeter();
        } catch {
            // Audio analyser is progressive enhancement; speech recognition works even if analyser fails
        }
    };

    const clearVoiceTimers = () => {
        if (silenceTimeoutRef.current) {
            window.clearTimeout(silenceTimeoutRef.current);
            silenceTimeoutRef.current = null;
        }
        if (maxSessionTimeoutRef.current) {
            window.clearTimeout(maxSessionTimeoutRef.current);
            maxSessionTimeoutRef.current = null;
        }
    };

    const scheduleVoiceSilenceTimer = () => {
        if (silenceTimeoutRef.current) {
            window.clearTimeout(silenceTimeoutRef.current);
            silenceTimeoutRef.current = null;
        }

        // If user has not spoken yet, give them 5 seconds to begin speaking
        if (!hasSpokenRef.current) {
            setVoiceStatus("awaiting");
            silenceTimeoutRef.current = window.setTimeout(() => {
                if (recognitionActiveRef.current && !hasSpokenRef.current) {
                    stopVoiceInput(false);
                    setError("Microphone stopped: No voice input was detected.");
                }
            }, VOICE_INITIAL_TIMEOUT_MS);
            return;
        }

        // If user is actively speaking, allow 2.4s pause tolerance between phrases
        setVoiceStatus("listening");
        silenceTimeoutRef.current = window.setTimeout(() => {
            if (recognitionActiveRef.current) {
                setVoiceStatus("finalizing");
                stopVoiceInput(true);
            }
        }, VOICE_PAUSE_TIMEOUT_MS);
    };

    const handleClear = () => {
        setText("");
        setDraftTranscript("");
        baseTranscriptRef.current = "";
        setUploadedFile(null);
        setIsDragging(false);
        setOverview("");
        setSummary("");
        setDisplayedSummary("");
        setAnalysis(null);
        setHallucinationAnalysis(null);
        setHallucinationLoading(false);
        setError("");
        setCopied(false);
        setResultsOpen(false);
        clearVoiceTimers();
        stopAudioAnalyser();
    };

    const persistHistory = (payload) => {
        const entry = {
            id: Date.now(),
            createdAt: new Date().toLocaleString(),
            inputPreview: payload.sourceText.slice(0, 200),
            overview: payload.overview,
            summary: payload.summary,
            analysis: payload.analysis,
            hallucinationAnalysis: payload.hallucinationAnalysis,
            sourceFile: payload.sourceFile,
            preset: payload.preset,
            model: payload.model,
        };
        setHistoryItems((current) => [entry, ...current].slice(0, 12));
    };

    const startVoiceInput = async () => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setError("Voice input is not supported in this browser. Please use Chrome, Edge, or Safari.");
            return;
        }
        if (recognitionActiveRef.current) {
            return;
        }

        hasSpokenRef.current = false;
        baseTranscriptRef.current = text.trim();
        setDraftTranscript("");
        setError("");

        if (!recognitionRef.current) {
            const recognition = new SpeechRecognition();
            recognition.lang = "en-US";
            recognition.interimResults = true;
            recognition.continuous = true;
            recognition.maxAlternatives = 1;

            recognition.onstart = () => {
                recognitionActiveRef.current = true;
                setIsListening(true);
                scheduleVoiceSilenceTimer();
            };

            recognition.onresult = (event) => {
                let finalTranscript = "";
                let interimTranscript = "";

                for (let index = event.resultIndex; index < event.results.length; index += 1) {
                    const result = event.results[index];
                    const segment = result[0]?.transcript || "";
                    if (!segment.trim()) {
                        continue;
                    }
                    if (result.isFinal) {
                        finalTranscript += `${segment} `;
                    } else {
                        interimTranscript += `${segment} `;
                    }
                }

                if (finalTranscript.trim() || interimTranscript.trim()) {
                    hasSpokenRef.current = true;
                }

                if (finalTranscript.trim()) {
                    const formattedFinal = formatTranscript(finalTranscript.trim());
                    const nextBase = baseTranscriptRef.current
                        ? `${baseTranscriptRef.current} ${formattedFinal}`.trim()
                        : formattedFinal;
                    baseTranscriptRef.current = nextBase;
                    setText(nextBase);
                }

                if (interimTranscript.trim()) {
                    setDraftTranscript(formatTranscript(interimTranscript.trim()));
                } else {
                    setDraftTranscript("");
                }

                scheduleVoiceSilenceTimer();
            };

            recognition.onspeechstart = () => {
                hasSpokenRef.current = true;
                scheduleVoiceSilenceTimer();
            };

            recognition.onspeechend = () => {
                scheduleVoiceSilenceTimer();
            };

            recognition.onerror = (event) => {
                if (event.error === "no-speech") {
                    if (!hasSpokenRef.current) {
                        setError("No voice input was detected.");
                    }
                    stopVoiceInput(false);
                    return;
                }
                const errorMap = {
                    "not-allowed": "Microphone permission was denied. Please allow microphone access and try again.",
                    "service-not-allowed": "Speech recognition service is not allowed in this browser.",
                    network: "A network issue interrupted voice input. Please try again.",
                    "audio-capture": "No microphone was found. Please check your audio input device.",
                    aborted: "",
                };
                if (errorMap[event.error]) {
                    setError(errorMap[event.error]);
                }
                stopVoiceInput(false);
            };

            recognition.onend = () => {
                if (recognitionActiveRef.current) {
                    stopVoiceInput(true);
                }
            };

            recognitionRef.current = recognition;
        }

        try {
            recognitionRef.current.start();
            startAudioAnalyser();

            maxSessionTimeoutRef.current = window.setTimeout(() => {
                if (recognitionActiveRef.current) {
                    stopVoiceInput(true);
                }
            }, VOICE_MAX_SESSION_MS);
        } catch {
            recognitionActiveRef.current = false;
            setIsListening(false);
            setVoiceStatus("idle");
            stopAudioAnalyser();
            clearVoiceTimers();
            setError("Voice input could not start. If the microphone is already active, stop it and try again.");
        }
    };

    const stopVoiceInput = (autoCommit = false) => {
        recognitionActiveRef.current = false;
        try {
            recognitionRef.current?.stop();
        } catch {
            recognitionRef.current?.abort?.();
        }

        stopAudioAnalyser();
        clearVoiceTimers();

        setIsListening(false);
        setVoiceStatus("idle");
        setDraftTranscript("");

        if (autoCommit && hasSpokenRef.current && baseTranscriptRef.current) {
            const formatted = ensureTerminalPunctuation(baseTranscriptRef.current);
            baseTranscriptRef.current = formatted;
            setText(formatted);
        }
        hasSpokenRef.current = false;
    };

    const handleCopy = async () => {
        if (!summary) return;
        await navigator.clipboard.writeText(summary);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
    };

    const handleDownloadPdf = () => {
        if (!summary) return;
        exportVerificationReportPdf({
            summary,
            overview,
            analysis,
            hallucinationAnalysis,
            summaryModel,
            summaryPreset,
            uploadedFile,
        });
    };

    const loadHistoryItem = (item) => {
        setOverview(item.overview || "");
        setSummary(item.summary);
        setDisplayedSummary(item.summary);
        setAnalysis(item.analysis);
        setHallucinationAnalysis(
            item.hallucinationAnalysis || null
        );
        setSummaryPreset(item.preset);
        setSummaryModel(item.model || "bart");
        setUploadedFile(null);
        setText(item.inputPreview);
        setHistoryOpen(false);
        setResultsOpen(true);
    };

    const handleSummarize = async () => {
        if (!text.trim() && !uploadedFile) {
            setError("Add some text or upload a file before summarizing.");
            return;
        }
        setLoading(true);
        setResultsOpen(true);
        setError("");
        setCopied(false);
        // Clear previous output so old summary does not linger on retry or failure
        setSummary("");
        setDisplayedSummary("");
        setOverview("");
        setAnalysis(null);
        setHallucinationAnalysis(null);
        setHallucinationLoading(false);

        try {
            const formData = new FormData();
            formData.append("text", text);
            formData.append("summary_preset", summaryPreset);
            formData.append("summary_length", String(SUMMARY_PRESETS[summaryPreset] || 3));
            formData.append("model_type", summaryModel);
            if (uploadedFile) {
                formData.append("file", uploadedFile);
            }
            const controller = new AbortController();
            // Long docs use chunked map-reduce (16k+ tokens ≈ 16-18 chunks) — allow 10 minutes
            const timeoutId = setTimeout(() => controller.abort(), 600_000);
            let response;
            try {
                response = await fetch("/summarize", {
                    method: "POST",
                    body: formData,
                    signal: controller.signal,
                });
            } catch (fetchErr) {
                if (fetchErr.name === "AbortError") {
                    throw new Error("Request timed out. The model may still be generating — please try again in a moment.");
                }
                throw new Error("Failed to reach the server. Make sure the Flask backend is running.");
            } finally {
                clearTimeout(timeoutId);
            }
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Unable to summarize this content.");
            }
            
            const generatedSummary = (data.summary || "").trim();
            if (!generatedSummary) {
                throw new Error("The model did not return a valid summary.");
            }

            setOverview(data.overview || "");
            setSummary(generatedSummary);
            setAnalysis(data.analysis || null);
            setResultsOpen(true);
            
            persistHistory({
                sourceText: uploadedFile ? `${uploadedFile.name} upload` : text,
                overview: data.overview || "",
                summary: generatedSummary,
                analysis: data.analysis || null,
                hallucinationAnalysis: null,
                sourceFile: data.source_file || null,
                preset: summaryPreset,
                model: summaryModel,
            });
            
            // Decoupled API call for hallucination detection
            setHallucinationLoading(true);
            fetch("/analyze_hallucination", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: data.source_text || text,
                    summary: generatedSummary,
                })
            })
            .then(res => res.json())
            .then(hallucinationData => {
                setHallucinationAnalysis(hallucinationData);
            })
            .catch(err => {
                console.error("Hallucination analysis failed:", err);
                setHallucinationAnalysis({ error: "Failed to analyze hallucinations", details: err.message });
            })
            .finally(() => {
                setHallucinationLoading(false);
            });
            
        } catch (requestError) {
            setError(requestError.message);
            setResultsOpen(false);
        } finally {
            setLoading(false);
        }
    };


    return (
        <div className="relative min-h-screen bg-canvas-soft text-ink">
            <TopNav
                theme={theme}
                toggleTheme={toggleTheme}
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
                    isListening={isListening}
                    voiceStatus={voiceStatus}
                    audioLevel={audioLevel}
                    startVoiceInput={startVoiceInput}
                    stopVoiceInput={stopVoiceInput}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    handleSummarize={handleSummarize}
                    hasSummary={Boolean(summary)}
                    setResultsOpen={setResultsOpen}
                />

                <FeatureGrid />

                <FaqSection />
            </main>

            <Footer />

            <HistoryDrawer
                historyOpen={historyOpen}
                setHistoryOpen={setHistoryOpen}
                historyItems={historyItems}
                loadHistoryItem={loadHistoryItem}
            />

            <ResultsWindow
                resultsOpen={resultsOpen}
                setResultsOpen={setResultsOpen}
                loading={loading}
                elapsedTime={elapsedTime}
                analysis={analysis}
                hallucinationAnalysis={hallucinationAnalysis}
                hallucinationLoading={hallucinationLoading}
                confidence={confidence}
                aiSignal={aiSignal}
                overview={overview}
                displayedSummary={displayedSummary}
                summary={summary}
                outputSentences={outputSentences}
                keyTerms={keyTerms}
                copied={copied}
                handleCopy={handleCopy}
                handleDownloadPdf={handleDownloadPdf}
                uploadedFile={uploadedFile}
                summaryModel={summaryModel}
                summaryPreset={summaryPreset}
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

function TopNav({ theme, toggleTheme, setHistoryOpen, mobileNavOpen, setMobileNavOpen }) {
    return (
        <nav className="nav-bar">
            <div className="mx-auto flex w-full max-w-page items-center justify-between px-4 sm:px-6 lg:px-8">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <div className="flex h-6 w-6 items-center justify-center rounded-sm bg-ink" aria-hidden="true">
                            <Sparkles size={14} className="text-on-primary" />
                        </div>
                        <a href="/" className="text-body-sm font-medium text-ink no-underline" aria-label="AI Text Summarizer — Home">Summarizer &amp; Verifier</a>
                    </div>
                </div>

                <div className="hidden items-center gap-2 sm:flex">
                    <button
                        type="button"
                        onClick={toggleTheme}
                        className="icon-btn text-ink transition hover:border-hairline-strong"
                        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
                        title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
                    >
                        {theme === "dark" ? <Sun size={16} className="text-amber-400" /> : <Moon size={16} />}
                    </button>
                    <button type="button" onClick={() => setHistoryOpen(true)} className="btn-secondary-sm">
                        History
                    </button>
                </div>

                <div className="flex items-center gap-2 sm:hidden">
                    <button
                        type="button"
                        onClick={toggleTheme}
                        className="icon-btn text-ink"
                        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
                        title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
                    >
                        {theme === "dark" ? <Sun size={16} className="text-amber-400" /> : <Moon size={16} />}
                    </button>
                    <button
                        type="button"
                        onClick={() => setMobileNavOpen(!mobileNavOpen)}
                        className="icon-btn sm:hidden"
                        aria-label="Toggle menu"
                    >
                        {mobileNavOpen ? <X size={18} /> : <Menu size={18} />}
                    </button>
                </div>
            </div>

            <AnimatePresence>
                {mobileNavOpen && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="absolute left-0 right-0 top-16 overflow-hidden border-b border-hairline bg-canvas sm:hidden"
                    >
                        <div className="flex flex-col gap-1 px-4 py-4">
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
                            <button
                                type="button"
                                onClick={() => {
                                    toggleTheme();
                                    setMobileNavOpen(false);
                                }}
                                className="btn-ghost justify-start gap-2"
                            >
                                {theme === "dark" ? <Sun size={15} className="text-amber-400" /> : <Moon size={15} />}
                                <span>{theme === "dark" ? "Switch to Light Theme" : "Switch to Dark Theme"}</span>
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </nav>
    );
}

function HeroBand({ onSummarize, loading }) {
    return (
        <section className="relative overflow-hidden bg-canvas">
            <div className="pointer-events-none absolute inset-0 overflow-hidden">
                <div className="mesh-gradient absolute -left-1/4 -top-1/4 h-[600px] w-[800px] animate-mesh-drift" />
                <div className="mesh-gradient absolute -right-1/4 top-0 h-[500px] w-[700px] animate-mesh-drift [animation-delay:3s]" />
            </div>

            <div className="relative z-10 mx-auto max-w-page px-4 py-16 sm:px-6 sm:py-24 lg:px-8 lg:py-32">
                <div className="mx-auto max-w-3xl text-center">
                    <h1 className="text-display-xl text-ink sm:text-[48px]">
                        Summarize any text.<br/> Verify the facts.
                    </h1>
                    <p className="mx-auto mt-6 max-w-xl text-body-lg text-body">
                        Paste, upload, or speak your content. Get concise summaries equipped with deep Natural Language Inference to detect hallucinated claims.
                    </p>
                    <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                        <button type="button" onClick={onSummarize} disabled={loading} className="btn-primary">
                            <Sparkles size={16} />
                            {loading ? "Summarizing..." : "Start Summarizing"}
                        </button>
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
        isListening,
        voiceStatus = "idle",
        audioLevel = 0,
        startVoiceInput,
        stopVoiceInput,
        isDragging,
        setIsDragging,
        handleSummarize,
        hasSummary,
        setResultsOpen,
    } = props;


    return (
        <motion.section
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
                    {uploadedFile ? (
                        <span className="badge-secondary">File attached</span>
                    ) : (
                        <>
                            <span className="badge-secondary">{inputWordCount} words</span>
                            <span className="badge-secondary">{inputCharacterCount} chars</span>
                        </>
                    )}
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
                        <div className="flex h-10 w-10 items-center justify-center rounded-md bg-canvas-soft-2 text-ink" aria-hidden="true">
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
                        <input type="file" accept=".txt,.md,.csv,.json,.xml,.html,.htm,.log,.py,.js,.ts,.css,.java,.c,.cpp,.rb,.go,.php,.sql,.yaml,.yml,.ini,.rtf,.pdf,.docx,.pptx" className="hidden" onChange={(event) => handleUpload(event.target.files?.[0] ?? null)} />
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
                {isListening && (
                    <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between gap-3 rounded-md border border-violet/30 bg-canvas/95 px-3.5 py-2.5 text-body-sm shadow-elevation-2 backdrop-blur-sm">
                        <div className="flex items-center gap-2.5 overflow-hidden">
                            <span className="relative flex h-2.5 w-2.5 shrink-0">
                                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet opacity-75"></span>
                                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-violet"></span>
                            </span>
                            <div className="truncate text-body-sm font-medium text-ink">
                                {draftTranscript ? (
                                    <span>
                                        <span className="font-normal text-mute">Hearing: </span>
                                        &quot;{draftTranscript}&quot;
                                    </span>
                                ) : voiceStatus === "awaiting" ? (
                                    <span className="font-normal italic text-mute">
                                        Speak now (auto-stops after 5s if silent)...
                                    </span>
                                ) : voiceStatus === "finalizing" ? (
                                    <span className="font-normal italic text-violet-deep">
                                        Silence detected — saving transcript...
                                    </span>
                                ) : (
                                    <span className="font-normal italic text-mute">Listening...</span>
                                )}
                            </div>
                        </div>
                        <button
                            type="button"
                            onClick={() => stopVoiceInput(true)}
                            className="shrink-0 rounded px-2 py-0.5 text-xs font-semibold text-violet hover:bg-violet-soft/40 transition"
                        >
                            Done
                        </button>
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
                        onClick={isListening ? () => stopVoiceInput(true) : startVoiceInput}
                        className={`btn-outline relative overflow-hidden transition-all ${
                            isListening
                                ? "border-violet bg-violet-soft/60 text-violet-deep font-medium shadow-sm"
                                : ""
                        }`}
                        title={isListening ? "Click to stop recording" : "Click to speak"}
                    >
                        {isListening ? (
                            <span className="flex items-center gap-2">
                                <span className="relative flex h-2 w-2">
                                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet opacity-75"></span>
                                    <span className="relative inline-flex h-2 w-2 rounded-full bg-violet"></span>
                                </span>
                                {voiceStatus === "awaiting" && <span>Waiting for voice...</span>}
                                {voiceStatus === "listening" && (
                                    <span className="flex items-center gap-1.5">
                                        <span>Recording</span>
                                        <span className="inline-flex h-3 items-end gap-0.5">
                                            <span
                                                className="w-0.5 rounded-full bg-violet transition-all duration-75"
                                                style={{ height: `${Math.max(3, audioLevel * 14)}px` }}
                                            />
                                            <span
                                                className="w-0.5 rounded-full bg-violet transition-all duration-75"
                                                style={{ height: `${Math.max(5, audioLevel * 18)}px` }}
                                            />
                                            <span
                                                className="w-0.5 rounded-full bg-violet transition-all duration-75"
                                                style={{ height: `${Math.max(3, audioLevel * 11)}px` }}
                                            />
                                        </span>
                                    </span>
                                )}
                                {voiceStatus === "finalizing" && <span>Finalizing...</span>}
                            </span>
                        ) : (
                            <>
                                <Mic size={14} />
                                Voice Input
                            </>
                        )}
                    </button>
                    <button type="button" onClick={handleClear} className="btn-outline">
                        <Trash2 size={14} />
                        Clear
                    </button>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                    {hasSummary && (
                        <button
                            type="button"
                            onClick={() => setResultsOpen(true)}
                            className="btn-outline"
                        >
                            <Sparkles size={14} />
                            View Summary Window
                        </button>
                    )}
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

function DotMatrixLoader() {
    const [step, setStep] = useState(0);

    useEffect(() => {
        const interval = window.setInterval(() => {
            setStep((s) => (s + 1) % 12);
        }, 120);
        return () => window.clearInterval(interval);
    }, []);

    const activeSet = new Set([
        step,
        (step + 1) % 12,
        (step + 6) % 12,
        (step + 7) % 12,
    ]);

    const getPerimeterIdx = (r, c) => {
        if (r === 0) return c; // top row: 0, 1, 2, 3
        if (c === 3) return 3 + r; // right col: 4, 5, 6
        if (r === 3) return 6 + (3 - c); // bottom row: 6, 7, 8, 9
        if (c === 0) return 9 + (3 - r); // left col: 10, 11
        return -1; // inner 4 dots (1,1), (1,2), (2,1), (2,2)
    };

    return (
        <div className="grid grid-cols-4 gap-1.5 w-fit select-none" aria-hidden="true">
            {[0, 1, 2, 3].map((r) =>
                [0, 1, 2, 3].map((c) => {
                    const pIdx = getPerimeterIdx(r, c);
                    const isActive = pIdx !== -1 && activeSet.has(pIdx);
                    return (
                        <span
                            key={`${r}-${c}`}
                            className={`h-2.5 w-2.5 sm:h-3 sm:w-3 rounded-full transition-colors duration-150 ${
                                isActive ? "bg-ink" : "bg-hairline"
                            }`}
                        />
                    );
                })
            )}
        </div>
    );
}

function FeatureGrid() {
    const features = [
        {
            title: "BART model.",
            body: "State-of-the-art bidirectional and autoregressive transformer model for high-quality abstractive summarization.",
            mono: "bart",
        },
        {
            title: "PEGASUS model.",
            body: "Pre-trained with gap-sentence generation tailored for fluent, human-like summaries on longer documents.",
            mono: "pegasus",
        },
    ];

    return (
        <section className="mb-16">
            <div className="mb-8 text-center">
                <p className="caption-mono mb-2">Capabilities</p>
                <h2 className="text-display-lg text-ink">A compute model for all workloads.</h2>
                <p className="mx-auto mt-4 max-w-xl text-body-md text-body">
                    Two state-of-the-art transformer summarization engines, one unified interface.
                </p>
            </div>
            <div className="grid gap-6 md:grid-cols-2">
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
        elapsedTime,
        analysis,
        hallucinationAnalysis,
        hallucinationLoading,
        confidence,
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

    const [activeTab, setActiveTab] = useState("summary");

    // Calculate the overall risk percentage safely
    const hasHallucinationError = hallucinationAnalysis && hallucinationAnalysis.error;
    const hallucinationPercentage  = hallucinationLoading ? "..." : (hasHallucinationError ? "Err" : (hallucinationAnalysis 
        ? (hallucinationAnalysis.hallucination_ratio * 100).toFixed(1)
        : "0.0"));

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
                            className={`flex max-h-[92vh] min-h-0 w-[min(1240px,96vw)] flex-col overflow-hidden rounded-lg bg-canvas p-4 shadow-elevation-5 sm:p-6 ${
                                loading ? "min-h-[440px] sm:min-h-[500px]" : ""
                            }`}
                        >
                            {loading ? (
                                <div className="relative flex flex-1 flex-col items-center justify-center p-6 sm:p-12">
                                    <button
                                        type="button"
                                        onClick={() => setResultsOpen(false)}
                                        className="icon-btn absolute right-0 top-0 text-mute hover:text-ink transition"
                                        aria-label="Close"
                                    >
                                        <X size={18} />
                                    </button>
                                    <div className="flex items-center gap-4 sm:gap-5">
                                        <DotMatrixLoader />
                                        <div className="flex items-baseline gap-2.5 sm:gap-3">
                                            <span className="text-base sm:text-lg font-normal text-ink">
                                                Generating summary
                                            </span>
                                            <span className="font-mono text-sm sm:text-base text-mute">
                                                {(elapsedTime || 0).toFixed(1)}s
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <>
                                    <div className="mb-5 flex items-start justify-between gap-4">
                                <div>
                                    <p className="caption-mono mb-2">Interactive Results</p>
                                    <h3 className="text-display-md text-ink">Verified Summary.</h3>
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

                                <MetricCard
                                    label="Summary Confidence"
                                    value={`${confidence}%`}
                                />

                                <MetricCard
                                    label="Hallucination Ratio"
                                    value={hallucinationLoading ? "Loading..." : `${hallucinationPercentage}%`}
                                />

                                <MetricCard
                                    label="Supported Claims"
                                    value={hallucinationLoading ? "Loading..." : `${hallucinationAnalysis?.supported_claims ?? 0}`}
                                />

                                <MetricCard
                                    label="Contradicted Claims"
                                    value={hallucinationLoading ? "Loading..." : `${hallucinationAnalysis?.contradicted_claims ?? 0}`}
                                />

                            </div>

                            <GuideBox />

                            <div className="grid min-h-0 flex-1 gap-5 overflow-hidden lg:grid-cols-[0.9fr_1.1fr]">
                                <div className="scrollbar-thin flex min-h-0 flex-col gap-4 overflow-y-auto pr-1">
                                    <div className="rounded-md border border-hairline bg-link-bg-soft/40 px-5 py-4">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Sparkles size={14} className="text-link" />
                                            <span className="caption-mono font-semibold text-link uppercase tracking-wider">Document Description</span>
                                            <span className="text-[10px] text-ink-muted border border-link/20 rounded px-1.5 py-0.5 bg-canvas/60">Executive Overview</span>
                                        </div>
                                        <p className="text-body-sm leading-6 text-ink font-medium">
                                            {overview || "A clear executive description of the source material will appear here."}
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
                                                        ? `${analysis.input_word_count} words, ${analysis.input_sentence_count} sentences`
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
                                                title="Verification Status"
                                                body={
                                                    hallucinationLoading
                                                        ? "Analyzing claims..."
                                                        : hasHallucinationError
                                                            ? "❌ Failed to analyze hallucinations."
                                                            : hallucinationAnalysis?.overall_label === "HALLUCINATION_DETECTED"
                                                                ? "⚠️ Contradictory claims detected. Review the evidence below."
                                                                : hallucinationAnalysis?.overall_label === "POTENTIAL_HALLUCINATION"
                                                                    ? "⚠️ Some claims could not be reliably verified."
                                                                    : "✅ The generated summary is supported by the source."
                                                }
                                            />
                                            <InsightCard
                                                icon={<Sparkles size={16} />}
                                                title="Top Keywords"
                                                body={analysis?.top_keywords?.join(", ") || "Keywords will surface after summarization."}
                                            />
                                        </div>
                                    </div>
                                </div>

                                <div className="min-h-0 overflow-hidden rounded-md border border-hairline bg-canvas p-4 sm:p-5 flex flex-col">
                                    <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-hairline pb-3">
                                        <div className="flex items-center gap-2">
                                            <button
                                                type="button"
                                                onClick={() => setActiveTab("summary")}
                                                className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
                                                    activeTab === "summary"
                                                        ? "bg-ink text-on-primary shadow-sm"
                                                        : "bg-canvas-soft text-body hover:text-ink hover:bg-hairline/60"
                                                }`}
                                            >
                                                Generated Summary
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => setActiveTab("verification")}
                                                className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition ${
                                                    activeTab === "verification"
                                                        ? "bg-ink text-on-primary shadow-sm"
                                                        : "bg-canvas-soft text-body hover:text-ink hover:bg-hairline/60"
                                                }`}
                                            >
                                                <span>Factual Claims &amp; NLI</span>
                                                {hallucinationLoading && (
                                                    <span className="h-1.5 w-1.5 rounded-full bg-link animate-ping" />
                                                )}
                                                {hallucinationAnalysis?.sentence_analysis && (
                                                    <span className="ml-1 rounded-full bg-link-bg-soft px-1.5 py-0.2 text-[10px] text-link">
                                                        {hallucinationAnalysis.sentence_analysis.length}
                                                    </span>
                                                )}
                                            </button>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {copied && <span className="text-xs text-link font-medium">Copied!</span>}
                                            <button
                                                type="button"
                                                onClick={handleCopy}
                                                className="btn-outline text-xs py-1 px-2.5"
                                                aria-label="Copy summary text"
                                            >
                                                <Copy size={13} />
                                                Copy
                                            </button>
                                            <button
                                                type="button"
                                                onClick={handleDownloadPdf}
                                                className="btn-outline text-xs py-1 px-2.5"
                                                aria-label="Export summary to PDF"
                                            >
                                                <Download size={13} />
                                                PDF
                                            </button>
                                        </div>
                                    </div>

                                    <div className="scrollbar-thin flex-1 min-h-[280px] overflow-y-auto pr-2 pb-6">
                                        {activeTab === "summary" ? (
                                            loading ? (
                                                <div className="flex h-full min-h-[220px] flex-col items-center justify-center gap-4">
                                                    <motion.span
                                                        animate={{ rotate: 360 }}
                                                        transition={{ repeat: Infinity, duration: 2.4, ease: "linear" }}
                                                        className="flex h-12 w-12 items-center justify-center rounded-full bg-ink text-on-primary"
                                                    >
                                                        <Sparkles size={20} />
                                                    </motion.span>
                                                    <p className="text-body-sm text-mute">Generating summary...</p>
                                                </div>
                                            ) : summary ? (
                                                <div className="space-y-3">
                                                    {outputSentences.map((sentence, index) => {
                                                        const isKeySentence = Array.from(keyTerms || []).some((term) =>
                                                            sentence.toLowerCase().includes(term.toLowerCase())
                                                        );
                                                        return (
                                                            <motion.p
                                                                key={`${sentence.slice(0, 24)}-${index}-modal`}
                                                                initial={{ opacity: 0, y: 4 }}
                                                                animate={{ opacity: 1, y: 0 }}
                                                                transition={{ delay: index * 0.04 }}
                                                                className={`rounded-md border px-4 py-3 text-body-sm leading-relaxed transition ${
                                                                    isKeySentence
                                                                        ? "border-link/30 bg-link-bg-soft/20 text-ink font-medium"
                                                                        : "border-hairline bg-canvas-soft text-ink"
                                                                }`}
                                                            >
                                                                {sentence}
                                                            </motion.p>
                                                        );
                                                    })}
                                                    {displayedSummary.length < summary.length && (
                                                        <span className="inline-block h-4 w-[2px] animate-pulse bg-link align-middle ml-1" />
                                                    )}
                                                    {analysis && (
                                                        <div className="mt-4 flex flex-wrap items-center gap-2 pt-2 border-t border-hairline text-xs text-mute">
                                                            <span>Words: <strong className="text-ink">{analysis.summary_word_count}</strong></span>
                                                            <span>•</span>
                                                            <span>Compression: <strong className="text-ink">{analysis.compression_ratio}%</strong></span>
                                                            <span>•</span>
                                                            <span>Est. read time: <strong className="text-ink">{analysis.summary_reading_time_minutes} min</strong></span>
                                                        </div>
                                                    )}
                                                </div>
                                            ) : (
                                                <div className="flex h-full min-h-[220px] items-center justify-center">
                                                    <p className="text-body-sm leading-6 text-mute">{DEFAULT_OUTPUT}</p>
                                                </div>
                                            )
                                        ) : hallucinationLoading ? (
                                            <div className="flex h-full min-h-[240px] flex-col items-center justify-center gap-4">
                                                <motion.span
                                                    animate={{ rotate: 360 }}
                                                    transition={{ repeat: Infinity, duration: 2.4, ease: "linear" }}
                                                    className="flex h-12 w-12 items-center justify-center rounded-full bg-ink text-on-primary"
                                                >
                                                    <Sparkles size={20} />
                                                </motion.span>
                                                <p className="text-body-sm text-mute">
                                                    Verifying facts against source document (NLI DeBERTa)...
                                                </p>
                                            </div>
                                        ) : hasHallucinationError ? (
                                            <div className="rounded-md border border-error-subtle bg-error-subtle/20 p-4 text-center">
                                                <p className="text-body-sm text-error-deep font-medium">Verification Error</p>
                                                <p className="mt-1 text-xs text-mute">{hallucinationAnalysis.error}</p>
                                                <p className="mt-2 text-xs text-ink">Note: Your generated summary was generated successfully and is preserved intact in the &quot;Generated Summary&quot; tab.</p>
                                            </div>
                                        ) : hallucinationAnalysis ? (
                                            <div className="space-y-4 pt-2">
                                        
                                                <div className="rounded-md border border-hairline bg-canvas-soft p-4">
                                                    <p className="caption-mono mb-2 text-mute">
                                                        Hallucination analysis
                                                    </p>
                                        
                                                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                                        
                                                        <MetricCard
                                                            label="Claims"
                                                            value={hallucinationAnalysis.total_claims ?? 0}
                                                        />
                                        
                                                        <MetricCard
                                                            label="Supported"
                                                            value={hallucinationAnalysis.supported_claims ?? 0}
                                                        />
                                        
                                                        <MetricCard
                                                            label="Contradicted"
                                                            value={hallucinationAnalysis.contradicted_claims ?? 0}
                                                        />
                                        
                                                        <MetricCard
                                                            label="Unverifiable"
                                                            value={hallucinationAnalysis.unverifiable_claims ?? 0}
                                                        />
                                        
                                                    </div>
                                                </div>
                                        
                                                {hallucinationAnalysis.sentence_analysis?.map(
                                                    (item, index) => {
                                        
                                                        const isSupported =
                                                            item.label === "ENTAILMENT";
                                        
                                                        const isContradiction =
                                                            item.label === "CONTRADICTION";
                                        
                                                        const isUnverifiable =
                                                            item.label === "UNVERIFIABLE";
                                        
                                                        return (
                                                            <motion.div
                                                                key={`${index}-xai`}
                                                                initial={{
                                                                    opacity: 0,
                                                                    y: 8,
                                                                }}
                                                                animate={{
                                                                    opacity: 1,
                                                                    y: 0,
                                                                }}
                                                                transition={{
                                                                    delay: index * 0.08,
                                                                }}
                                                                className="rounded-md border border-hairline bg-canvas-soft p-4"
                                                            >
                                        
                                                                <div className="flex items-start justify-between gap-3">
                                        
                                                                    <div className="flex min-w-0 gap-2">
                                        
                                                                        <span className="mt-0.5">
                                                                            {isSupported
                                                                                ? "✅"
                                                                                : isContradiction
                                                                                    ? "⚠️"
                                                                                    : "❔"}
                                                                        </span>
                                        
                                                                        <p className="text-body-sm leading-6 text-ink">
                                                                            {item.claim}
                                                                        </p>
                                        
                                                                    </div>
                                        
                                                                    <span className="shrink-0 rounded-full bg-canvas px-2.5 py-1 text-xs font-medium">
                                                                        {item.label}
                                                                    </span>
                                        
                                                                </div>
                                        
                                                                <div className="mt-3 grid grid-cols-3 gap-2">
                                        
                                                                    <div className="rounded bg-canvas p-2">
                                                                        <p className="text-xs text-mute">
                                                                            Entailment
                                                                        </p>
                                        
                                                                        <p className="mt-1 text-sm font-medium">
                                                                            {(
                                                                                item.scores?.entailment * 100
                                                                            ).toFixed(1)}%
                                                                        </p>
                                                                    </div>
                                        
                                                                    <div className="rounded bg-canvas p-2">
                                                                        <p className="text-xs text-mute">
                                                                            Contradiction
                                                                        </p>
                                        
                                                                        <p className="mt-1 text-sm font-medium">
                                                                            {(
                                                                                item.scores?.contradiction * 100
                                                                            ).toFixed(1)}%
                                                                        </p>
                                                                    </div>
                                        
                                                                    <div className="rounded bg-canvas p-2">
                                                                        <p className="text-xs text-mute">
                                                                            Unverifiable
                                                                        </p>
                                        
                                                                        <p className="mt-1 text-sm font-medium">
                                                                            {(
                                                                                item.scores?.unverifiable * 100
                                                                            ).toFixed(1)}%
                                                                        </p>
                                                                    </div>
                                        
                                                                </div>
                                        
                                                                {item.evidence && (
                                                                    <div className="mt-3 rounded-md border border-hairline bg-canvas p-3">
                                        
                                                                        <p className="caption-mono mb-1 text-mute">
                                                                            Source evidence
                                                                        </p>
                                        
                                                                        <p className="text-body-sm leading-6 text-body">
                                                                            {item.evidence}
                                                                        </p>
                                        
                                                                        <p className="mt-2 text-xs text-mute">
                                                                            Evidence similarity:
                                                                            {" "}
                                                                            {(
                                                                                item.evidence_similarity * 100
                                                                            ).toFixed(1)}%
                                                                        </p>
                                        
                                                                    </div>
                                                                )}
                                        
                                                                {isUnverifiable && (
                                                                    <p className="mt-3 text-xs text-mute">
                                                                        The source did not provide enough
                                                                        evidence to verify this claim.
                                                                    </p>
                                                                )}
                                        
                                                            </motion.div>
                                                        );
                                                    }
                                                )}
                                        
                                            </div>
                                        ) : (
                                            <div className="flex h-full min-h-[220px] flex-col items-center justify-center p-6 text-center">
                                                <p className="text-body-sm text-mute">
                                                    Factual verification analysis will appear here after summarization.
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                                </>
                            )}
                        </motion.section>
                    </div>
                </>
            )}
        </AnimatePresence>
    );
}

function HistoryDrawer({ historyOpen, setHistoryOpen, historyItems, loadHistoryItem }) {
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
                    </motion.aside>
                </>
            )}
        </AnimatePresence>
    );
}

function Footer() {
    return (
        <footer className="border-t border-hairline bg-canvas px-4 py-8 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-page">
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <div className="flex h-5 w-5 items-center justify-center rounded-sm bg-ink" aria-hidden="true">
                            <Sparkles size={12} className="text-on-primary" />
                        </div>
                        <span className="text-body-sm text-mute">AI Text Summarizer &amp; Verifier</span>
                    </div>
                    <nav className="flex items-center gap-4 text-xs text-mute" aria-label="Footer navigation">
                        <a href="/privacy" className="hover:text-ink transition-colors">Privacy Policy</a>
                        <span aria-hidden="true">·</span>
                        <span>&#169; {new Date().getFullYear()} All rights reserved.</span>
                    </nav>
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

function GuideBox() {
    const [open, setOpen] = useState(false);
    return (
        <div className="mb-5 rounded-md border border-hairline bg-link-bg-soft/30">
            <button type="button" onClick={() => setOpen(!open)} className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-ink hover:bg-canvas-soft transition-colors">
                <span className="flex items-center gap-2"><Info size={16} className="text-link" /> How to read these results</span>
                <span className="text-mute text-lg leading-none">{open ? "−" : "+"}</span>
            </button>
            <AnimatePresence>
                {open && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="border-t border-hairline px-4 py-4 text-xs text-body space-y-3">
                            <p><strong className="text-ink">Summary Confidence:</strong> The model's internal confidence score regarding the overall quality and relevance of the summary.</p>
                            <p><strong className="text-ink">Hallucination Ratio:</strong> The percentage of sentences in the summary that the AI suspects might be fabricated, contradicted, or unverifiable based on the source.</p>
                            <p><strong className="text-ink">Supported / Contradicted Claims:</strong> Breakdown of individual statements. Supported claims are entailed by the text. Contradicted claims directly oppose the source text.</p>
                            <p><strong className="text-ink">Top Keywords:</strong> The most significant technical terms and central topics extracted from your source document using TF-IDF.</p>
                            <p><strong className="text-ink">Explainable XAI Breakdown:</strong> A sentence-by-sentence analysis of the summary, showing exact source evidence and similarity scores used to verify facts.</p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

// export default App;


function HallucinationPanel({
    hallucinationAnalysis,
}) {
    if (!hallucinationAnalysis) {
        return null;
    }

    if (hallucinationAnalysis.error) {
        return (
            <div className="rounded-xl border border-red-200 bg-red-50 p-4">
                <p className="font-medium text-red-700">
                    Hallucination analysis failed
                </p>

                <p className="mt-1 text-sm text-red-600">
                    {hallucinationAnalysis.details}
                </p>
            </div>
        );
    }

    const {
        overall_label,
        overall_confidence,
        total_claims,
        supported_claims,
        contradicted_claims,
        unverifiable_claims,
        hallucinated_claims,
        hallucination_ratio,
        sentence_analysis,
    } = hallucinationAnalysis;

    const percentage = Math.round(
        overall_confidence * 100
    );

    return (
        <div className="space-y-4">

            <div className="rounded-xl border border-hairline bg-canvas p-5 text-ink">

                <div className="flex items-center justify-between">

                    <div>
                        <p className="text-sm text-mute">
                            Hallucination Detection
                        </p>

                        <h3 className="mt-1 text-xl font-semibold">
                            {overall_label}
                        </h3>
                    </div>

                    <div className="text-right">

                        <p className="text-2xl font-bold">
                            {percentage}%
                        </p>

                        <p className="text-xs text-mute">
                            prediction confidence
                        </p>

                    </div>
                </div>

                <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">

                    <Metric
                        label="Claims"
                        value={total_claims}
                    />

                    <Metric
                        label="Supported"
                        value={supported_claims}
                    />

                    <Metric
                        label="Contradicted"
                        value={contradicted_claims}
                    />

                    <Metric
                        label="Unverifiable"
                        value={unverifiable_claims}
                    />

                </div>

                <div className="mt-4 text-sm text-mute">
                    Hallucinated claim ratio:
                    {" "}
                    {Math.round(
                        hallucination_ratio * 100
                    )}%
                </div>

            </div>

            <div className="space-y-3">

                {sentence_analysis?.map(
                    (item, index) => (
                        <ClaimCard
                            key={index}
                            item={item}
                        />
                    )
                )}

            </div>

        </div>
    );
}


function Metric({ label, value }) {
    return (
        <div className="rounded-lg border border-hairline bg-canvas-soft p-3">
            <p className="text-xs text-mute">
                {label}
            </p>

            <p className="mt-1 text-lg font-semibold text-ink">
                {value}
            </p>
        </div>
    );
}

function ClaimCard({ item }) {

    const isBad =
        item.label === "CONTRADICTION";

    const isUnknown =
        item.label === "UNVERIFIABLE";

    const status =
        item.label === "ENTAILMENT"
            ? "Supported"
            : isBad
                ? "Hallucination"
                : "Not enough evidence";

    return (
        <div className="rounded-xl border border-hairline bg-canvas p-4 text-ink">

            <div className="flex items-start justify-between gap-4">

                <div className="min-w-0">

                    <p className="text-sm font-medium">
                        {item.claim}
                    </p>

                    <p className="mt-1 text-xs text-mute">
                        {status}
                        {" · "}
                        {Math.round(
                            item.confidence * 100
                        )}%
                    </p>

                </div>

                <span className="shrink-0 rounded-full border border-hairline bg-canvas-soft-2 px-2.5 py-1 text-xs font-medium text-ink">
                    {item.label}
                </span>

            </div>

            {item.evidence && (
                <div className="mt-3 rounded-lg border border-hairline bg-canvas-soft p-3">

                    <p className="text-xs font-medium text-mute">
                        Source evidence
                    </p>

                    <p className="mt-1 text-sm text-body">
                        {item.evidence}
                    </p>

                    <p className="mt-2 text-xs text-mute">
                        Evidence similarity:
                        {" "}
                        {Math.round(
                            item.evidence_similarity * 100
                        )}%
                    </p>

                </div>
            )}

            <div className="mt-3 grid grid-cols-3 gap-2 text-xs">

                <Score
                    label="Entailment"
                    value={item.scores.entailment}
                />

                <Score
                    label="Contradiction"
                    value={item.scores.contradiction}
                />

                <Score
                    label="Unverifiable"
                    value={item.scores.unverifiable}
                />

            </div>

        </div>
    );
}

function Score({ label, value }) {
    return (
        <div className="rounded border border-hairline bg-canvas-soft p-2">
            <div className="text-mute text-xs">
                {label}
            </div>

            <div className="font-medium text-ink">
                {Math.round(value * 100)}%
            </div>
        </div>
    );
}

/* ─────────────────────────── FAQ Section ─────────────────────────── */

const FAQ_ITEMS = [
    {
        q: "Why does BART or PEGASUS take longer to load?",
        a: "BART and PEGASUS are large transformer models (~1–2 GB each) that are downloaded on first use and cached locally. Subsequent runs are much faster.",
    },
    {
        q: "What file formats are supported?",
        a: "PDF, DOCX, PPTX, TXT, MD, CSV, JSON, HTML, XML, and most plain-text code or config files. Legacy .ppt files are not supported — please convert to .pptx first.",
    },
    {
        q: "Are my documents stored or shared?",
        a: "No. Text and files are processed entirely in memory and are never saved to disk or a database. Summary history lives only in your browser's localStorage and never leaves your device.",
    },
    {
        q: "What is hallucination detection?",
        a: "After summarization, a DeBERTa NLI (Natural Language Inference) model checks each sentence in the summary against the source text. Claims that are contradicted or unverifiable are flagged as potential hallucinations.",
    },
    {
        q: "Which model should I use?",
        a: "Use BART for the best overall quality on English documents. Use PEGASUS for news-style or journalism content. Both produce different trade-offs — experiment to see what works best for your source material.",
    },
];

function FaqSection() {
    const [openIndex, setOpenIndex] = useState(null);

    const toggle = (index) => {
        setOpenIndex((prev) => (prev === index ? null : index));
    };

    return (
        <section className="mb-16" aria-labelledby="faq-heading">
            <div className="mb-8 text-center">
                <p className="caption-mono mb-2">Support</p>
                <h2 id="faq-heading" className="text-display-lg text-ink">Frequently asked questions.</h2>
                <p className="mx-auto mt-4 max-w-xl text-body-md text-body">
                    Common questions about using the summarizer and its models.
                </p>
            </div>

            <div className="mx-auto max-w-2xl space-y-2">
                {FAQ_ITEMS.map((item, index) => (
                    <div key={index} className="rounded-lg border border-hairline bg-canvas overflow-hidden">
                        <button
                            type="button"
                            onClick={() => toggle(index)}
                            aria-expanded={openIndex === index}
                            className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left text-body-sm font-medium text-ink hover:bg-canvas-soft transition-colors"
                        >
                            <span>{item.q}</span>
                            <span className="shrink-0 text-mute transition-transform" style={{ transform: openIndex === index ? "rotate(45deg)" : "rotate(0deg)" }}>
                                +
                            </span>
                        </button>
                        <AnimatePresence initial={false}>
                            {openIndex === index && (
                                <motion.div
                                    key="content"
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.2, ease: "easeInOut" }}
                                    className="overflow-hidden"
                                >
                                    <p className="px-5 pb-4 text-body-sm text-body">{item.a}</p>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                ))}
            </div>
        </section>
    );
}

export default App;