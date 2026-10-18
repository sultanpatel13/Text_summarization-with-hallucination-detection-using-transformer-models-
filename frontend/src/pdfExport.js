import { jsPDF } from "jspdf";

/**
 * Generate and download an executive-grade AI Summary & Factual Verification PDF report.
 *
 * @param {Object} options
 * @param {string} options.summary - The generated summary text
 * @param {string} [options.overview] - High-level synthesis ("What it's about")
 * @param {Object} [options.analysis] - Text analytics (word counts, compression, reading time)
 * @param {Object} [options.hallucinationAnalysis] - NLI DeBERTa factual claims audit
 * @param {string} [options.summaryModel] - Model used ('bart' | 'pegasus')
 * @param {string} [options.summaryPreset] - Preset ('short' | 'medium' | 'long')
 * @param {File|Object|string} [options.uploadedFile] - File object or name of the input
 */
export function exportVerificationReportPdf(options = {}) {
    const {
        summary = "",
        overview = "",
        analysis = null,
        hallucinationAnalysis = null,
        summaryModel = "bart",
        summaryPreset = "medium",
        uploadedFile = null,
    } = options;

    if (!summary || !summary.trim()) {
        return;
    }

    const doc = new jsPDF({ orientation: "p", unit: "pt", format: "a4" });

    // Page Geometry (A4 in points: 595.28 x 841.89)
    const pageWidth = 595.28;
    const pageHeight = 841.89;
    const marginX = 40;
    const contentWidth = pageWidth - marginX * 2; // 515.28 pt
    const bottomLimit = 780; // Margin before page-break

    let currentY = 40;

    // Helper: Page-break checker
    const checkBreak = (neededHeight) => {
        if (currentY + neededHeight > bottomLimit) {
            doc.addPage();
            currentY = 52; // Ample breathing room below running header (y=32)
            return true;
        }
        return false;
    };

    /**
     * Strict font-aware text measurement and splitting helper.
     * Guarantees that splitTextToSize always uses the exact font family, style, and size
     * that will be used during rendering, preventing measurement mismatch and text overlaps.
     */
    const splitText = (text, maxWidth, fontSize, fontStyle = "normal", fontFamily = "helvetica") => {
        doc.setFont(fontFamily, fontStyle);
        doc.setFontSize(fontSize);
        return doc.splitTextToSize(text || "", maxWidth);
    };

    // 1. TOP BRAND ACCENT BAR
    doc.setFillColor(79, 70, 229); // Indigo #4f46e5
    doc.rect(0, 0, pageWidth, 5, "F");

    // 2. HEADER TITLE & SUBTITLE
    currentY += 15;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.setTextColor(15, 23, 42); // Slate #0f172a
    doc.text("AI Summary & Factual Verification Report", marginX, currentY);

    currentY += 16;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(100, 116, 139); // Slate #64748b
    doc.text(
        "Abstractive Synthesis with Deep Natural Language Inference (NLI) Audit",
        marginX,
        currentY
    );

    currentY += 16;

    // 3. METADATA STRIP (Rounded card with border)
    const sourceLabel =
        typeof uploadedFile === "string"
            ? uploadedFile
            : uploadedFile?.name || "Manual Text Input";
    const modelLabel =
        summaryModel?.toUpperCase() === "PEGASUS" ? "PEGASUS" : "BART";
    const presetLabel =
        summaryPreset ? summaryPreset.charAt(0).toUpperCase() + summaryPreset.slice(1) : "Medium";
    const formattedDate = new Date().toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });

    const metaBoxH = 38;
    doc.setFillColor(248, 250, 252); // Slate-50
    doc.setDrawColor(226, 232, 240); // Slate-200
    doc.roundedRect(marginX, currentY, contentWidth, metaBoxH, 4, 4, "FD");

    const colW = contentWidth / 4;
    const metaY1 = currentY + 14;
    const metaY2 = currentY + 28;

    // Meta Column 1: Source
    doc.setFont("helvetica", "bold");
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184); // Slate-400
    doc.text("SOURCE", marginX + 10, metaY1);
    const sourceLines = splitText(sourceLabel, colW - 16, 8.5, "bold");
    const displaySource = sourceLines.length > 1 ? `${sourceLines[0].slice(0, -3)}...` : sourceLines[0];
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(30, 41, 59);
    doc.text(displaySource, marginX + 10, metaY2);

    // Meta Column 2: Model
    doc.setFont("helvetica", "bold");
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184);
    doc.text("MODEL ENGINE", marginX + colW + 10, metaY1);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(30, 41, 59);
    doc.text(modelLabel, marginX + colW + 10, metaY2);

    // Meta Column 3: Preset
    doc.setFont("helvetica", "bold");
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184);
    doc.text("LENGTH PRESET", marginX + colW * 2 + 10, metaY1);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(30, 41, 59);
    doc.text(presetLabel, marginX + colW * 2 + 10, metaY2);

    // Meta Column 4: Date
    doc.setFont("helvetica", "bold");
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184);
    doc.text("GENERATED ON", marginX + colW * 3 + 10, metaY1);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);
    doc.setTextColor(30, 41, 59);
    doc.text(formattedDate, marginX + colW * 3 + 10, metaY2);

    currentY += metaBoxH + 14;

    // 4. EXECUTIVE ANALYTICS KPI GRID (4 Cards)
    if (analysis) {
        checkBreak(52);
        const kpiW = (contentWidth - 3 * 8) / 4;
        const kpiH = 44;

        const kpis = [
            {
                label: "SUMMARY WORDS",
                val: `${analysis.summary_word_count || 0}`,
                sub: `from ${analysis.input_word_count || 0} src`,
            },
            {
                label: "COMPRESSION",
                val: `${analysis.compression_ratio || 0}%`,
                sub: "content reduction",
            },
            {
                label: "READ TIME",
                val: `${analysis.summary_reading_time_minutes || 0} min`,
                sub: "estimated",
            },
            {
                label: "ACCURACY SCORE",
                val: `${analysis.content_accuracy_score || 0}%`,
                sub: "heuristic match",
            },
        ];

        kpis.forEach((kpi, idx) => {
            const x = marginX + idx * (kpiW + 8);
            doc.setFillColor(248, 250, 252);
            doc.setDrawColor(226, 232, 240);
            doc.roundedRect(x, currentY, kpiW, kpiH, 4, 4, "FD");

            doc.setFont("helvetica", "bold");
            doc.setFontSize(7);
            doc.setTextColor(148, 163, 184);
            doc.text(kpi.label, x + 8, currentY + 12);

            doc.setFont("helvetica", "bold");
            doc.setFontSize(12);
            doc.setTextColor(15, 23, 42);
            doc.text(kpi.val, x + 8, currentY + 27);

            doc.setFont("helvetica", "normal");
            doc.setFontSize(7);
            doc.setTextColor(100, 116, 139);
            doc.text(kpi.sub, x + 8, currentY + 38);
        });

        currentY += kpiH + 16;
    }

    // 5. CONTEXT OVERVIEW ("WHAT IT'S ABOUT")
    if (overview && overview.trim()) {
        const overviewLines = splitText(overview.trim(), contentWidth - 32, 9, "normal");
        const boxH = Math.max(46, overviewLines.length * 13.5 + 26);
        checkBreak(boxH);

        doc.setFillColor(248, 250, 252); // Slate-50
        doc.setDrawColor(226, 232, 240);
        doc.roundedRect(marginX, currentY, contentWidth, boxH, 4, 4, "FD");

        // Indigo accent bar on the left
        doc.setFillColor(79, 70, 229);
        doc.roundedRect(marginX, currentY, 3.5, boxH, 2, 2, "F");

        doc.setFont("helvetica", "bold");
        doc.setFontSize(7.5);
        doc.setTextColor(79, 70, 229);
        doc.text("DOCUMENT DESCRIPTION & EXECUTIVE OVERVIEW", marginX + 14, currentY + 14);

        doc.setFont("helvetica", "normal");
        doc.setFontSize(9);
        doc.setTextColor(30, 41, 59);
        let ovY = currentY + 27;
        overviewLines.forEach((line) => {
            doc.text(line, marginX + 14, ovY);
            ovY += 13.5;
        });

        currentY += boxH + 16;
    }

    // 6. GENERATED SUMMARY NARRATIVE SECTION
    checkBreak(55);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10.5);
    doc.setTextColor(15, 23, 42);
    doc.text("EXECUTIVE SUMMARY", marginX, currentY);

    // Thin section rule
    doc.setDrawColor(226, 232, 240);
    doc.line(marginX, currentY + 6, marginX + contentWidth, currentY + 6);
    currentY += 18;

    const summaryLines = splitText(summary.trim(), contentWidth, 9.5, "normal");
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(30, 41, 59);

    for (let i = 0; i < summaryLines.length; i++) {
        checkBreak(15);
        doc.text(summaryLines[i], marginX, currentY);
        currentY += 14.5;
    }

    currentY += 18;

    // 7. FACTUAL CLAIMS & HALLUCINATION ANALYSIS (NLI)
    if (hallucinationAnalysis && !hallucinationAnalysis.error) {
        checkBreak(80);

        doc.setFont("helvetica", "bold");
        doc.setFontSize(10.5);
        doc.setTextColor(15, 23, 42);
        doc.text("FACTUAL CONSISTENCY & NLI VERIFICATION AUDIT", marginX, currentY);

        doc.setDrawColor(226, 232, 240);
        doc.line(marginX, currentY + 6, marginX + contentWidth, currentY + 6);
        currentY += 16;

        // Overall Status Banner
        const isHallucination =
            hallucinationAnalysis.overall_label === "HALLUCINATION_DETECTED";
        const isPotential =
            hallucinationAnalysis.overall_label === "POTENTIAL_HALLUCINATION";

        let bannerBg = [240, 253, 244]; // Green-50
        let bannerBorder = [134, 239, 172]; // Green-300
        let bannerTextCol = [21, 128, 61]; // Green-700
        let bannerTitle = "VERIFIED FAITHFUL";
        let bannerSub =
            "All generated claims are substantiated by the source document without detected contradictions.";

        if (isHallucination) {
            bannerBg = [254, 242, 242]; // Red-50
            bannerBorder = [252, 165, 165]; // Red-300
            bannerTextCol = [185, 28, 28]; // Red-700
            bannerTitle = "CONTRADICTORY CLAIMS DETECTED";
            bannerSub =
                "One or more claims in the summary contradict statements in the source document.";
        } else if (isPotential) {
            bannerBg = [255, 251, 235]; // Amber-50
            bannerBorder = [252, 211, 77]; // Amber-300
            bannerTextCol = [180, 83, 9]; // Amber-700
            bannerTitle = "POTENTIAL UNVERIFIED CLAIMS";
            bannerSub =
                "Some claims could not be conclusively proven or refuted from the provided source text.";
        }

        const bannerH = 54;
        doc.setFillColor(bannerBg[0], bannerBg[1], bannerBg[2]);
        doc.setDrawColor(bannerBorder[0], bannerBorder[1], bannerBorder[2]);
        doc.roundedRect(marginX, currentY, contentWidth, bannerH, 4, 4, "FD");

        doc.setFont("helvetica", "bold");
        doc.setFontSize(9);
        doc.setTextColor(bannerTextCol[0], bannerTextCol[1], bannerTextCol[2]);
        doc.text(bannerTitle, marginX + 12, currentY + 16);

        doc.setFont("helvetica", "normal");
        doc.setFontSize(8);
        doc.setTextColor(71, 85, 105);
        doc.text(bannerSub, marginX + 12, currentY + 29);

        // Stats strip below banner text
        const ratioPercent = (
            (hallucinationAnalysis.hallucination_ratio || 0) * 100
        ).toFixed(1);
        const statsStr = `Total Claims: ${hallucinationAnalysis.total_claims ?? 0}   •   Supported: ${hallucinationAnalysis.supported_claims ?? 0}   •   Contradicted: ${hallucinationAnalysis.contradicted_claims ?? 0}   •   Unverifiable: ${hallucinationAnalysis.unverifiable_claims ?? 0}   •   Hallucination Ratio: ${ratioPercent}%`;
        doc.setFont("helvetica", "bold");
        doc.setFontSize(7.5);
        doc.setTextColor(15, 23, 42);
        doc.text(statsStr, marginX + 12, currentY + 44);

        currentY += bannerH + 16;

        // Itemized Claims Cards
        const claims = hallucinationAnalysis.sentence_analysis || [];
        if (claims.length > 0) {
            // Require enough space for BOTH the heading and the first claim card (110pt)
            // so we NEVER leave an orphan heading alone at the bottom of a page!
            checkBreak(110);
            doc.setFont("helvetica", "bold");
            doc.setFontSize(8.5);
            doc.setTextColor(100, 116, 139);
            doc.text("ITEMIZED CLAIM VERIFICATION", marginX, currentY);
            currentY += 12;

            claims.forEach((item, cIdx) => {
                const label = item.label || "UNVERIFIABLE";
                const isSupp = label === "ENTAILMENT";
                const isContra = label === "CONTRADICTION";

                // Visual theme colors per claim verdict
                let accentColor = [203, 213, 225]; // Slate-300
                let badgeBg = [241, 245, 249];
                let badgeBorder = [226, 232, 240];
                let badgeTextCol = [100, 116, 139];
                let badgeLabel = "UNVERIFIED";

                if (isSupp) {
                    accentColor = [34, 197, 94]; // Emerald-500
                    badgeBg = [240, 253, 244];
                    badgeBorder = [187, 247, 208];
                    badgeTextCol = [21, 128, 61];
                    badgeLabel = "SUPPORTED";
                } else if (isContra) {
                    accentColor = [239, 68, 68]; // Red-500
                    badgeBg = [254, 242, 242];
                    badgeBorder = [254, 202, 202];
                    badgeTextCol = [185, 28, 28];
                    badgeLabel = "CONTRADICTION";
                } else {
                    accentColor = [245, 158, 11]; // Amber-500
                    badgeBg = [255, 251, 235];
                    badgeBorder = [254, 240, 138];
                    badgeTextCol = [161, 98, 7];
                    badgeLabel = "UNVERIFIABLE";
                }

                // 1. Precise text splitting using the exact font settings
                // The claim statement gets the FULL inner card width (contentWidth - 24)
                // because the badge sits on the header row ABOVE it, preventing any overlap!
                const claimLines = splitText(item.claim, contentWidth - 24, 8.5, "bold");

                const evidenceText = item.evidence
                    ? `"${item.evidence}" (${((item.evidence_similarity || 0) * 100).toFixed(1)}% match)`
                    : null;
                const evidenceLines = evidenceText
                    ? splitText(evidenceText, contentWidth - 36, 7.5, "italic")
                    : [];

                // 2. Exact Card Height Calculation
                const cardHeaderH = 22;
                const claimH = claimLines.length * 12;
                const scoresH = 14;
                const evidenceBoxH = evidenceLines.length > 0 ? evidenceLines.length * 10.5 + 20 : 0;
                const cardPaddingBottom = 10;
                const cardH = cardHeaderH + claimH + scoresH + evidenceBoxH + cardPaddingBottom;

                checkBreak(cardH + 8);

                // 3. Card Container & Border
                doc.setFillColor(255, 255, 255);
                doc.setDrawColor(226, 232, 240);
                doc.roundedRect(marginX, currentY, contentWidth, cardH, 4, 4, "FD");

                // 4. Color-Coded Verdict Accent Bar on the left
                doc.setFillColor(accentColor[0], accentColor[1], accentColor[2]);
                doc.roundedRect(marginX, currentY, 3.5, cardH, 2, 2, "F");

                // 5. Card Header Row: Claim Number on Left, Status Badge on Right
                const headerY = currentY + 8;
                doc.setFont("helvetica", "bold");
                doc.setFontSize(7.5);
                doc.setTextColor(100, 116, 139);
                doc.text(`CLAIM ${cIdx + 1} OF ${claims.length}`, marginX + 12, headerY + 9);

                // Render Badge Pill at top right
                const badgeW = 82;
                const badgeH = 15;
                const badgeX = marginX + contentWidth - badgeW - 10;
                doc.setFillColor(badgeBg[0], badgeBg[1], badgeBg[2]);
                doc.setDrawColor(badgeBorder[0], badgeBorder[1], badgeBorder[2]);
                doc.roundedRect(badgeX, headerY, badgeW, badgeH, 3, 3, "FD");

                doc.setFont("helvetica", "bold");
                doc.setFontSize(7);
                doc.setTextColor(badgeTextCol[0], badgeTextCol[1], badgeTextCol[2]);
                doc.text(badgeLabel, badgeX + badgeW / 2, headerY + 10.5, { align: "center" });

                // 6. Claim Statement (Full width, positioned cleanly below header row)
                let textY = currentY + cardHeaderH + 6;
                doc.setFont("helvetica", "bold");
                doc.setFontSize(8.5);
                doc.setTextColor(15, 23, 42);
                claimLines.forEach((line) => {
                    doc.text(line, marginX + 12, textY);
                    textY += 12;
                });

                // 7. NLI Probability Scores
                textY += 3;
                const sEnt = ((item.scores?.entailment || 0) * 100).toFixed(1);
                const sCon = ((item.scores?.contradiction || 0) * 100).toFixed(1);
                const sUnv = ((item.scores?.unverifiable || 0) * 100).toFixed(1);
                const scoresLine = `Entailment: ${sEnt}%   •   Contradiction: ${sCon}%   •   Unverifiable: ${sUnv}%`;

                doc.setFont("helvetica", "normal");
                doc.setFontSize(7.5);
                doc.setTextColor(100, 116, 139);
                doc.text(scoresLine, marginX + 12, textY);

                // 8. Source Evidence Container (Rendered neatly if evidence was retrieved)
                if (evidenceLines.length > 0) {
                    textY += 8;
                    const evBoxY = textY;
                    const innerBoxW = contentWidth - 20;

                    // Light tinted background for retrieved evidence
                    doc.setFillColor(248, 250, 252); // Slate-50
                    doc.setDrawColor(241, 245, 249); // Slate-100
                    doc.roundedRect(marginX + 10, evBoxY, innerBoxW, evidenceBoxH, 3, 3, "FD");

                    // Subtle quotation bar on left of evidence box
                    doc.setFillColor(203, 213, 225); // Slate-300
                    doc.rect(marginX + 10, evBoxY, 2.5, evidenceBoxH, "F");

                    doc.setFont("helvetica", "bold");
                    doc.setFontSize(7);
                    doc.setTextColor(71, 85, 105);
                    doc.text("Source Evidence:", marginX + 16, evBoxY + 10);

                    doc.setFont("helvetica", "italic");
                    doc.setFontSize(7.5);
                    doc.setTextColor(51, 65, 85);
                    let evLineY = evBoxY + 20;
                    evidenceLines.forEach((evLine) => {
                        doc.text(evLine, marginX + 16, evLineY);
                        evLineY += 10.5;
                    });
                }

                currentY += cardH + 8;
            });
        }
    }

    // 8. TWO-PASS GLOBAL HEADER & FOOTER
    const totalPages = doc.getNumberOfPages();
    for (let p = 1; p <= totalPages; p++) {
        doc.setPage(p);

        // Header for pages > 1
        if (p > 1) {
            doc.setFont("helvetica", "normal");
            doc.setFontSize(8);
            doc.setTextColor(148, 163, 184);
            doc.text(
                "AI Summary & Factual Verification Report",
                marginX,
                26
            );
            doc.setDrawColor(241, 245, 249);
            doc.line(marginX, 32, marginX + contentWidth, 32);
        }

        // Global Footer (bottom margin)
        doc.setDrawColor(226, 232, 240);
        doc.line(marginX, pageHeight - 32, marginX + contentWidth, pageHeight - 32);

        doc.setFont("helvetica", "normal");
        doc.setFontSize(8);
        doc.setTextColor(148, 163, 184);
        doc.text(
            "Confidential • Verified by Natural Language Inference (DeBERTa-v3)",
            marginX,
            pageHeight - 20
        );

        const pageStr = `Page ${p} of ${totalPages}`;
        doc.text(pageStr, marginX + contentWidth, pageHeight - 20, { align: "right" });
    }

    // 9. TRIGGER DOWNLOAD WITH CLEAN FILENAME
    const safeBaseName = (sourceLabel || "Summary")
        .replace(/\.[^/.]+$/, "") // strip extension
        .replace(/[^a-zA-Z0-9_-]/g, "_")
        .slice(0, 30);
    const dateStamp = new Date().toISOString().slice(0, 10);
    const fileName = `Summary_Report_${safeBaseName}_${dateStamp}.pdf`;

    doc.save(fileName);
}
