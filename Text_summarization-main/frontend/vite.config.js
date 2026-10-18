import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            "/summarize": "http://127.0.0.1:5000",
            "/health": "http://127.0.0.1:5000",
        },
    },
    build: {
        outDir: fileURLToPath(new URL("../static/dist", import.meta.url)),
        emptyOutDir: true,
        assetsDir: "assets",
        rollupOptions: {
            output: {
                entryFileNames: "app.js",
                assetFileNames: (assetInfo) => {
                    if (assetInfo.name && assetInfo.name.endsWith(".css")) {
                        return "app.css";
                    }
                    return "assets/[name][extname]";
                },
            },
        },
    },
});
