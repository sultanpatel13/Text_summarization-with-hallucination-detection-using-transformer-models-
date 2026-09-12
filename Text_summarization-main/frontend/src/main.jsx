import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import "./index.css";

console.log("main.jsx: Starting React app mount");

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error("React Error:", error, errorInfo);
        this.setState({ error, errorInfo });
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{ padding: '20px', fontFamily: 'monospace', color: 'red', whiteSpace: 'pre-wrap', background: 'white', minHeight: '100vh' }}>
                    <h2>React Error Caught:</h2>
                    <p>{this.state.error && this.state.error.toString()}</p>
                    <p>{this.state.errorInfo && this.state.errorInfo.componentStack}</p>
                </div>
            );
        }
        return this.props.children;
    }
}

const rootElement = document.getElementById("root");
console.log("main.jsx: root element:", rootElement);

if (!rootElement) {
    console.error("main.jsx: Root element not found!");
    document.body.innerHTML = '<div style="padding: 20px; color: red;">Error: Root element #root not found in DOM</div>';
} else {
    console.log("main.jsx: Mounting React app");
    ReactDOM.createRoot(rootElement).render(
        <React.StrictMode>
            <ErrorBoundary>
                <App />
            </ErrorBoundary>
        </React.StrictMode>
    );
    console.log("main.jsx: React app mounted");
}