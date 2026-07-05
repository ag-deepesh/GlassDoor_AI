import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API_BASE defaults to the local FastAPI dev server (see SETUP.md).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
