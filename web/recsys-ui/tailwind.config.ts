import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#ff4f00",
        onPrimary: "#fffefb",
        ink: "#201515",
        inkSoft: "#2f2a26",
        inkMid: "#36342e",
        body: "#605d52",
        bodyMid: "#939084",
        mute: "#c5c0b1",
        canvas: "#fffefb",
        canvasSoft: "#f8f4f0",
      },
      borderRadius: {
        brand: "12px",
        input: "6px",
      },
      spacing: {
        "4xl": "64px",
        "3xl": "48px",
      },
      maxWidth: {
        container: "1280px",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        display: ["var(--font-inter)", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
