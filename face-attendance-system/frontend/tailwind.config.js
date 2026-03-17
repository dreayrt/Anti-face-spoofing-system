/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      animation: {
        blob: "blob 7s infinite",
        "fade-in-up": "fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        scan: "scan 2s linear infinite",
        shimmer: "shimmer 2s linear infinite",
        shake: "shake 0.5s cubic-bezier(.36,.07,.19,.97) both",
      },
      keyframes: {
        blob: {
          "0%": {
            transform: "translate(0px, 0px) scale(1)",
          },
          "33%": {
            transform: "translate(30px, -50px) scale(1.1)",
          },
          "66%": {
            transform: "translate(-20px, 20px) scale(0.9)",
          },
          "100%": {
            transform: "translate(0px, 0px) scale(1)",
          },
        },
        fadeInUp: {
          "0%": {
            opacity: 0,
            transform: "translateY(20px)",
          },
          "100%": {
            opacity: 1,
            transform: "translateY(0)",
          },
        },
        scan: {
          "0%": {
            transform: "translateY(-100%)",
          },
          "100%": {
            transform: "translateY(100vh)", /* This will scan across the container if overflow is hidden */
          },
        },
        shimmer: {
          "100%": {
            transform: "translateX(100%)",
          },
        },
        shake: {
          "10%, 90%": {
            transform: "translate3d(-1px, 0, 0)",
          },
          "20%, 80%": {
            transform: "translate3d(2px, 0, 0)",
          },
          "30%, 50%, 70%": {
            transform: "translate3d(-4px, 0, 0)",
          },
          "40%, 60%": {
            transform: "translate3d(4px, 0, 0)",
          },
        },
      },
    },
  },
  plugins: [],
}
