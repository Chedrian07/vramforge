import type { Metadata, Viewport } from "next";
import Script from "next/script";

import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fine-Tuning VRAM Calculator · VRAMForge",
  description:
    "데이터셋을 자르지 않고 학습할 때 필요한 GPU 메모리를 실제 토크나이저와 전체 데이터 분석으로 산정합니다.",
  icons: { icon: "/icon.svg" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

// Applies the stored theme before first paint so the page does not flash (light/dark/system).
const THEME_INIT = `(function(){try{var p=localStorage.getItem("vf-theme");var d=p==="dark"||((p!=="light")&&window.matchMedia("(prefers-color-scheme: dark)").matches);var e=document.documentElement;if(d){e.classList.add("dark")}else{e.classList.remove("dark")}}catch(_){}})();`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <head>
        <Script id="vf-theme-init" strategy="beforeInteractive">
          {THEME_INIT}
        </Script>
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
