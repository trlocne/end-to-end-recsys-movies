import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { NavBar } from "@/components/NavBar";
import { Footer } from "@/components/Footer";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "SeanMovies — discover films",
  description: "Search and personalized picks powered by your taste.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans">
        <div className="flex min-h-screen flex-col">
          <NavBar />
          <main className="mx-auto w-full max-w-container flex-1 px-6 py-8 md:px-8">
            {children}
          </main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
