import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Auditor Kelulusan Mahasiswa",
  description: "Prediksi estimasi IPK kelulusan mahasiswa menggunakan model machine learning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
