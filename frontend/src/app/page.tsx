"use client";

import { useState, useEffect, useRef } from "react";
import { driver } from "driver.js";
import "driver.js/dist/driver.css";
import styles from "./page.module.css";
import { IpkGauge, ProbabilityBarChart, AuditDonutChart } from "./components/Charts";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

// Helper: Menghasilkan angka acak di rentang min - max (inklusif)
const randomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;

// Fungsi untuk menghasilkan data dummy acak yang realistis
const generateRandomStudent = () => ({
  age: randomInt(1, 3),
  sex: randomInt(1, 2),
  graduated_h_school_type: randomInt(1, 3),
  scholarship_type: randomInt(1, 5),
  additional_work: randomInt(1, 2),
  activity: randomInt(1, 2),
  partner: randomInt(1, 2),
  total_salary: randomInt(1, 5),
  transport: randomInt(1, 4),
  accomodation: randomInt(1, 4),
  mother_ed: randomInt(1, 6),
  farther_ed: randomInt(1, 6),
  siblings: randomInt(0, 5),
  parental_status: randomInt(1, 3),
  mother_occup: randomInt(1, 10),
  father_occup: randomInt(1, 10),
  weekly_study_hours: randomInt(1, 5), // 1-5 scale
  reading_non_scientific: randomInt(1, 5),
  reading_scientific: randomInt(1, 5),
  attendance_seminars_dep: randomInt(1, 5),
  impact_of_projects: randomInt(1, 5),
  attendances_classes: randomInt(1, 5),
  preparation_midterm_company: randomInt(1, 5),
  preparation_midterm_time: randomInt(1, 5),
  taking_notes: randomInt(1, 5),
  listenning: randomInt(1, 5),
  discussion_improves_interest: randomInt(1, 5),
  flip_classrom: randomInt(1, 5),
  grade_previous: randomInt(1, 5),
  grade_expected: randomInt(1, 5),
  course_id: randomInt(1, 15),
});

// Penjelasan/Saran Akademik per Kelompok IPK
const IPK_EXPLANATION: Record<number, string> = {
  0: "Berdasarkan data, mahasiswa memiliki risiko tinggi untuk tidak lulus. Diperlukan intervensi akademik yang komprehensif, evaluasi ulang kebiasaan belajar, dan bimbingan konseling yang intensif.",
  1: "Prediksi menunjukkan IPK yang sangat kurang. Mahasiswa ini sangat disarankan untuk mengikuti kelas remedial, mengubah gaya belajar, dan mendapatkan pendampingan tutor sebaya.",
  2: "Hasil prediksi menunjukkan IPK yang kurang. Mahasiswa perlu meningkatkan kehadiran di kelas, lebih aktif berdiskusi, dan mulai mengatur jadwal belajar mingguan dengan lebih disiplin.",
  3: "Prediksi IPK mahasiswa berada di batas cukup. Meski lulus, masih banyak ruang untuk perbaikan. Mahasiswa dapat didorong untuk lebih banyak membaca jurnal ilmiah dan mengurangi kegiatan di luar jika dirasa mengganggu akademik.",
  4: "Mahasiswa diprediksi lulus dengan memuaskan. Pola belajar saat ini sudah cukup baik, pertahankan konsistensi dan mulailah mengeksplorasi proyek-proyek praktis untuk menambah portofolio.",
  5: "Hasil yang sangat memuaskan. Mahasiswa memiliki peluang besar untuk lulus dengan nilai yang sangat baik. Sangat disarankan untuk mulai mencari peluang magang atau asisten peneliti.",
  6: "Prediksi menunjukkan kelulusan dengan pujian. Performa akademik mahasiswa ini sangat cemerlang. Sangat direkomendasikan untuk mengikuti kompetisi akademik atau program fast-track.",
  7: "Luar biasa! Mahasiswa diprediksi lulus dengan predikat Cum Laude (Tertinggi). Berikan dukungan maksimal untuk mempertahankan performa sempurna ini hingga akhir masa studi.",
};

// Mapping prediksi numerik ke rentang IPK
const IPK_MAP: Record<number, string> = {
  0: "0.00 - 0.99 (Tidak Lulus)",
  1: "1.00 - 1.49 (Sangat Kurang)",
  2: "1.50 - 1.99 (Kurang)",
  3: "2.00 - 2.49 (Cukup)",
  4: "2.50 - 2.99 (Memuaskan)",
  5: "3.00 - 3.49 (Sangat Memuaskan)",
  6: "3.50 - 3.74 (Dengan Pujian)",
  7: "3.75 - 4.00 (Cum Laude)",
};

export default function Home() {
  const [theme, setTheme] = useState("light");
  const [apiConnected, setApiConnected] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("random_forest.pkl");

  const [activeTab, setActiveTab] = useState("academic");
  // Mulai dengan form kosong secara default
  const [formData, setFormData] = useState<Record<string, number | "">>({});
  
  // Karena useState awal kosong, kita perlu mengisi nilai awal di useEffect agar hidrasi SSR tidak berantakan
  useEffect(() => {
    handleClear(); // Set semua field menjadi ""
  }, []);

  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<{
    prediction: number;
    probabilities: Record<string, number> | null;
    model_used: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [auditResult, setAuditResult] = useState<{
    total_records: number;
    summary: Record<string, number>;
    csv_content: string;
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ──────────────────────────── driver.js tour ────────────────────────────
  const startTour = () => {
    const driverObj = driver({
      showProgress: true,
      animate: true,
      overlayColor: "rgba(0,0,0,0.55)",
      stagePadding: 8,
      stageRadius: 10,
      popoverClass: "driverjs-theme",
      nextBtnText: "Lanjut",
      prevBtnText: "Kembali",
      doneBtnText: "Selesai",
      progressText: "Langkah {{current}} dari {{total}}",
      steps: [
        {
          element: "#header-area",
          popover: {
            title: "Selamat Datang!",
            description:
              "Ini adalah Auditor Kelulusan Mahasiswa. Sistem ini memprediksi estimasi IPK mahasiswa berdasarkan data profil dan kebiasaan belajar menggunakan model machine learning.",
          },
        },
        {
          element: "#status-pill",
          popover: {
            title: "Status Koneksi API",
            description:
              "Indikator ini menunjukkan apakah server backend FastAPI sedang aktif dan terhubung. Jika terputus, pastikan backend berjalan di port 8000.",
          },
        },
        {
          element: "#theme-toggle",
          popover: {
            title: "Ganti Tema",
            description: "Klik tombol ini untuk beralih antara mode terang dan mode gelap sesuai preferensi tampilan Anda.",
          },
        },
        {
          element: "#header-area",
          popover: {
            title: "Selamat Datang di Auditor IPK! 👋",
            description: "Aplikasi ini membantu Anda memprediksi estimasi nilai IPK kelulusan mahasiswa berdasarkan profil akademik, demografi, dan kebiasaan belajar mereka menggunakan teknologi Machine Learning cerdas.",
          },
        },
        {
          element: "#form-card",
          popover: {
            title: "1. Formulir Profil Mahasiswa",
            description: "Di area utama ini, Anda akan memasukkan data metrik mahasiswa. Semakin akurat data yang diisi, semakin presisi hasil prediksi yang akan diberikan oleh sistem.",
          },
        },
        {
          element: "#tabs-area",
          popover: {
            title: "2. Kategori Data (Tab)",
            description: "Data sangat banyak, jadi kami membaginya ke dalam 4 Tab: Profil Akademik, Demografi, Kebiasaan Belajar, dan Sosial. Pastikan Anda mengklik dan mengecek setiap tab agar tidak ada data yang terlewat!",
          },
        },
        {
          element: "#prefill-btn",
          popover: {
            title: "3. Mode Simulasi (Data Contoh)",
            description: "Malas mengisi puluhan kolom secara manual? Tenang! Klik tombol ini untuk otomatis mengisi form dengan data simulasi acak yang sangat berguna untuk pengujian cepat.",
          },
        },
        {
          element: "#clear-btn",
          popover: {
            title: "4. Reset Data",
            description: "Jika Anda ingin memulai dari kertas kosong, tekan tombol merah ini untuk menghapus seluruh isian form secara instan. Awas, tombol ini menghapus data di semua tab!",
          },
        },
        {
          element: "#model-selector",
          popover: {
            title: "5. Pilihan 'Otak' Prediksi",
            description: "Di bawah sini Anda bisa memilih algoritma Machine Learning spesifik (seperti XGBoost, Random Forest) yang bertugas menjadi 'otak' perhitungannya. Tiap model punya karakteristik akurasi yang berbeda.",
          },
        },
        {
          element: "#predict-btn",
          popover: {
            title: "6. Eksekusi Prediksi! 🚀",
            description: "Sudah yakin semua form terisi? Klik tombol ini untuk menjalankan sihir Machine Learning! Sistem akan menganalisis data Anda secara real-time.",
          },
        },
        {
          element: "#result-card",
          popover: {
            title: "7. Panel Hasil & Rekomendasi",
            description: "Voila! Hasilnya akan muncul di kotak kanan ini. Anda tidak hanya akan melihat kelompok IPK, tapi juga probabilitas kepastian model dan rekomendasi akademik khusus untuk mahasiswa tersebut.",
          },
        },
        {
          element: "#batch-card",
          popover: {
            title: "Bonus: Audit Massal 📁",
            description: "Punya ratusan data mahasiswa dalam format Excel/CSV? Tarik dan lepas file CSV Anda di sini! Sistem akan mengaudit ratusan mahasiswa sekaligus dalam hitungan detik dan memberikan laporan lengkap.",
          },
        },
      ],
    });

    driverObj.drive();
  };

  // ──────────────────────────── lifecycle ────────────────────────────
  const toggleTheme = () => {
    const nextTheme = theme === "light" ? "dark" : "light";
    setTheme(nextTheme);
    document.documentElement.setAttribute("data-theme", nextTheme);
  };

  useEffect(() => {
    async function checkStatus() {
      try {
        const res = await fetch(`${API_BASE}/`);
        if (res.ok) {
          setApiConnected(true);
          const modelsRes = await fetch(`${API_BASE}/models`);
          if (modelsRes.ok) {
            const data = await modelsRes.json();
            setModels(data.models || []);
            if (data.models && data.models.length > 0) {
              setSelectedModel(data.models[0]);
            }
          }
        } else {
          setApiConnected(false);
        }
      } catch (err) {
        setApiConnected(false);
      }
    }
    checkStatus();
  }, []);

  const handleInputChange = (field: string, val: number | "") => {
    setFormData((prev) => ({ ...prev, [field]: val }));
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validasi form: pastikan tidak ada yang kosong
    const emptyFields = Object.keys(formData).filter(key => formData[key] === "");
    if (emptyFields.length > 0) {
      setError("Harap isi semua kolom form sebelum memulai prediksi. Pastikan Anda telah mengecek semua tab (Akademik, Demografi, dsb).");
      return;
    }

    setLoading(true);
    setError(null);
    setPredictionResult(null);

    try {
      const res = await fetch(
        `${API_BASE}/predict?model_name=${selectedModel}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        }
      );

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Gagal melakukan prediksi.");
      }

      const data = await res.json();
      setPredictionResult(data);
    } catch (err: any) {
      setError(err.message || "Terjadi kesalahan koneksi.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setCsvFile(e.target.files[0]);
      setAuditResult(null);
    }
  };

  const handleCsvSubmit = async () => {
    if (!csvFile) return;
    setUploading(true);
    setError(null);

    const body = new FormData();
    body.append("file", csvFile);

    try {
      const res = await fetch(
        `${API_BASE}/audit-upload?model_name=${selectedModel}`,
        { method: "POST", body }
      );

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Audit CSV gagal.");
      }

      const data = await res.json();
      setAuditResult(data);
    } catch (err: any) {
      setError(err.message || "Gagal memproses file CSV.");
    } finally {
      setUploading(false);
    }
  };

  const downloadAuditedCsv = () => {
    if (!auditResult) return;
    const blob = new Blob([auditResult.csv_content], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `hasil_audit_${csvFile?.name || "data.csv"}`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handlePreFill = () => {
    setFormData(generateRandomStudent());
    setError(null);
    setPredictionResult(null);
  };

  const handleClear = () => {
    // Buat template object dengan semua field bernilai "" berdasarkan struktur generateRandomStudent()
    const template = generateRandomStudent();
    const cleared = Object.keys(template).reduce((acc, curr) => {
      acc[curr] = "";
      return acc;
    }, {} as Record<string, number | "">);
    
    setFormData(cleared);
    setPredictionResult(null);
    setError(null);
  };

  // ──────────────────────────── render ────────────────────────────
  return (
    <div className={styles.container}>
      {/* ───── Header ───── */}
      <header className={styles.header} id="header-area">
        <div className={styles.titleArea}>
          <h1>Auditor Kelulusan Mahasiswa</h1>
          <p>
            Prediksi estimasi IPK kelulusan mahasiswa menggunakan model machine
            learning
          </p>
        </div>
        <div className={styles.metaActions}>
          <div
            id="status-pill"
            className={`${styles.statusPill} ${apiConnected ? styles.connected : ""}`}
          >
            <span className={styles.statusIndicator}></span>
            {apiConnected ? "API Terhubung" : "API Terputus"}
          </div>
          <button
            id="tour-btn"
            className={`${styles.btn} ${styles.btnSecondary}`}
            style={{ padding: "0.375rem 0.75rem", fontSize: "0.75rem" }}
            onClick={startTour}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
            Panduan Fitur
          </button>
          <button
            id="theme-toggle"
            className={styles.themeBtn}
            onClick={toggleTheme}
            aria-label="Ganti Tema"
          >
            {theme === "light" ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
            )}
          </button>
        </div>
      </header>

      {/* ───── Main Grid ───── */}
      <div className={styles.grid}>
        {/* ── Left: Form ── */}
        <div className={styles.card} id="form-card">
          <div className={styles.cardTitle}>
            <span>Prediksi IPK Kelulusan</span>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button
                id="prefill-btn"
                className={`${styles.btn} ${styles.btnSecondary}`}
                style={{ padding: "0.25rem 0.75rem", fontSize: "0.75rem" }}
                onClick={handlePreFill}
              >
                Gunakan Data Contoh
              </button>
              <button
                id="clear-btn"
                className={`${styles.btn} ${styles.btnDanger}`}
                style={{ padding: "0.25rem 0.75rem", fontSize: "0.75rem" }}
                onClick={handleClear}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                Kosongkan Form
              </button>
            </div>
          </div>

          <div className={styles.tabs} id="tabs-area">
            <button className={`${styles.tabBtn} ${activeTab === "academic" ? styles.active : ""}`} onClick={() => setActiveTab("academic")}>Profil Akademik</button>
            <button className={`${styles.tabBtn} ${activeTab === "demographics" ? styles.active : ""}`} onClick={() => setActiveTab("demographics")}>Demografi</button>
            <button className={`${styles.tabBtn} ${activeTab === "study" ? styles.active : ""}`} onClick={() => setActiveTab("study")}>Kebiasaan Belajar</button>
            <button className={`${styles.tabBtn} ${activeTab === "social" ? styles.active : ""}`} onClick={() => setActiveTab("social")}>Sosial & Dukungan</button>
          </div>

          <form onSubmit={handlePredict}>
            {activeTab === "academic" && (
              <div className={styles.formGrid}>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>ID Mata Kuliah</label>
                  <input type="number" className={styles.input} value={formData.course_id} onChange={(e) => handleInputChange("course_id", e.target.value === "" ? "" : parseInt(e.target.value))} />
                  <span className={styles.helperText}>Kode numerik mata kuliah yang sedang ditempuh</span>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>IPK yang Diharapkan</label>
                  <select className={styles.select} value={formData.grade_expected} onChange={(e) => handleInputChange("grade_expected", parseInt(e.target.value))}>
                    <option value={1}>IPK &lt; 2.00 (Sangat Rendah)</option>
                    <option value={2}>IPK 2.00 - 2.49 (Kurang)</option>
                    <option value={3}>IPK 2.50 - 2.99 (Cukup)</option>
                    <option value={4}>IPK 3.00 - 3.49 (Baik)</option>
                    <option value={5}>IPK 3.50 - 4.00 (Sangat Baik)</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>IPK Semester Sebelumnya</label>
                  <select className={styles.select} value={formData.grade_previous} onChange={(e) => handleInputChange("grade_previous", parseInt(e.target.value))}>
                    <option value={1}>IPK &lt; 2.00 (Sangat Rendah)</option>
                    <option value={2}>IPK 2.00 - 2.49 (Kurang)</option>
                    <option value={3}>IPK 2.50 - 2.99 (Cukup)</option>
                    <option value={4}>IPK 3.00 - 3.49 (Baik)</option>
                    <option value={5}>IPK 3.50 - 4.00 (Sangat Baik)</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Jenis Beasiswa</label>
                  <select className={styles.select} value={formData.scholarship_type} onChange={(e) => handleInputChange("scholarship_type", parseInt(e.target.value))}>
                    <option value={1}>Tidak Ada Beasiswa</option>
                    <option value={2}>Potongan 25%</option>
                    <option value={3}>Potongan 50%</option>
                    <option value={4}>Potongan 75%</option>
                    <option value={5}>Beasiswa Penuh 100%</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Asal Sekolah Menengah</label>
                  <select className={styles.select} value={formData.graduated_h_school_type} onChange={(e) => handleInputChange("graduated_h_school_type", parseInt(e.target.value))}>
                    <option value={1}>Sekolah Swasta</option>
                    <option value={2}>Sekolah Negeri</option>
                    <option value={3}>Sekolah Lainnya</option>
                  </select>
                </div>
              </div>
            )}

            {activeTab === "demographics" && (
              <div className={styles.formGrid}>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Kelompok Umur</label>
                  <select className={styles.select} value={formData.age} onChange={(e) => handleInputChange("age", parseInt(e.target.value))}>
                    <option value={1}>18 - 21 Tahun</option>
                    <option value={2}>22 - 25 Tahun</option>
                    <option value={3}>26 Tahun atau Lebih</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Jenis Kelamin</label>
                  <select className={styles.select} value={formData.sex} onChange={(e) => handleInputChange("sex", parseInt(e.target.value))}>
                    <option value={1}>Perempuan</option>
                    <option value={2}>Laki-laki</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Jumlah Saudara Kandung</label>
                  <input type="number" className={styles.input} value={formData.siblings} onChange={(e) => handleInputChange("siblings", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Status Orang Tua</label>
                  <select className={styles.select} value={formData.parental_status} onChange={(e) => handleInputChange("parental_status", parseInt(e.target.value))}>
                    <option value={1}>Menikah / Bersama</option>
                    <option value={2}>Bercerai</option>
                    <option value={3}>Orang Tua Tunggal</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Status Hubungan</label>
                  <select className={styles.select} value={formData.partner} onChange={(e) => handleInputChange("partner", parseInt(e.target.value))}>
                    <option value={1}>Lajang</option>
                    <option value={2}>Memiliki Pasangan</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Tingkat Pendidikan Ibu</label>
                  <select className={styles.select} value={formData.mother_ed === "" ? "" : formData.mother_ed} onChange={(e) => handleInputChange("mother_ed", e.target.value === "" ? "" : parseInt(e.target.value))}>
                    <option value="" disabled>Pilih Pendidikan</option>
                    <option value={1}>Sekolah Dasar (SD) / Sederajat</option>
                    <option value={2}>Sekolah Menengah Pertama (SMP) / Sederajat</option>
                    <option value={3}>Sekolah Menengah Atas (SMA) / Sederajat</option>
                    <option value={4}>Diploma (D1-D3)</option>
                    <option value={5}>Sarjana (S1/D4)</option>
                    <option value={6}>Pascasarjana (S2/S3)</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Tingkat Pendidikan Ayah</label>
                  <select className={styles.select} value={formData.farther_ed === "" ? "" : formData.farther_ed} onChange={(e) => handleInputChange("farther_ed", e.target.value === "" ? "" : parseInt(e.target.value))}>
                    <option value="" disabled>Pilih Pendidikan</option>
                    <option value={1}>Sekolah Dasar (SD) / Sederajat</option>
                    <option value={2}>Sekolah Menengah Pertama (SMP) / Sederajat</option>
                    <option value={3}>Sekolah Menengah Atas (SMA) / Sederajat</option>
                    <option value={4}>Diploma (D1-D3)</option>
                    <option value={5}>Sarjana (S1/D4)</option>
                    <option value={6}>Pascasarjana (S2/S3)</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Pekerjaan Ibu</label>
                  <select className={styles.select} value={formData.mother_occup === "" ? "" : formData.mother_occup} onChange={(e) => handleInputChange("mother_occup", e.target.value === "" ? "" : parseInt(e.target.value))}>
                    <option value="" disabled>Pilih Pekerjaan</option>
                    <option value={1}>PNS / TNI / Polri</option>
                    <option value={2}>Pegawai Swasta</option>
                    <option value={3}>Wiraswasta / Pengusaha</option>
                    <option value={4}>Petani / Peternak</option>
                    <option value={5}>Pedagang</option>
                    <option value={6}>Buruh / Pekerja Harian</option>
                    <option value={7}>Profesional (Guru, Dokter, dll)</option>
                    <option value={8}>Pensiunan</option>
                    <option value={9}>Ibu Rumah Tangga / Tidak Bekerja</option>
                    <option value={10}>Lainnya</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Pekerjaan Ayah</label>
                  <select className={styles.select} value={formData.father_occup === "" ? "" : formData.father_occup} onChange={(e) => handleInputChange("father_occup", e.target.value === "" ? "" : parseInt(e.target.value))}>
                    <option value="" disabled>Pilih Pekerjaan</option>
                    <option value={1}>PNS / TNI / Polri</option>
                    <option value={2}>Pegawai Swasta</option>
                    <option value={3}>Wiraswasta / Pengusaha</option>
                    <option value={4}>Petani / Peternak</option>
                    <option value={5}>Pedagang</option>
                    <option value={6}>Buruh / Pekerja Harian</option>
                    <option value={7}>Profesional (Guru, Dokter, dll)</option>
                    <option value={8}>Pensiunan</option>
                    <option value={9}>Tidak Bekerja</option>
                    <option value={10}>Lainnya</option>
                  </select>
                </div>
              </div>
            )}

            {activeTab === "study" && (
              <div className={styles.formGrid}>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Jam Belajar per Minggu</label>
                  <input type="number" className={styles.input} value={formData.weekly_study_hours} onChange={(e) => handleInputChange("weekly_study_hours", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Membaca Non-Ilmiah</label>
                  <input type="number" className={styles.input} value={formData.reading_non_scientific} onChange={(e) => handleInputChange("reading_non_scientific", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Membaca Jurnal/Buku Ilmiah</label>
                  <input type="number" className={styles.input} value={formData.reading_scientific} onChange={(e) => handleInputChange("reading_scientific", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Kehadiran di Kelas</label>
                  <input type="number" className={styles.input} value={formData.attendances_classes} onChange={(e) => handleInputChange("attendances_classes", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Kehadiran di Seminar</label>
                  <input type="number" className={styles.input} value={formData.attendance_seminars_dep} onChange={(e) => handleInputChange("attendance_seminars_dep", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Dampak Proyek Terhadap Belajar</label>
                  <input type="number" className={styles.input} value={formData.impact_of_projects} onChange={(e) => handleInputChange("impact_of_projects", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Persiapan UTS Berkelompok</label>
                  <input type="number" className={styles.input} value={formData.preparation_midterm_company} onChange={(e) => handleInputChange("preparation_midterm_company", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Lama Persiapan UTS</label>
                  <input type="number" className={styles.input} value={formData.preparation_midterm_time} onChange={(e) => handleInputChange("preparation_midterm_time", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Rajin Mencatat Materi</label>
                  <input type="number" className={styles.input} value={formData.taking_notes} onChange={(e) => handleInputChange("taking_notes", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Kemampuan Menyimak</label>
                  <input type="number" className={styles.input} value={formData.listenning} onChange={(e) => handleInputChange("listenning", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Minat Meningkat Karena Diskusi</label>
                  <input type="number" className={styles.input} value={formData.discussion_improves_interest} onChange={(e) => handleInputChange("discussion_improves_interest", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Flipped Classroom</label>
                  <input type="number" className={styles.input} value={formData.flip_classrom} onChange={(e) => handleInputChange("flip_classrom", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
              </div>
            )}

            {activeTab === "social" && (
              <div className={styles.formGrid}>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Pekerjaan Sampingan</label>
                  <select className={styles.select} value={formData.additional_work} onChange={(e) => handleInputChange("additional_work", parseInt(e.target.value))}>
                    <option value={1}>Ada Pekerjaan Sampingan</option>
                    <option value={2}>Tidak Bekerja</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Kegiatan Organisasi / UKM</label>
                  <select className={styles.select} value={formData.activity} onChange={(e) => handleInputChange("activity", parseInt(e.target.value))}>
                    <option value={1}>Aktif Organisasi / UKM</option>
                    <option value={2}>Tidak Aktif</option>
                  </select>
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Penghasilan / Uang Saku Bulanan</label>
                  <input type="number" className={styles.input} value={formData.total_salary} onChange={(e) => handleInputChange("total_salary", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Moda Transportasi</label>
                  <input type="number" className={styles.input} value={formData.transport} onChange={(e) => handleInputChange("transport", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Jenis Tempat Tinggal</label>
                  <input type="number" className={styles.input} value={formData.accomodation} onChange={(e) => handleInputChange("accomodation", e.target.value === "" ? "" : parseInt(e.target.value))} />
                </div>
              </div>
            )}

            {/* Action Row */}
            <div className={styles.actionRow}>
              <div className={styles.modelSelectorBox} id="model-selector">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
                <label>Model ML:</label>
                <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m.replace(".pkl", "").toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>
              <button
                id="predict-btn"
                type="submit"
                disabled={loading || !apiConnected}
                className={`${styles.btn} ${styles.btnPrimary}`}
              >
                {loading ? "Menghitung..." : "Mulai Prediksi"}
              </button>
            </div>
          </form>
        </div>

        {/* ── Right: Results + Batch ── */}
        <div className={styles.sidebar}>
          {/* Hasil Prediksi */}
          <div className={styles.card} id="result-card">
            <div className={styles.cardTitle}>Hasil Prediksi IPK</div>

            {loading ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "4rem 1rem", color: "var(--brand-primary)" }}>
                <svg className={styles.spinner} xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg>
                <div style={{ marginTop: "1rem", fontWeight: "600", color: "var(--text-primary)" }}>Memproses Data...</div>
                <div style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>Model ML sedang memprediksi IPK kelulusan</div>
              </div>
            ) : predictionResult ? (
              <div className={styles.resultContainer}>
                <IpkGauge value={predictionResult.prediction} />
                <div className={styles.gradeLabel}>
                  {IPK_MAP[predictionResult.prediction] ?? `Kelompok IPK ${predictionResult.prediction}`}
                </div>
                <div className={styles.modelTag}>
                  Model: {predictionResult.model_used}
                </div>

                <div style={{
                  backgroundColor: "var(--bg-primary)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "8px",
                  padding: "1rem",
                  marginTop: "0.5rem",
                  fontSize: "0.875rem",
                  color: "var(--text-secondary)",
                  lineHeight: "1.5",
                  textAlign: "left"
                }}>
                  <div style={{ fontWeight: "600", color: "var(--text-primary)", marginBottom: "0.375rem" }}>
                    Analisis & Rekomendasi:
                  </div>
                  {IPK_EXPLANATION[predictionResult.prediction] ?? "Tidak ada data penjelasan untuk kelompok ini."}
                </div>

                {predictionResult.probabilities && (
                  <div style={{ width: "100%", marginTop: "0.5rem" }}>
                    <div style={{ fontSize: "0.8125rem", fontWeight: "600", marginBottom: "0.25rem", color: "var(--text-secondary)" }}>Distribusi Probabilitas per Kelompok IPK:</div>
                    <ProbabilityBarChart probabilities={predictionResult.probabilities} />
                  </div>
                )}
              </div>
            ) : error ? (
              <div
                style={{
                  color: "var(--error)",
                  padding: "1rem",
                  backgroundColor: "var(--error-light)",
                  borderRadius: "6px",
                  fontSize: "0.875rem",
                }}
              >
                Gagal: {error}
              </div>
            ) : (
              <div className={styles.emptyStateGuide}>
                <div style={{ marginBottom: "1rem", fontWeight: "600", color: "var(--text-primary)", fontSize: "0.95rem" }}>
                  Cara Menggunakan Auditor:
                </div>
                <ol style={{ paddingLeft: "1.25rem", color: "var(--text-secondary)", fontSize: "0.875rem", display: "flex", flexDirection: "column", gap: "0.75rem", margin: 0 }}>
                  <li>Pilih <strong>kategori tab</strong> di sebelah kiri (Akademik, Demografi, dsb).</li>
                  <li>Isi <strong>data profil</strong> mahasiswa sesuai dengan keadaan sebenarnya, atau gunakan tombol <strong>Gunakan Data Contoh</strong>.</li>
                  <li>Pilih <strong>model machine learning</strong> yang ingin digunakan di bagian bawah formulir.</li>
                  <li>Klik tombol <strong>Mulai Prediksi</strong> untuk melihat estimasi IPK kelulusan.</li>
                </ol>
              </div>
            )}
          </div>

          {/* Audit Massal */}
          <div className={styles.card} id="batch-card">
            <div className={styles.cardTitle}>Audit Dataset Massal</div>

            <div
              className={styles.dropzone}
              onClick={() => fileInputRef.current?.click()}
            >
              <svg className={styles.uploadIcon} xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
              <div className={styles.fileInfo} style={{ marginTop: "0.5rem" }}>
                {csvFile
                  ? csvFile.name
                  : "Tarik & lepas file CSV di sini, atau klik untuk memilih file"}
              </div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".csv"
                style={{ display: "none" }}
              />
            </div>

            {csvFile && (
              <div
                style={{
                  marginTop: "1rem",
                  display: "flex",
                  justifyContent: "flex-end",
                  gap: "0.5rem",
                }}
              >
                <button
                  className={`${styles.btn} ${styles.btnSecondary}`}
                  onClick={() => setCsvFile(null)}
                >
                  Batal
                </button>
                <button
                  className={`${styles.btn} ${styles.btnPrimary}`}
                  onClick={handleCsvSubmit}
                  disabled={uploading || !apiConnected}
                >
                  {uploading ? "Menganalisis..." : "Audit File CSV"}
                </button>
              </div>
            )}

            {auditResult && (
              <div className={styles.auditResults}>
                <div className={styles.downloadRow}>
                  <span>
                    Audit Selesai ({auditResult.total_records} mahasiswa)
                  </span>
                  <button
                    className={styles.btnDownload}
                    onClick={downloadAuditedCsv}
                  >
                    Unduh CSV Hasil
                  </button>
                </div>

                <div style={{ marginTop: "0.5rem" }}>
                  <div
                    style={{
                      fontSize: "0.8125rem",
                      fontWeight: "600",
                      marginBottom: "0.5rem",
                      color: "var(--text-secondary)",
                    }}
                  >
                    Distribusi Estimasi IPK Kelulusan:
                  </div>
                  <AuditDonutChart summary={auditResult.summary} total={auditResult.total_records} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
