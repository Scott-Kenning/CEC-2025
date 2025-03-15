"use client";
import React, { useState, FormEvent } from "react";
import UploadArea from "./components/UploadArea";
import ResultDisplay from "./components/ResultDisplay";
import Link from "next/link";

interface Result {
  classification: string;
  confidence: number;
  error?: string | null;
  image?: string; // Base64-encoded PNG image of the heatmap
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleFileSelect = (file: File, preview: string) => {
    setSelectedFile(file);
    setPreviewUrl(preview);
    setResult(null);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      setLoading(true);
      const res = await fetch("/api/classify", {
        method: "POST",
        body: formData,
      });
      const data: Result = await res.json();
      console.log("Result:" , data);
      setResult(data);
    } catch (error) {
      console.error("Error uploading image:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-screen h-screen bg-neutral-300">
      {/* Main content container */}
      <div className="flex flex-col items-center justify-center h-full p-4">
        <div className="bg-white p-6 rounded shadow-md w-full max-w-md">
          <h1 className="text-2xl font-semibold mb-4 text-center">
            Brain Tumor Classifier
          </h1>
          <form onSubmit={handleSubmit} className="flex flex-col space-y-4">
            <UploadArea
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              onFileSelect={handleFileSelect}
              onRemoveFile={handleRemoveFile}
            />
            <button
              type="submit"
              disabled={loading || !selectedFile}
              className="bg-violet-600 text-white py-2 rounded font-medium hover:bg-violet-700 transition-colors disabled:opacity-50"
            >
              {loading ? "Processing..." : "Upload and Classify"}
            </button>
          </form>
          <ResultDisplay result={result} />
        </div>
        <Link href="/info" target="_blank" className="underline mt-6">
          Understanding Results →
        </Link>
      </div>
    </div>
  );
}
